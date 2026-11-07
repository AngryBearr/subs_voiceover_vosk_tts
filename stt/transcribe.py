"""Episode transcription: diarization + ASR + voice profiles -> speaker-attributed subtitles.

Run from the repo root (each model stage runs in its own venv, results are cached):
    subs_env/bin/python -m stt.transcribe EPISODE.mkv --out output/stt/ep01 \
        --asr whisper-large-v3 --diarizer pyannote-community-1

Outputs in --out:
    audio.wav, chunks.json        decoded 16 kHz mono audio and VAD chunks
    diarization.<diarizer>.json   speaker turns (+ exclusive turns, speaker embeddings)
    diarization.<diarizer>.refined.<model>.json   turns re-assigned by a stronger voice model
                                  (--refine-speakers, stt.refine_speakers); used for the subtitles
    asr.<model>.json              ASR segments with word timings
    voice.<diarizer>.json         gender/age per diarized speaker
    words.json                    every word with time and speaker
    onscreen.json                 with OCR: on-screen captions, name cards, lower thirds to show in Russian (stt.onscreen)
    subs.en.json / subs.en.srt    cues in the project subtitle schema (+ speaker, overlap)
    subs.en.speakers.srt          the same cues prefixed with speaker IDs, for review
    speakers.json                 speaker cards: ID, talk time, gender, age, embedding
    subtitle_hints.json           with --subtitles: what existing subtitles told about who speaks (stt.subtitle_hints)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stt import gpu_budget, progress
from stt.refine_speakers import REDIMNET2, WESPEAKER, model_cache_key
from stt.runner import load_json, save_json

if TYPE_CHECKING:
    from stt.assemble import Cue, CueOptions
    from stt.diarize import Diarization
    from stt.types import Word

log = logging.getLogger("stt.transcribe")


def _srt_time(ms: int) -> str:
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(path: Path, records: list[dict[str, Any]], *, with_speaker: bool) -> None:
    blocks = []
    for r in records:
        text = " ".join(r["text"])
        if with_speaker:
            text = f"[{r['speaker'] or '??'}{'*' if r.get('overlap') else ''}] {text}"
        blocks.append(f"{r['index']}\n{_srt_time(r['start'])} --> {_srt_time(r['end'])}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def assemble_cues(words: list[Word], diarization: Diarization, options: CueOptions, *, offset: float,
                  gender_of: Callable[[str], str], extra_labels: Iterable[str] = ()
                  ) -> tuple[list[Cue], dict[str, str], list[dict[str, Any]]]:
    """Cues, speaker ids (S01.. by first appearance, then ``extra_labels``) and subtitle records of labelled words.

    Shared by the transcription and by ``stt.season_voices``, which assigns the speakers again."""
    from stt.assemble import build_cues, mark_overlaps, speaker_ids, subtitle_records

    cues = build_cues(words, options)
    mark_overlaps(cues, diarization)
    ids = speaker_ids(cues)
    for label in extra_labels:
        ids.setdefault(label, f"S{len(ids) + 1:02d}")
    genders = {ids[label]: gender_of(label) for label in ids}
    return cues, ids, subtitle_records(cues, ids, genders, offset_s=offset)


def write_transcript(out: Path, records: list[dict[str, Any]], words: list[Word], ids: dict[str, str],
                     offset: float) -> None:
    """``subs.en.json``, ``subs.en.srt``, ``subs.en.speakers.srt`` and ``words.json`` (times shifted by ``offset``)."""
    save_json(out / "subs.en.json", records)
    write_srt(out / "subs.en.srt", records, with_speaker=False)
    write_srt(out / "subs.en.speakers.srt", records, with_speaker=True)
    save_json(
        out / "words.json",
        [{**{k: v for k, v in asdict(w).items() if k != "caption" or v},  # caption only when set
          "start": round(w.start + offset, 3), "end": round(w.end + offset, 3),
          "speaker": ids.get(w.speaker) if w.speaker else None} for w in words],
    )


def prepare_audio(media: Path, out: Path, start: float, duration: float | None, force: bool) -> tuple[float, bool]:
    """Decode the media interval to audio.wav and compute VAD chunks.

    Returns (audio duration, whether audio was (re)decoded, invalidating later stages).
    """
    from stt.audio import SAMPLE_RATE, load_audio, read_wav, save_wav
    from stt.vad import asr_chunks, vad_settings

    source = {"media": str(media.resolve()), "start": start, "duration": duration, "size": media.stat().st_size,
              "vad": vad_settings()}
    source_path, audio_path, chunks_path = out / "source.json", out / "audio.wav", out / "chunks.json"
    if not force and source_path.exists() and audio_path.exists() and chunks_path.exists():
        if load_json(source_path) == source:
            return len(read_wav(audio_path)) / SAMPLE_RATE, False
    log.info("Decoding audio from %s", media.name)
    save_wav(audio_path, load_audio(media, start=start, duration=duration))
    audio = read_wav(audio_path)
    save_json(chunks_path, [[round(s, 3), round(e, 3)] for s, e in asr_chunks(audio)])
    save_json(source_path, source)
    return len(audio) / SAMPLE_RATE, True


def run_stage(name: str, env: str, args: list[str], out_path: Path, log_dir: Path, force: bool,
              *, count_step: bool = True) -> dict[str, Any]:
    """Run a cached worker stage; returns its JSON (`count_step=False`: the caller reports the progress step)."""
    from stt.worker import call

    if out_path.exists() and not force:
        log.info("[%s] cached: %s", name, out_path.name)
        data = load_json(out_path)
    else:
        log.info("[%s] running in %s ...", name, env)
        t0 = time.perf_counter()
        data = call(env, args, out_path, log_dir / f"{name}.log")
        log.info("[%s] done in %.1fs", name, time.perf_counter() - t0)
    if count_step:
        _step_done()
    return data


def _step_done() -> None:
    """One more worker step finished (or found cached) in the progress of the stage; steps nobody planned grow the total."""
    tracker = progress.current()
    if tracker is not None:
        done = (tracker.done or 0) + 1
        progress.tick(done, max(done, tracker.total or 0), "steps")


class BackgroundStage:
    """A cached worker stage in a background thread; `join` returns its JSON or re-raises its failure.

    The worker is a subprocess with its own log file (`logs/<name>.log`), so a stage running next to the
    others never shares a log with them. Progress steps are counted by `join` in the calling thread.
    """

    def __init__(self, name: str, env: str, args: list[str], out_path: Path, log_dir: Path, force: bool) -> None:
        self.name = name
        self.cached = out_path.exists() and not force
        self.run_s = 0.0  # wall time of the run in the thread (0 when cached)
        self.wait_s = 0.0  # how long `join` blocked the caller
        self._data: dict[str, Any] | None = None
        self._error: BaseException | None = None

        def target() -> None:
            t0 = time.perf_counter()
            try:
                self._data = run_stage(name, env, args, out_path, log_dir, force, count_step=False)
            except BaseException as exc:  # re-raised in join(), as if the stage had run inline
                self._error = exc
            self.run_s = 0.0 if self.cached else time.perf_counter() - t0

        self._thread = threading.Thread(target=target, name=f"stage-{name}", daemon=True)
        self._thread.start()

    def wait(self) -> None:
        """Block until the thread ends, without looking at the result."""
        t0 = time.perf_counter()
        self._thread.join()
        self.wait_s += time.perf_counter() - t0

    def join(self) -> dict[str, Any]:
        self.wait()
        if self._error is not None:
            raise self._error
        assert self._data is not None
        _step_done()
        return self._data

    @property
    def hidden_s(self) -> float:
        """Seconds of the stage that ran behind other work (not waited for)."""
        return max(0.0, self.run_s - self.wait_s)


def transcribe(args: argparse.Namespace) -> Path:
    from stt.asr import ASR_MODELS
    from stt.diarize import DIARIZERS

    wall0 = time.perf_counter()
    media, out = Path(args.media), Path(args.out)
    if not media.is_file():
        raise SystemExit(f"Media not found: {media}")
    if args.asr not in ASR_MODELS:
        raise SystemExit(f"Unknown ASR model {args.asr!r}; known: {list(ASR_MODELS)}")
    if args.diarizer not in DIARIZERS:
        raise SystemExit(f"Unknown diarizer {args.diarizer!r}; known: {list(DIARIZERS)}")
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    # worker steps: speakers, [refinement], recognition (+ models fused), [on-screen text], voices
    progress.tick(0, 2 + (args.refine_speakers != "none") + 1 + len([m for m in (args.fuse or "").split(",") if m.strip()])
                  + bool(args.ocr), "steps")
    stages = set(args.force or [])
    all_forced = "all" in stages or args.force == []  # bare --force recomputes everything

    duration, decoded = prepare_audio(media, out, args.start, args.duration, all_forced or "audio" in stages)
    all_forced = all_forced or decoded
    log.info("Audio: %.1f min", duration / 60)
    audio, chunks = str(out / "audio.wav"), str(out / "chunks.json")

    # On-screen text needs only the video: scan it in the background while the GPU models run in turn, if the card
    # has room for it beside them (gpu.parallel); otherwise it runs after the alignment, as before.
    ocr_job: BackgroundStage | None = None
    ocr_inline: Callable[[], dict[str, Any]] | None = None
    ocr_overlapped = False
    if args.ocr:
        ocr_path = out / "ocr.json"
        ocr_args = ["ocr", "--media", str(media.resolve()), "--out", str(ocr_path), "--device", args.device,
                    "--start", str(args.start), "--fps", str(args.ocr_fps)]
        ocr_args += ["--duration", str(args.duration)] if args.duration else []
        ocr_forced = all_forced or "ocr" in stages
        # a cached scan costs nothing: it is simply read; the card is asked only when the scan has to run
        ocr_overlapped = (ocr_path.exists() and not ocr_forced) or gpu_budget.allow_overlap(
            "ocr", getattr(args, "gpu_parallel", "auto"), uses_gpu=args.device != "cpu")
        if ocr_overlapped:
            ocr_job = BackgroundStage("ocr", "subs_env", ocr_args, ocr_path, logs, ocr_forced)
        else:
            ocr_inline = lambda: run_stage("ocr", "subs_env", ocr_args, ocr_path, logs, ocr_forced)  # noqa: E731
    try:
        return _transcribe_steps(args, media, out, logs, duration, audio, chunks, stages, all_forced,
                                 ocr_job, wall0, ocr_inline)
    finally:
        if ocr_job is not None:
            ocr_job.wait()  # never leave the scan running after a failure elsewhere


def _transcribe_steps(args: argparse.Namespace, media: Path, out: Path, logs: Path, duration: float, audio: str,
                      chunks: str, stages: set[str], all_forced: bool, ocr_job: BackgroundStage | None,
                      wall0: float, ocr_inline: Callable[[], dict[str, Any]] | None = None) -> Path:
    """Everything after the audio is decoded and the background scan has started (or, with ``ocr_inline``, is to run
    after the alignment)."""
    from stt.asr import ASR_MODELS
    from stt.ocr import reclassify
    from stt.assemble import (
        CueOptions,
        assign_speakers,
        collect_words,
        merge_captions,
        on_screen_hints,
        smooth_speakers,
        snap_speaker_boundaries,
    )
    from stt.diarize import DIARIZER_OPTION_DEFAULTS, DIARIZERS, Diarization, local_model_dir, non_default_options
    from stt.textfix import collapse_repeats
    from stt.types import Segment

    diar_path = out / f"diarization.{args.diarizer}.json"
    diar_args = ["diarize", "--audio", audio, "--model", args.diarizer, "--out", str(diar_path),
                 "--device", args.device]
    for flag, value in (("--num-speakers", args.num_speakers), ("--min-speakers", args.min_speakers),
                        ("--max-speakers", args.max_speakers)):
        if value:
            diar_args += [flag, str(value)]
    wanted = {"median_filter": not args.diar_no_median_filter, "exclusive": args.diar_exclusive,
              "ahc_threshold": args.diar_ahc_threshold}
    local_dir = local_model_dir(DIARIZERS[args.diarizer])
    if local_dir:  # a different checkpoint must not reuse the cached turns
        wanted["model_dir"] = local_dir
    if args.diar_ahc_threshold != DIARIZER_OPTION_DEFAULTS["ahc_threshold"]:
        diar_args += ["--ahc-threshold", str(args.diar_ahc_threshold)]
    if args.diar_no_median_filter:
        diar_args.append("--no-median-filter")
    if args.diar_exclusive:
        diar_args.append("--exclusive")
    options_changed = diar_path.exists() and (
        load_json(diar_path).get("options") or {}) != non_default_options(wanted)
    if options_changed:
        log.info("[diarize] options differ from the cached ones; re-running with %s", non_default_options(wanted))
    diar_forced = all_forced or "diarize" in stages or options_changed
    diar_data = run_stage("diarize", DIARIZERS[args.diarizer].env, diar_args, diar_path, logs, diar_forced)
    diarize_s, prints_s, refine_s = diar_data.get("diarize_s"), None, None
    if diar_data["turns"] and not diar_data.get("embeddings"):  # DiariZen: prints for speakers.json / the bank
        prints_args = ["prints", "--audio", audio, "--diarization", str(diar_path), "--out", str(diar_path),
                       "--device", args.device]
        diar_data = run_stage("prints", "subs_env", prints_args, diar_path, logs, True)
        prints_s = diar_data.get("prints_s")
    if args.refine_speakers != "none":
        refined_path = out / f"diarization.{args.diarizer}.refined.{model_cache_key(args.refine_speakers)}.json"
        refine_args = ["refine", "--audio", audio, "--diarization", str(diar_path), "--out", str(refined_path),
                       "--model", args.refine_speakers, "--margin", str(args.refine_margin), "--device", args.device]
        stale = refined_path.exists() and (
            load_json(refined_path).get("refined", {}).get("margin") != args.refine_margin
            or refined_path.stat().st_mtime < diar_path.stat().st_mtime)
        diar_data = run_stage("refine", "subs_env", refine_args, refined_path, logs,
                              diar_forced or "refine" in stages or stale)
        refine_s = diar_data.get("refine_s")
        log.info("[refine] %s", diar_data.get("refined"))

    hotwords = None
    if args.hotwords_file:
        raw = re.split(r"[,\n]", Path(args.hotwords_file).read_text(encoding="utf-8"))
        hotwords = ", ".join(w.strip() for w in raw if w.strip())
    asr_forced = all_forced or "asr" in stages

    def asr_stage(model: str) -> tuple[Path, dict[str, Any]]:
        path = out / f"asr.{model}{'@hw' if hotwords else ''}.json"  # biased results cached separately
        stage_args = ["asr", "--audio", audio, "--chunks", chunks, "--model", model, "--out", str(path),
                      "--device", args.device]
        if hotwords:
            stage_args += ["--hotwords", hotwords]
        return path, run_stage(f"asr.{model}", ASR_MODELS[model].env, stage_args, path, logs, asr_forced)

    asr_path, asr_data = asr_stage(args.asr)
    asr_models: dict[str, dict[str, Any]] = {}  # per model: times from its own json (the fused/aligned one has none)

    def note_asr(model: str, data: dict[str, Any]) -> None:
        asr_models[model] = {k: data.get(k) for k in ("load_s", "warmup_s", "transcribe_s", "peak_vram_mb")}

    note_asr(args.asr, asr_data)
    fuse_s: float | None = None
    reference_path = asr_path if any(seg.get("words") for seg in asr_data["segments"]) else None  # timed words anchor CTC times
    if args.fuse:
        from stt.fusion import fuse
        from stt.llm import OpenRouterClient

        hypotheses = {args.asr: [Segment.from_dict(s) for s in asr_data["segments"]]}
        for model in [m.strip() for m in args.fuse.split(",") if m.strip()]:
            model_path, model_data = asr_stage(model)
            hypotheses[model] = [Segment.from_dict(s) for s in model_data["segments"]]
            note_asr(model, model_data)
            if reference_path is None and any(seg.get("words") for seg in model_data["segments"]):
                reference_path = model_path  # Whisper: the first recognition with word times
        asr_path = out / "asr.fused.json"
        refuse = (asr_forced or "fuse" in stages or not asr_path.exists()
                  or load_json(asr_path).get("sources") != list(hypotheses))  # other models than cached
        if refuse:
            glossary = Path(args.glossary).read_text(encoding="utf-8") if args.glossary else ""
            client = OpenRouterClient(args.fuse_model)
            chunk_list = [tuple(c) for c in load_json(Path(chunks))]
            t0 = time.perf_counter()
            fused = fuse(chunk_list, hypotheses, client, glossary=glossary)
            save_json(asr_path, {"model": "fused", "sources": list(hypotheses), "llm": args.fuse_model,
                                 "fuse_s": round(time.perf_counter() - t0, 1), "usage": client.usage.to_dict(),
                                 "segments": [seg.to_dict() for seg in fused]})
            log.info("[fuse] %s -> %d chunks, %s", ", ".join(hypotheses), len(fused), client.usage.to_dict())
        asr_data = load_json(asr_path)
        fuse_s = asr_data.get("fuse_s")
    if asr_data["segments"] and not any(seg.get("words") for seg in asr_data["segments"]):
        # Text-only output (model or fusion): get word times from forced alignment.
        aligned_path = asr_path.with_name(asr_path.stem + f".aligned.{args.aligner}.json")
        align_args = ["align", "--audio", audio, "--asr", str(asr_path), "--out", str(aligned_path),
                      "--device", args.device, "--aligner", args.aligner]
        if args.fuse and args.aligner == "ctc" and reference_path is not None:  # Whisper's times anchor drifted CTC times
            align_args += ["--reference", str(reference_path)]
        realign = asr_forced or "align" in stages or "fuse" in stages or (
            aligned_path.exists() and aligned_path.stat().st_mtime < asr_path.stat().st_mtime)  # text changed
        asr_data = run_stage("align", "subs_env" if args.aligner == "ctc" else "stt_qwen_env", align_args,
                             aligned_path, logs, realign)

    ocr_data: dict[str, Any] | None = ocr_inline() if ocr_inline is not None else None  # one at a time

    voice_path = out / f"voice.{args.diarizer}.json"
    voice_args = ["voice", "--audio", audio, "--diarization", str(diar_path), "--out", str(voice_path),
                  "--device", args.device, "--threshold", str(args.gender_threshold)]
    voice_data = run_stage("voice", "subs_env", voice_args, voice_path, logs,
                           all_forced or "voice" in stages or diar_forced)

    if ocr_job is not None:
        ocr_data = ocr_job.join()

    # ---- assemble
    diarization = Diarization.from_dict(diar_data)
    options = CueOptions(max_gap_s=args.max_gap, max_gap_mid_s=args.max_gap_mid, max_duration_s=args.max_cue_duration,
                         max_chars=args.max_chars, fragment_words=args.fragment_words, max_join_s=args.max_join)
    segments, loop_words = collapse_repeats([Segment.from_dict(s) for s in asr_data["segments"]])
    if loop_words:
        log.info("Repetition guard removed %d looped words", loop_words)
    words = collect_words(segments)
    screen_items: list[dict[str, Any]] = []
    if ocr_data:
        # rules may have changed since the (slow) scan; the spoken words help to repair the OCR text
        ocr_data = reclassify(ocr_data, [w.text for w in words])
        caption_decisions: list[dict[str, Any]] = []
        words, ocr_stats = merge_captions(words, ocr_data["captions"], decisions=caption_decisions)
        log.info("On-screen captions: %s", ocr_stats)
        from stt.ocr_repair import proper_names
        from stt.onscreen import collect

        screen_items = collect(ocr_data, caption_decisions, offset=args.start,
                               names=proper_names(w.text for w in words))
        save_json(out / "onscreen.json", screen_items)
        log.info("On-screen items: %d -> %s", len(screen_items), out / "onscreen.json")
    elif (out / "onscreen.json").exists():
        (out / "onscreen.json").unlink()  # left over from a run with OCR
    words = assign_speakers(words, diarization, options)
    own_labels = [w.speaker for w in words]
    word_hints: Any = None  # existing subtitles (--subtitles): who speaks, pinned to the words
    sub_info: Any = None
    hints_path = out / "subtitle_hints.json"
    if args.subtitles:
        from stt.subtitle_hints import build_hints

        word_hints, sub_info = build_hints(Path(args.subtitles), words, args.start)
    elif hints_path.exists():
        hints_path.unlink()  # left over from a run with subtitles
    if args.word_speakers == "viterbi":
        from stt.audio import read_wav
        from stt.boundaries import RelabelParams, relabel_words
        from stt.refine_speakers import make_embedder

        params = RelabelParams(model=args.word_speakers_model, window_s=args.word_window,
                               switch_penalty=args.word_switch_penalty, prior=args.word_prior,
                               change_prior=args.word_change_prior, passes=args.word_passes,
                               turn_bonus=args.subtitle_turn_bonus)

        class _LazyEmbedder:  # the model is loaded once, and only when some embeddings are not cached
            model: Any = None

            def batch(self, windows: Any) -> Any:
                if self.model is None:
                    self.model = make_embedder(params.model, args.device)
                return self.model.batch(windows)

        words, relabel_stats = relabel_words(words, read_wav(out / "audio.wav"), _LazyEmbedder(), params,
                                             cache=out / f"words.emb.{model_cache_key(params.model)}.npz", hints=word_hints)
        log.info("[word-speakers] %s", relabel_stats)
    elif word_hints is not None and word_hints.persons:  # no Viterbi: named lines still take their person's state
        from stt.boundaries import plan_states, restore_fixed

        plan_states(words, own_labels, word_hints, {s for s in own_labels if s is not None})
        words = restore_fixed(words, word_hints.fixed)
    if args.word_smooth == "full":
        words = smooth_speakers(words, options)
    elif args.word_smooth == "snap":
        words = snap_speaker_boundaries(words, options.snap_words)
    if word_hints is not None and word_hints.fixed:
        from stt.boundaries import restore_fixed

        words = restore_fixed(words, word_hints.fixed)  # smoothing must not undo a named line
    hints = (on_screen_hints(ocr_data, words, diarization, screen_items=screen_items, offset=args.start or 0.0)
             if ocr_data else {})
    named_by_subtitles: dict[str, str] = {}
    if word_hints is not None and sub_info is not None:
        from stt.subtitle_hints import evidence, report, subtitle_persons

        for state, items in evidence(word_hints, sub_info).items():
            hints[state] = items + hints.get(state, [])
        named_by_subtitles = subtitle_persons(word_hints, sub_info)
        save_json(hints_path, report(word_hints, sub_info, words, own_labels))
    voices = {v["speaker"]: v for v in voice_data["speakers"]}
    offset = args.start
    cues, ids, records = assemble_cues(words, diarization, options, offset=offset,
                                       gender_of=lambda label: voices.get(label, {}).get("gender", "unknown"),
                                       extra_labels=diarization.speakers)  # speakers without recognized words go last
    write_transcript(out, records, words, ids, offset)

    speakers = []
    for label, sid in sorted(ids.items(), key=lambda kv: kv[1]):
        own = [c for c in cues if c.speaker == label]
        voice = voices.get(label, {})
        speakers.append(
            {
                "id": sid,
                "diarizer_label": label,
                "name": None,
                "gender": voice.get("gender", "unknown"),
                "gender_source": "voice" if voice.get("gender", "unknown") != "unknown" else None,
                "gender_conf": voice.get("gender_conf"),
                "age": voice.get("age"),
                "p_child": voice.get("p_child"),
                "speech_s": voice.get("speech_s"),
                "clean_s": voice.get("clean_s"),
                "cues": len(own),
                "words": sum(len(c.words) for c in own),
                "first_seen": round(own[0].start + offset, 3) if own else None,
                "sample": own[0].text[:120] if own else None,
                "on_screen": hints.get(label, []),
                **({"subtitle_person": named_by_subtitles[label]} if label in named_by_subtitles else {}),
                "embedding": diarization.embeddings.get(label),
            }
        )
    save_json(out / "speakers.json", speakers)

    ocr_s = ocr_data.get("ocr_s") if ocr_data else None
    transcribe_s = [m["transcribe_s"] for m in asr_models.values() if m["transcribe_s"] is not None]
    timings = {
        "audio_s": round(duration, 1),
        "diarize_s": diarize_s,
        "prints_s": prints_s,
        "refine_s": refine_s,
        "asr_s": round(sum(transcribe_s), 3) if transcribe_s else None,  # inference of all models, loading apart
        "asr_models": asr_models,
        "fuse_s": fuse_s,
        "align_s": asr_data.get("align_s"),
        "ocr_s": ocr_s,
        "voice_s": voice_data.get("run_s"),
        "ocr_hidden_s": round(ocr_job.hidden_s, 1) if ocr_job is not None else 0.0,  # OCR time behind other steps
        "ocr_parallel": ocr_job is not None,  # False: after the alignment (gpu.parallel, or no OCR)
        "transcribe_wall_s": round(time.perf_counter() - wall0, 1),
        "vram_mb": {"diarize": diar_data.get("peak_vram_mb"),
                    "asr": max((m["peak_vram_mb"] or 0 for m in asr_models.values()), default=0) or None,
                    "voice": voice_data.get("peak_vram_mb"), "ocr": ocr_data.get("peak_vram_mb") if ocr_data else None},
    }
    save_json(out / "timings.json", timings)
    unassigned = sum(1 for r in records if r["speaker"] is None)
    log.info(
        "%d cues, %d speakers (%s), %d overlapped, %d without speaker -> %s",
        len(records), len(speakers), ", ".join(f"{s['id']}:{s['gender'][0]}" for s in speakers),
        sum(r["overlap"] for r in records), unassigned, out / "subs.en.json",
    )
    return out / "subs.en.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Speaker-attributed English transcription of a video/audio file")
    parser.add_argument("media", help="Video or audio file")
    parser.add_argument("--out", required=True, help="Output directory (stage results are cached here)")
    parser.add_argument("--asr", default="whisper-large-v3", help="ASR model (see stt/asr.py ASR_MODELS)")
    parser.add_argument("--diarizer", default="pyannote-community-1", help="Diarizer (see stt/diarize.py DIARIZERS)")
    parser.add_argument("--diar-no-median-filter", action="store_true",
                        help="DiariZen: no median filter on the segmentation (sharper speaker boundaries)")
    parser.add_argument("--diar-exclusive", action="store_true",
                        help="DiariZen: also derive exclusive (one speaker per frame) turns")
    parser.add_argument("--diar-ahc-threshold", type=float, default=0.5,
                        help="DiariZen: AHC threshold before VBx (lower: more clusters)")
    parser.add_argument("--start", type=float, default=0.0, help="Process from this time, seconds")
    parser.add_argument("--duration", type=float, default=None, help="Process this many seconds")
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers, if known")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gender-threshold", type=float, default=0.7, help="Min confidence, else 'unknown'")
    parser.add_argument("--refine-speakers", default="none", choices=["none", "pyannote", *WESPEAKER, *REDIMNET2],
                        help="Re-assign diarized turns that sound clearly like another speaker (stt.refine_speakers)")
    parser.add_argument("--refine-margin", type=float, default=0.1,
                        help="How much closer to the other speaker a turn must sound to move")
    parser.add_argument("--word-speakers", default="none", choices=["none", "viterbi"],
                        help="Relabel words by a Viterbi pass over per-word speaker embeddings (stt.boundaries)")
    parser.add_argument("--word-speakers-model", default="resnet293",
                        choices=["pyannote", *WESPEAKER, *REDIMNET2], help="Embedding model for --word-speakers")
    parser.add_argument("--word-switch-penalty", type=float, default=0.3,
                        help="Cost of a different speaker on consecutive words")
    parser.add_argument("--word-prior", type=float, default=0.0, help="Bonus for the diarizer's own label")
    parser.add_argument("--word-change-prior", type=float, default=0.15,
                        help="Bonus for a speaker change where the diarizer has one")
    parser.add_argument("--word-window", type=float, default=1.0, help="Audio window per word, seconds")
    parser.add_argument("--word-passes", type=int, default=1, help="Passes re-estimating centroids")
    parser.add_argument("--word-smooth", default="full", choices=["full", "snap", "none"],
                        help="Sentence-level speaker smoothing after the word labels: majority + snapping "
                             "(default), snapping to sentence ends only, or none")
    parser.add_argument("--subtitles", default=None,
                        help="Existing English subtitles of the episode (stt.subtitles.find_subtitles): named lines "
                             "(\"NAME:\") fix who speaks, \">>\" marks reward a speaker change there")
    parser.add_argument("--subtitle-turn-bonus", type=float, default=0.3,
                        help="Bonus for a speaker change where the subtitles open another voice (>> marks)")
    parser.add_argument("--hotwords-file", default=None,
                        help="Names/terms for ASR biasing (e.g. hotwords.txt from stt.glossary)")
    parser.add_argument("--fuse", default=None,
                        help="Extra ASR models to fuse with --asr by an LLM, e.g. cohere-transcribe,qwen3-asr-1.7b")
    parser.add_argument("--fuse-model", default="openai/gpt-6-luna", help="OpenRouter model for --fuse")
    parser.add_argument("--glossary", default=None, help="glossary.txt (stt.glossary) used by --fuse")
    parser.add_argument("--aligner", default="ctc", choices=["ctc", "qwen"],
                        help="Word times for text-only ASR/fusion: wav2vec2 CTC (default) or Qwen3-ForcedAligner")
    parser.add_argument("--no-ocr", dest="ocr", action="store_false",
                        help="Skip on-screen text (captions for hard-to-hear speech, name cards)")
    parser.add_argument("--gpu-parallel", choices=list(gpu_budget.MODES), default="auto",
                        help="Scan on-screen text beside the GPU models: auto = only if the card has room for it "
                             "(else one at a time), on = always, off = never")
    parser.add_argument("--ocr-fps", type=float, default=2.0, help="Frames per second scanned for on-screen text")
    parser.add_argument("--max-gap", type=float, default=0.8, help="Pause (s) after a sentence that starts a new cue")
    parser.add_argument("--max-gap-mid", type=float, default=1.0, help="Pause (s) inside a sentence that starts a new cue")
    parser.add_argument("--fragment-words", type=int, default=2,
                        help="Keep a phrase together over a pause (up to --max-join) that would split off this many "
                             "words or fewer (0: off)")
    parser.add_argument("--max-join", type=float, default=4.0, help="Longest pause (s) --fragment-words bridges")
    parser.add_argument("--max-cue-duration", type=float, default=7.0)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument(
        "--force",
        nargs="*",
        choices=["all", "audio", "diarize", "asr", "fuse", "align", "ocr", "voice"],
        help="Recompute these cached stages (bare --force: all)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    transcribe(args)


if __name__ == "__main__":
    main()
