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
    subs.en.json / subs.en.srt    cues in the project subtitle schema (+ speaker, overlap)
    subs.en.speakers.srt          the same cues prefixed with speaker IDs, for review
    speakers.json                 speaker cards: ID, talk time, gender, age, embedding
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stt.refine_speakers import REDIMNET2, WESPEAKER
from stt.runner import load_json, save_json

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


def run_stage(name: str, env: str, args: list[str], out_path: Path, log_dir: Path, force: bool) -> dict[str, Any]:
    """Run a cached worker stage; returns its JSON."""
    from stt.worker import call

    if out_path.exists() and not force:
        log.info("[%s] cached: %s", name, out_path.name)
        return load_json(out_path)
    log.info("[%s] running in %s ...", name, env)
    t0 = time.perf_counter()
    data = call(env, args, out_path, log_dir / f"{name}.log")
    log.info("[%s] done in %.1fs", name, time.perf_counter() - t0)
    return data


def transcribe(args: argparse.Namespace) -> Path:
    from stt.asr import ASR_MODELS
    from stt.ocr import reclassify
    from stt.assemble import (
        CueOptions,
        assign_speakers,
        build_cues,
        collect_words,
        mark_overlaps,
        merge_captions,
        on_screen_hints,
        smooth_speakers,
        snap_speaker_boundaries,
        speaker_ids,
        subtitle_records,
    )
    from stt.diarize import DIARIZER_OPTION_DEFAULTS, DIARIZERS, Diarization, non_default_options
    from stt.textfix import collapse_repeats
    from stt.types import Segment

    media, out = Path(args.media), Path(args.out)
    if not media.is_file():
        raise SystemExit(f"Media not found: {media}")
    if args.asr not in ASR_MODELS:
        raise SystemExit(f"Unknown ASR model {args.asr!r}; known: {list(ASR_MODELS)}")
    if args.diarizer not in DIARIZERS:
        raise SystemExit(f"Unknown diarizer {args.diarizer!r}; known: {list(DIARIZERS)}")
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    stages = set(args.force or [])
    all_forced = "all" in stages or args.force == []  # bare --force recomputes everything

    duration, decoded = prepare_audio(media, out, args.start, args.duration, all_forced or "audio" in stages)
    all_forced = all_forced or decoded
    log.info("Audio: %.1f min", duration / 60)
    audio, chunks = str(out / "audio.wav"), str(out / "chunks.json")

    diar_path = out / f"diarization.{args.diarizer}.json"
    diar_args = ["diarize", "--audio", audio, "--model", args.diarizer, "--out", str(diar_path),
                 "--device", args.device]
    for flag, value in (("--num-speakers", args.num_speakers), ("--min-speakers", args.min_speakers),
                        ("--max-speakers", args.max_speakers)):
        if value:
            diar_args += [flag, str(value)]
    wanted = {"median_filter": not args.diar_no_median_filter, "exclusive": args.diar_exclusive,
              "ahc_threshold": args.diar_ahc_threshold}
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
    if diar_data["turns"] and not diar_data.get("embeddings"):  # DiariZen: prints for speakers.json / the bank
        prints_args = ["prints", "--audio", audio, "--diarization", str(diar_path), "--out", str(diar_path),
                       "--device", args.device]
        diar_data = run_stage("prints", "subs_env", prints_args, diar_path, logs, True)
    if args.refine_speakers != "none":
        refined_path = out / f"diarization.{args.diarizer}.refined.{args.refine_speakers}.json"
        refine_args = ["refine", "--audio", audio, "--diarization", str(diar_path), "--out", str(refined_path),
                       "--model", args.refine_speakers, "--margin", str(args.refine_margin), "--device", args.device]
        stale = refined_path.exists() and (
            load_json(refined_path).get("refined", {}).get("margin") != args.refine_margin
            or refined_path.stat().st_mtime < diar_path.stat().st_mtime)
        diar_data = run_stage("refine", "subs_env", refine_args, refined_path, logs,
                              diar_forced or "refine" in stages or stale)
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
    if args.fuse:
        from stt.fusion import fuse
        from stt.llm import OpenRouterClient

        hypotheses = {args.asr: [Segment.from_dict(s) for s in asr_data["segments"]]}
        for model in [m.strip() for m in args.fuse.split(",") if m.strip()]:
            hypotheses[model] = [Segment.from_dict(s) for s in asr_stage(model)[1]["segments"]]
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
    if asr_data["segments"] and not any(seg.get("words") for seg in asr_data["segments"]):
        # Text-only output (model or fusion): get word times from forced alignment.
        aligned_path = asr_path.with_name(asr_path.stem + f".aligned.{args.aligner}.json")
        align_args = ["align", "--audio", audio, "--asr", str(asr_path), "--out", str(aligned_path),
                      "--device", args.device, "--aligner", args.aligner]
        realign = asr_forced or "align" in stages or "fuse" in stages or (
            aligned_path.exists() and aligned_path.stat().st_mtime < asr_path.stat().st_mtime)  # text changed
        asr_data = run_stage("align", "subs_env" if args.aligner == "ctc" else "stt_qwen_env", align_args,
                             aligned_path, logs, realign)

    ocr_data: dict[str, Any] | None = None
    if args.ocr:
        ocr_path = out / "ocr.json"
        ocr_args = ["ocr", "--media", str(media.resolve()), "--out", str(ocr_path), "--device", args.device,
                    "--start", str(args.start), "--fps", str(args.ocr_fps)]
        ocr_args += ["--duration", str(args.duration)] if args.duration else []
        ocr_data = run_stage("ocr", "subs_env", ocr_args, ocr_path, logs, all_forced or "ocr" in stages)

    voice_path = out / f"voice.{args.diarizer}.json"
    voice_args = ["voice", "--audio", audio, "--diarization", str(diar_path), "--out", str(voice_path),
                  "--device", args.device, "--threshold", str(args.gender_threshold)]
    voice_data = run_stage("voice", "subs_env", voice_args, voice_path, logs,
                           all_forced or "voice" in stages or diar_forced)

    # ---- assemble
    diarization = Diarization.from_dict(diar_data)
    options = CueOptions(max_gap_s=args.max_gap, max_gap_mid_s=args.max_gap_mid, max_duration_s=args.max_cue_duration,
                         max_chars=args.max_chars, fragment_words=args.fragment_words, max_join_s=args.max_join)
    segments, loop_words = collapse_repeats([Segment.from_dict(s) for s in asr_data["segments"]])
    if loop_words:
        log.info("Repetition guard removed %d looped words", loop_words)
    words = collect_words(segments)
    if ocr_data:
        ocr_data = reclassify(ocr_data)  # rules may have changed since the (slow) scan
        words, ocr_stats = merge_captions(words, ocr_data["captions"])
        log.info("On-screen captions: %s", ocr_stats)
    words = assign_speakers(words, diarization, options)
    if args.word_speakers == "viterbi":
        from stt.audio import read_wav
        from stt.boundaries import RelabelParams, relabel_words
        from stt.refine_speakers import make_embedder

        params = RelabelParams(model=args.word_speakers_model, window_s=args.word_window,
                               switch_penalty=args.word_switch_penalty, prior=args.word_prior,
                               change_prior=args.word_change_prior, passes=args.word_passes)

        class _LazyEmbedder:  # the model is loaded once, and only when some embeddings are not cached
            model: Any = None

            def batch(self, windows: Any) -> Any:
                if self.model is None:
                    self.model = make_embedder(params.model, args.device)
                return self.model.batch(windows)

        words, relabel_stats = relabel_words(words, read_wav(out / "audio.wav"), _LazyEmbedder(), params,
                                             cache=out / f"words.emb.{params.model}.npz")
        log.info("[word-speakers] %s", relabel_stats)
    if args.word_smooth == "full":
        words = smooth_speakers(words, options)
    elif args.word_smooth == "snap":
        words = snap_speaker_boundaries(words, options.snap_words)
    hints = on_screen_hints(ocr_data, words, diarization) if ocr_data else {}
    cues = build_cues(words, options)
    mark_overlaps(cues, diarization)
    ids = speaker_ids(cues)
    for label in diarization.speakers:  # speakers without recognized words go last
        ids.setdefault(label, f"S{len(ids) + 1:02d}")
    voices = {v["speaker"]: v for v in voice_data["speakers"]}
    genders = {ids[label]: voices.get(label, {}).get("gender", "unknown") for label in ids}
    offset = args.start
    records = subtitle_records(cues, ids, genders, offset_s=offset)

    save_json(out / "subs.en.json", records)
    write_srt(out / "subs.en.srt", records, with_speaker=False)
    write_srt(out / "subs.en.speakers.srt", records, with_speaker=True)
    save_json(
        out / "words.json",
        [{**asdict(w), "start": round(w.start + offset, 3), "end": round(w.end + offset, 3),
          "speaker": ids.get(w.speaker) if w.speaker else None} for w in words],
    )

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
                "embedding": diarization.embeddings.get(label),
            }
        )
    save_json(out / "speakers.json", speakers)

    timings = {
        "audio_s": round(duration, 1),
        "diarize_s": diar_data.get("diarize_s"),
        "asr_s": asr_data.get("transcribe_s"),
        "align_s": asr_data.get("align_s"),
        "voice_s": voice_data.get("run_s"),
        "vram_mb": {"diarize": diar_data.get("peak_vram_mb"), "asr": asr_data.get("peak_vram_mb"),
                    "voice": voice_data.get("peak_vram_mb")},
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
