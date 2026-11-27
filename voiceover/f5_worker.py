"""ESpeech F5-TTS synthesis job, run by ``voiceover.engine_f5`` inside ``f5_env``.

    f5_env/bin/python -m voiceover.f5_worker job.json results.json

The job lists lines: engine text (``+`` stress marks), a reference clip with its transcript, the
target length and the output WAV. Each line is synthesised at exactly that length (F5's own
length estimate counts UTF-8 bytes, so Cyrillic text would come out twice too long), heard by
GigaAM, and synthesised again with other seeds when a word is missing, garbled or drawn out
(with ``retry_lost_words`` also when a short word like "и" is not heard); the best take is kept.
``prefix`` ("— ") goes before the text and ``slack_s`` is added to the length: F5 swallows a
short first word less often then. A line with ``listen`` words (names, homographs: their stress
was checked) is also heard by TeraAnaly (``voiceover.stress_listen``) and taken again when the voice
said another stress; ``take_wav`` (a cached take not yet heard) is judged as the first take and
the new seeds follow its ``take_seed``. One model load for the whole job. Each finished line also
gets ``<key>.take.json`` next to its WAV, so an interrupted job is resumed by the engine, not redone.

Every result also has ``attempts_run`` (attempts executed; ``takes`` is the winner's number) and ``timing``
(seconds per phase, see ``voiceover.f5_stats``); the summary goes to the log and, with ``stats_path`` in the job,
to that JSON file. Reference loading is counted in ``sampling``.

Synthesis runs in the main thread (``DirectF5Synthesizer``): F5's ``infer_process`` starts a new thread for every
line, which sets up its CUDA state again each time (~0.15 s per line, 18% of the time at 32 steps). The direct path
does the same computation (loudness of the reference, its resampling, the text, ``fix_duration``, ``model.sample``,
Vocos) bit for bit, with the prepared reference kept for the whole job. A line F5 would split into several pieces
(long text for its reference) is made piece by piece in order, where F5 runs the pieces in parallel threads
that draw the random noise in any order (so such takes were not reproducible). ``compile`` in the job wraps the DiT in
``torch.compile`` (faster, slightly different audio, ~40 s to compile at the start).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from utils.atomic_io import fsync_file, write_text_atomic
from voiceover.f5_stats import LineTimer, format_summary, summarize

log = logging.getLogger("voiceover.f5_worker")

MODEL_CFG = {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
LOST_WORD_SCORE = 0.15  # per word of the text not heard (the speech check skips words under 3 letters)


@dataclass
class PreparedRef:
    """A reference clip as F5's ``infer_batch_process`` prepares it (mono, loudness raised, 24 kHz, on the device)."""

    audio: Any  # torch tensor (1, samples) at the model's rate, on the device
    rms: Any  # loudness of the original clip (0-d torch tensor, as F5 keeps it)
    ref_text: str  # transcript, with a space added after a final one-byte character (F5's rule)
    max_chars: int  # F5's piece length for this reference (``infer_process``)


class DirectF5Synthesizer:
    """F5 ``infer_process`` without its thread per line: same audio, references prepared once per job.

    ``utils`` is ``f5_tts.infer.utils_infer`` (passed in so tests can fake it)."""

    def __init__(self, model: Any, vocoder: Any, utils: Any, torch_module: Any, torchaudio_module: Any) -> None:
        self.model, self.vocoder, self.utils = model, vocoder, utils
        self.torch, self.torchaudio = torch_module, torchaudio_module
        self.refs: dict[tuple[str, str], PreparedRef] = {}
        self.direct_lines = 0
        self.split_lines = 0  # lines F5 splits into several pieces

    def prepare(self, ref_wav: str, ref_text: str) -> PreparedRef:
        key = (ref_wav, ref_text)
        if key in self.refs:
            return self.refs[key]
        torch, u = self.torch, self.utils
        audio, sr = self.torchaudio.load(ref_wav)
        max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr))
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < u.target_rms:
            audio = audio * u.target_rms / rms
        if sr != u.target_sample_rate:
            audio = self.torchaudio.transforms.Resample(sr, u.target_sample_rate)(audio)
        text = ref_text + " " if len(ref_text[-1].encode("utf-8")) == 1 else ref_text
        self.refs[key] = PreparedRef(audio.to(u.device), rms, text, max_chars)
        return self.refs[key]

    def synthesize(self, ref_wav: str, ref_text: str, gen_text: str, nfe_step: int,
                   fix_duration: float) -> tuple[Any, int]:
        """One take: ``(wave, sample_rate)``. A long text is split into pieces like F5 does, each gets its share of
        ``fix_duration`` and the pieces are cross-faded; F5 runs the pieces in parallel threads (which take the
        random numbers in any order), here they run one after another."""
        u = self.utils
        ref = self.prepare(ref_wav, ref_text)
        pieces = u.chunk_text(gen_text, max_chars=ref.max_chars)
        if not pieces:
            raise ValueError(f"F5 got no text to synthesise: {gen_text!r}")
        if len(pieces) == 1:
            self.direct_lines += 1
            return self._piece(ref, pieces[0], nfe_step, fix_duration), u.target_sample_rate
        self.split_lines += 1
        ref_sec = (ref.audio.shape[-1] // u.hop_length) * u.hop_length / u.target_sample_rate
        weights = [len(piece.encode("utf-8")) for piece in pieces]
        allocatable = fix_duration - ref_sec + u.cross_fade_duration * (len(pieces) - 1)
        waves = [self._piece(ref, piece, nfe_step, ref_sec + allocatable * w / sum(weights))
                 for piece, w in zip(pieces, weights)]
        return cross_fade(waves, int(u.cross_fade_duration * u.target_sample_rate)), u.target_sample_rate

    def _piece(self, ref: PreparedRef, text: str, nfe_step: int, fix_duration: float) -> Any:
        """F5's ``infer_batch_process._infer_basic`` with a fixed duration."""
        torch, u = self.torch, self.utils
        ref_audio_len = ref.audio.shape[-1] // u.hop_length
        duration = int(fix_duration * u.target_sample_rate / u.hop_length)
        with torch.inference_mode():
            generated, _ = self.model.sample(cond=ref.audio, text=u.convert_char_to_pinyin([ref.ref_text + text]),
                                             duration=duration, steps=nfe_step, cfg_strength=u.cfg_strength,
                                             sway_sampling_coef=u.sway_sampling_coef)
            generated = generated.to(torch.float32)[:, ref_audio_len:, :].permute(0, 2, 1)
            wave = self.vocoder.decode(generated)
            if ref.rms < u.target_rms:
                wave = wave * ref.rms / u.target_rms
            return wave.squeeze().cpu().numpy()


def cross_fade(waves: list[np.ndarray], samples: int) -> np.ndarray:
    """F5's linear cross-fade of consecutive pieces (``infer_batch_process``)."""
    final = waves[0]
    for wave in waves[1:]:
        n = min(samples, len(final), len(wave))
        if n <= 0:
            final = np.concatenate([final, wave])
            continue
        overlap = final[-n:] * np.linspace(1, 0, n) + wave[:n] * np.linspace(0, 1, n)
        final = np.concatenate([final[:-n], overlap, wave[n:]])
    return final


def compile_transformer(model: Any, torch_module: Any, compile_fn: Callable[..., Any] | None = None) -> None:
    """Wrap the DiT in ``torch.compile`` (dynamic shapes: every line has another length); compiled on first use."""
    torch_module._dynamo.config.cache_size_limit = max(64, torch_module._dynamo.config.cache_size_limit)
    model.transformer = (compile_fn or torch_module.compile)(model.transformer, dynamic=True)


def run(job: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    import torchaudio
    from f5_tts.infer import utils_infer
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT
    from huggingface_hub import hf_hub_download

    from voiceover.speech_check import (Recognizer, check_audio, check_clip, clip_score, leading_extra,
                                        missing_words, plain_words)
    from voiceover.stress_listen import STRESS_SCORE, StressListener, hear, record, wrong_words

    repo = job.get("model", "ESpeech/ESpeech-TTS-1_RL-V2")
    model = load_model(DiT, MODEL_CFG, hf_hub_download(repo, job.get("checkpoint", "espeech_tts_rlv2.pt")),
                       vocab_file=hf_hub_download(repo, "vocab.txt"))
    vocoder = load_vocoder()
    if job.get("compile"):
        compile_transformer(model, torch)
    synth = DirectF5Synthesizer(model, vocoder, utils_infer, torch, torchaudio)
    recognizer = Recognizer(job.get("device", "cuda")) if job.get("attempts", 3) > 1 else None
    prefix, slack = str(job.get("prefix", "")), float(job.get("slack_s", 0.0))
    retry_lost = bool(job.get("retry_lost_words", False))
    listener = StressListener(job.get("device", "cuda")) if recognizer and any(i.get("listen") for i in job["items"]) \
        else None
    ref_seconds: dict[str, float] = {}
    results: list[dict[str, Any]] = []
    t0 = time.time()
    for n, item in enumerate(job["items"], 1):
        timer = LineTimer()
        ref = item["ref_wav"]
        if ref not in ref_seconds:
            ref_seconds[ref] = sf.info(ref).duration
        best: tuple[float, np.ndarray, int, dict[str, Any] | None, list[dict[str, Any]]] | None = None
        listen = set(item.get("listen") or []) if listener else set()
        first_seed = int(item.get("take_seed", 1234))
        try:
            for attempt in range(int(job.get("attempts", 3))):
                if attempt == 0 and item.get("take_wav"):  # the cached take, heard before any new one
                    with timer.phase("take_read"):
                        wav, rate = sf.read(item["take_wav"], dtype="float32")
                else:
                    torch.manual_seed(first_seed + attempt)
                    nfe, fix = int(job.get("nfe", 32)), ref_seconds[ref] + float(item["duration"]) + slack
                    timer.attempts += 1
                    with timer.phase("sampling"):
                        wav, rate = synth.synthesize(ref, item["ref_text"], prefix + item["text"], nfe, fix)
                wav = np.asarray(wav, dtype=np.float32)
                check = None
                if recognizer:
                    with timer.phase("check"):
                        heard, words = recognizer(wav, rate)
                        check = check_clip({"tts_text": item["text"]}, heard, words, len(wav) / rate)
                        cut = leading_extra(item["text"], words)
                        if cut and cut > 0.15:  # the end of the reference spoken before the line: cut it off,
                            # unless that checks worse (the ASR split the first word: "Кортни" -> "корт ни")
                            cut_wav = wav[int((cut - 0.05) * rate):]
                            cut_check = check_audio(recognizer, item["text"], cut_wav, rate)
                            timer.cut_rechecks += 1
                            if clip_score(cut_check) <= clip_score(check):
                                wav, check = cut_wav, cut_check
                lost = missing_words(plain_words(item["text"]), plain_words(check["heard"])) if check else 0
                score = (clip_score(check) + (LOST_WORD_SCORE * lost if retry_lost else 0.0)) if check else 0.0
                stress: list[dict[str, Any]] = []
                if listen:
                    timer.stress_listened = True
                    with timer.phase("stress"):
                        stress = hear(listener, wav, rate, item["text"], listen)  # type: ignore[arg-type]
                    score += STRESS_SCORE * wrong_words(stress)
                if best is None or score < best[0]:
                    best = (score, wav, attempt, check, stress)
                if not check or not ({"pronunciation", "drawn_out"} & set(check["flags"]) or (retry_lost and lost)
                                     or wrong_words(stress)):
                    break
            assert best is not None
            wav_tmp = Path(item["out_wav"]).with_name(f".{Path(item['out_wav']).name}.tmp")
            with timer.phase("write"):
                sf.write(str(wav_tmp), best[1], rate, subtype="PCM_16", format="WAV")  # whole or not at all
                fsync_file(wav_tmp)
                os.replace(wav_tmp, item["out_wav"])
            entry: dict[str, Any] = {"key": item["key"], "seed": first_seed + best[2], "takes": best[2] + 1,
                                     "error": None}
            if best[3]:
                entry["check"] = {k: best[3][k] for k in ("cer", "stretch", "flags", "heard")}
            if listen:
                entry["stress"] = record(item["listen"], best[4])
            entry.update(attempts_run=timer.attempts, timing=timer.record())
            # written after the WAV: an interrupted job leaves its finished lines for the next run
            write_text_atomic(Path(item["out_wav"]).with_suffix(".take.json"), json.dumps(entry, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - recorded per line, never hidden
            entry = {"key": item["key"], "error": f"{type(exc).__name__}: {exc}"[:500], "attempts_run": timer.attempts,
                     "timing": timer.record()}
        results.append(entry)
        if n % 50 == 0:
            log.info("f5: %d/%d lines, %.1f s per line", n, len(job["items"]), (time.time() - t0) / n)
    if synth.split_lines:
        log.info("f5: %d takes made in several pieces (long text for the reference)", synth.split_lines)
    timed = [r for r in results if "timing" in r]
    if timed:
        summary = summarize([r["timing"] for r in timed])
        if torch.cuda.is_available():  # the largest video memory this worker held (PyTorch's allocator), for stt.gpu_budget
            summary["peak_vram_mb"] = {"allocated": round(torch.cuda.max_memory_allocated() / 2**20),
                                       "reserved": round(torch.cuda.max_memory_reserved() / 2**20)}
            log.info("f5 peak video memory: %s MB", summary["peak_vram_mb"])
        log.info(format_summary(summary))
        if job.get("stats_path"):
            write_text_atomic(job["stats_path"], json.dumps({"summary": summary, "lines": [
                {"key": r["key"], "error": r.get("error"), "takes": r.get("takes"), **r["timing"]} for r in timed]},
                ensure_ascii=False, indent=1))
    return results


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    results = run(job)
    write_text_atomic(args[1], json.dumps(results, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
