"""ESpeech F5-TTS synthesis job, run by ``voiceover.engine_f5`` inside ``f5_env``.

    f5_env/bin/python -m voiceover.f5_worker job.json results.json

The job lists lines: engine text (``+`` stress marks), a reference clip with its transcript, the
target length and the output WAV. Each line is synthesised at exactly that length (F5's own
length estimate counts UTF-8 bytes, so Cyrillic text would come out twice too long), heard by
GigaAM, and synthesised again with other seeds when a word is missing, garbled or drawn out;
the best take is kept. One model load for the whole job. Each finished line also gets
``<key>.take.json`` next to its WAV, so an interrupted job is resumed by the engine, not redone.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

log = logging.getLogger("voiceover.f5_worker")

MODEL_CFG = {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}


def run(job: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
    from f5_tts.model import DiT
    from huggingface_hub import hf_hub_download

    from voiceover.speech_check import Recognizer, check_audio, clip_score, leading_extra

    repo = job.get("model", "ESpeech/ESpeech-TTS-1_RL-V2")
    model = load_model(DiT, MODEL_CFG, hf_hub_download(repo, job.get("checkpoint", "espeech_tts_rlv2.pt")),
                       vocab_file=hf_hub_download(repo, "vocab.txt"))
    vocoder = load_vocoder()
    recognizer = Recognizer(job.get("device", "cuda")) if job.get("attempts", 3) > 1 else None
    quiet = lambda *args, **kwargs: None  # noqa: E731
    ref_seconds: dict[str, float] = {}
    results = []
    t0 = time.time()
    for n, item in enumerate(job["items"], 1):
        ref = item["ref_wav"]
        if ref not in ref_seconds:
            ref_seconds[ref] = sf.info(ref).duration
        best: tuple[float, np.ndarray, int, dict[str, Any] | None] | None = None
        try:
            for attempt in range(int(job.get("attempts", 3))):
                torch.manual_seed(1234 + attempt)
                wav, rate, _ = infer_process(ref, item["ref_text"], item["text"], model, vocoder,
                                             nfe_step=int(job.get("nfe", 32)), show_info=quiet,
                                             fix_duration=ref_seconds[ref] + float(item["duration"]))
                wav = np.asarray(wav, dtype=np.float32)
                if recognizer:  # the end of the reference spoken before the line: cut it off
                    cut = leading_extra(item["text"], recognizer(wav, rate)[1])
                    if cut and cut > 0.15:
                        wav = wav[int((cut - 0.05) * rate):]
                check = check_audio(recognizer, item["text"], wav, rate) if recognizer else None
                score = clip_score(check) if check else 0.0
                if best is None or score < best[0]:
                    best = (score, wav, attempt, check)
                if not check or not {"pronunciation", "drawn_out"} & set(check["flags"]):
                    break
            assert best is not None
            sf.write(item["out_wav"], best[1], rate, subtype="PCM_16")
            entry: dict[str, Any] = {"key": item["key"], "seed": 1234 + best[2], "takes": best[2] + 1, "error": None}
            if best[3]:
                entry["check"] = {k: best[3][k] for k in ("cer", "stretch", "flags", "heard")}
            # written after the WAV: an interrupted job leaves its finished lines for the next run
            Path(item["out_wav"]).with_suffix(".take.json").write_text(json.dumps(entry, ensure_ascii=False),
                                                                       encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - recorded per line, never hidden
            entry = {"key": item["key"], "error": f"{type(exc).__name__}: {exc}"[:500]}
        results.append(entry)
        if n % 50 == 0:
            log.info("f5: %d/%d lines, %.1f s per line", n, len(job["items"]), (time.time() - t0) / n)
    return results


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    results = run(job)
    Path(args[1]).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
