"""X-Voice cross-lingual cloning job: Russian samples of a person from an English reference clip.

    xvoice_env/bin/python -I voiceover/xvoice_worker.py job.json results.json

Runs inside ``xvoice_env`` (``bash scripts/setup_envs.sh xvoice``), started by ``stt.ru_samples``; it imports nothing
from this repository, because ``xvoice_env`` has none of ``subs_env``'s packages. The job is
``{"xvoice_repo": ..., "items": [{"ref_wav", "text", "out_wav", "seed"}, ...]}``: the model is loaded once, each item is
spoken in Russian in the voice of ``ref_wav`` (X-Voice Stage 2 in drop-text mode: the reference text is not needed;
speed predictor sets the length) and written as a 24 kHz mono float WAV. The results are
``[{"out_wav", "error", "seconds"}, ...]``. Weights: XRXRX/X-Voice (CC-BY-NC: non-commercial use only).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("voiceover.xvoice_worker")

REPO_ID = "XRXRX/X-Voice"
LANG = "ru"
RATE = 24000


def run(job: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from hydra.utils import get_class
    from importlib.resources import files
    from omegaconf import OmegaConf
    from x_voice.infer.utils_infer import (device, get_ipa_tokenizer_cache, infer_xvoice_droptext_process,
                                           load_model_sft, load_srp_model, load_vocoder, normalize_text_for_lang,
                                           preprocess_ref_audio_text)

    repo = Path(job["xvoice_repo"])
    ckpt = hf_hub_download(REPO_ID, "XVoice_Base_Stage2/model_70000.safetensors")
    srp_ckpt = hf_hub_download(REPO_ID, "SpeedPredictor/model_28000.safetensors")
    vocab = hf_hub_download(REPO_ID, "XVoice_Base_Stage1/vocab.txt")
    cfg = OmegaConf.load(repo / "src/x_voice/configs/XVoice_Base_Stage2.yaml")
    model_cls = get_class(f"x_voice.model.{cfg.model.backbone}")
    model_arc = OmegaConf.to_container(cfg.model.arch, resolve=True)
    tokenizer = cfg.model.tokenizer
    vocoder = load_vocoder("vocos", is_local=False, device=device)
    model = load_model_sft(model_cls, model_arc, ckpt, mel_spec_type="vocos", vocab_file=vocab, device=device,
                           use_total_text=False, tokenizer=tokenizer,
                           tokenizer_path=cfg.model.get("tokenizer_path", None), dataset_name=cfg.datasets.name,
                           mel_spec_kwargs=OmegaConf.to_container(cfg.model.mel_spec, resolve=True))
    lang_to_id = getattr(model.transformer, "lang_to_id", {})
    ipa_getter = get_ipa_tokenizer_cache(tokenizer, bool(cfg.model.get("stress", True)))
    srp = load_srp_model(str(files("rate_pred").joinpath("configs/SpeedPredict_Multilingual.yaml")), srp_ckpt, device)
    norm_cache: dict[str, str] = {}
    refs: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    for n, item in enumerate(job["items"], 1):
        started = time.time()
        try:
            text = normalize_text_for_lang(item["text"], LANG, norm_cache)
            if item["ref_wav"] not in refs:
                refs[item["ref_wav"]] = preprocess_ref_audio_text(item["ref_wav"],
                                                                  "Drop-text mode ignores reference text.")[0]
            torch.manual_seed(int(item["seed"]))
            torch.cuda.manual_seed_all(int(item["seed"]))
            segments, rate, _ = infer_xvoice_droptext_process(
                refs[item["ref_wav"]], [text], [[(LANG, text)]], tokenizer, ipa_getter, model, vocoder, lang_to_id,
                srp, mel_spec_type_value="vocos", target_rms_value=0.1, cross_fade_duration_value=0.15,
                nfe_step_value=32, cfg_strength_value=2.5, layered=True, cfg_strength2_value=4.0,
                cfg_schedule_value="square", cfg_decay_time_value=0.6, sway_sampling_coef_value=-1.0,
                local_speed=1.0, fix_duration_value=None, reverse=False, denoise_ref=True, loudness_norm=True,
                post_processing=True, device_name=device)
            wav = np.concatenate(segments if isinstance(segments, list) else [segments]).astype(np.float32)
            if rate != RATE:
                raise RuntimeError(f"unexpected sample rate {rate}")
            tmp = Path(item["out_wav"]).with_name(f".{Path(item['out_wav']).name}.tmp")
            sf.write(str(tmp), wav, rate, subtype="FLOAT", format="WAV")
            os.replace(tmp, item["out_wav"])
            entry: dict[str, Any] = {"out_wav": item["out_wav"], "error": None}
        except Exception as exc:  # noqa: BLE001 - recorded per item, never hidden
            entry = {"out_wav": item["out_wav"], "error": f"{type(exc).__name__}: {exc}"[:500]}
        entry["seconds"] = round(time.time() - started, 1)
        results.append(entry)
        log.info("xvoice: %d/%d (%.1f s)", n, len(job["items"]), entry["seconds"])
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
