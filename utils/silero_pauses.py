"""Fixed, listened-to mono Vosk policy; VAD agreement is not speech-safety proof.

Input is legacy PRETEMPO (silence reduction, then resampling). Output is a
distinct, one-pass PAUSES WAV; neither source samples nor tempo are modified.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import importlib.resources as resources
import json
import logging
import os
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

Interval = tuple[int, int]
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
POLICY = {"id": "silero-energy-internal-200-150-v1", "package": "silero-vad", "version": "6.2.3",
          "model_sha256": MODEL_SHA256, "backend": "bundled-onnx-opset16-cpu",
          "threshold": 0.5, "sampling_rate": 16000, "min_speech_duration_ms": 250,
          "min_silence_duration_ms": 100, "speech_pad_ms": 30,
          "energy_dbfs": -50, "energy_frame_ms": 10, "guard_after_ms": 40,
          "guard_before_ms": 20, "minimum_core_ms": 200, "retain_core_ms": 150}
PARAMETERS = {key: POLICY[key] for key in ("threshold", "sampling_rate", "min_speech_duration_ms",
                                         "min_silence_duration_ms", "speech_pad_ms")}
EDGE_POLICY = {"id": "inspection-edge-union-v1", "rms_dbfs": -60, "diagnostic_dbfs": -50,
               "frame_ms": 10, "normal_head_ms": 40, "normal_tail_ms": 80,
               "short_head_ms": 100, "short_tail_ms": 150, "short_words_max": 2,
               "short_duration_ms": 1500, "minimum_side_cut_ms": 40,
               "uncertainty_ms": 40, "no_speech_or_no_activity": "keep_entire_clip",
               "uncertain_edge": "keep_that_edge", "channels": "mono_only_no_downmix"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_identity() -> dict[str, Any]:
    """Inspect installed assets only; never download or import audio backends."""
    try:
        version = metadata.version("silero-vad")
        if version != POLICY["version"]:
            raise RuntimeError(f"Expected silero-vad==6.2.3, found {version}")
        asset = Path(str(resources.files("silero_vad.data").joinpath("silero_vad.onnx")))
        if digest(asset) != MODEL_SHA256:
            raise RuntimeError("Bundled Silero ONNX hash mismatch")
        distribution = metadata.distribution("silero-vad")
        return {"policy": dict(POLICY), "asset": str(asset), "helper_sha256": digest(Path(__file__)),
                "official_code": {name: digest(Path(distribution.locate_file(f"silero_vad/{name}")))
                                  for name in ("model.py", "utils_vad.py")}}
    except (ImportError, FileNotFoundError, metadata.PackageNotFoundError) as exc:
        raise RuntimeError("Silero pauses enabled but pinned local Silero 6.2.3/ONNX runtime is unavailable; "
                           "install explicitly or disable silero_pauses.enabled (no automatic downloads)") from exc


def validate_intervals(runs: list[Interval], n: int) -> None:
    previous = 0
    for a, b in runs:
        if type(a) is not int or type(b) is not int or not 0 <= previous <= a < b <= n:
            raise ValueError("Ordered disjoint bounded integer sample intervals required")
        previous = b


def map_speech(spans: list[dict[str, int]], sr: int, n: int, detector_n: int) -> list[Interval]:
    """Floor/ceil mapping protects speech; clamp bounds and unite touching spans."""
    if sr <= 0 or n <= 0 or detector_n <= 0:
        raise ValueError("Positive rates and sample counts required")
    source = [(s["start"], s["end"]) for s in spans]
    validate_intervals(source, detector_n)
    result: list[Interval] = []
    for start, end in source:
        a, b = min(n, start * sr // 16000), min(n, (end * sr + 15999) // 16000)
        if b <= a:
            raise ValueError("Mapped speech cannot be empty")
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def quiet_cores(pcm: np.ndarray, sr: int) -> list[Interval]:
    """Any active channel protects its 10 ms frame, plus 40/20 ms guards."""
    data = pcm.astype(np.float64) / 32768.0
    n, frame = len(data), max(1, round(sr * 0.01))
    quiet = [bool(np.max(np.sqrt(np.mean(data[a:a + frame] ** 2, axis=0))) < 10 ** (-50 / 20))
             for a in range(0, n, frame)]
    active = [i for i, silent in enumerate(quiet) if not silent]
    if not active:
        return []
    first, last = active[0] * frame, min(n, (active[-1] + 1) * frame)
    tail, head = (sr * 40 + 999) // 1000, (sr * 20 + 999) // 1000
    result: list[Interval] = []
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        a = i * frame
        while i < len(quiet) and quiet[i]:
            i += 1
        b = min(n, i * frame)
        if a >= first and b <= last and b - a > tail + head:
            result.append((a + tail, b - head))
    return result


def intersect(left: list[Interval], right: list[Interval], n: int) -> list[Interval]:
    validate_intervals(left, n)
    validate_intervals(right, n)
    result: list[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if a < b:
            result.append((a, b))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def cut_pcm(pcm: np.ndarray, cuts: list[Interval]) -> np.ndarray:
    validate_pcm(pcm, 1)
    validate_intervals(cuts, len(pcm))
    pieces: list[np.ndarray] = []
    cursor = 0
    for a, b in cuts:
        pieces.append(pcm[cursor:a])
        cursor = b
    pieces.append(pcm[cursor:])
    return np.concatenate(pieces)


def validate_pcm(pcm: np.ndarray, sr: int) -> None:
    if sr <= 0 or pcm.dtype != np.int16 or pcm.ndim != 2 or not len(pcm) or not pcm.shape[1]:
        raise ValueError("Nonempty PCM16 sample/channel array and positive rate required")


def detector_audio(pcm: np.ndarray, sr: int) -> np.ndarray:
    validate_pcm(pcm, sr)
    if pcm.shape[1] != 1:
        raise ValueError("Silero pauses require mono PCM16; stereo downmix/cancellation is prohibited")
    audio = soxr.resample(pcm[:, 0].astype(np.float32) / 32768.0, sr, 16000, quality="HQ")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Finite nonempty detector input required")
    return np.asarray(audio, dtype=np.float32)


def agreed_cuts(pcm: np.ndarray, sr: int, spans: list[dict[str, int]], detector_n: int
                ) -> tuple[list[Interval], list[Interval], list[Interval]]:
    validate_pcm(pcm, sr)
    speech = map_speech(spans, sr, len(pcm), detector_n)
    gaps = [(left[1], right[0]) for left, right in zip(speech, speech[1:]) if left[1] < right[0]]
    agreed = intersect(quiet_cores(pcm, sr), gaps, len(pcm))
    retained = (sr * 150 + 999) // 1000
    cuts = [(a + retained // 2, b - (retained - retained // 2)) for a, b in agreed
            if (b - a) * 1000 >= 200 * sr and b - a > retained]
    return cuts, speech, agreed


def normalized_edge_text(text: str) -> str:
    """Original chosen text, punctuation removed and whitespace collapsed only."""
    return " ".join("".join(c if not unicodedata.category(c).startswith("P") else " "
                            for c in text).split())


def edge_text_binding(text: str | None) -> dict[str, Any]:
    from utils.pipeline_text import text_sha256
    normalized = normalized_edge_text(text) if text is not None else None
    return {"chosen_text_sha256": text_sha256(text) if text is not None else None,
            "punctuation_only_text": normalized,
            "punctuation_only_sha256": text_sha256(normalized) if normalized is not None else None}


def edge_samples(ms: int, sr: int) -> int:
    return (ms * sr + 999) // 1000


def protected_after_cuts(speech: list[Interval], cuts: list[Interval], n: int) -> list[Interval]:
    """Shift padded speech through internal cuts without rounding or lost samples."""
    validate_intervals(speech, n)
    validate_intervals(cuts, n)
    mapped: list[Interval] = []
    for a, b in speech:
        if any(a < y and b > x for x, y in cuts):
            raise ValueError("Internal cut overlaps protected speech")
        shift = sum(y - x for x, y in cuts if y <= a)
        mapped.append((a - shift, b - shift))
    validate_intervals(mapped, n - sum(b - a for a, b in cuts))
    return mapped


def edge_activity(pcm: np.ndarray, sr: int, dbfs: int) -> dict[str, Any]:
    validate_pcm(pcm, sr)
    data = pcm.astype(np.float64) / 32768
    frame, threshold = max(1, edge_samples(10, sr)), 10 ** (dbfs / 20)
    active = [(a, min(len(pcm), a + frame)) for a in range(0, len(pcm), frame)
              if float(np.max(np.sqrt(np.mean(data[a:a + frame] ** 2, axis=0)))) >= threshold]
    return {"dbfs": dbfs, "frame_samples": frame, "active_intervals": active,
            "first": active[0][0] if active else None, "last": active[-1][1] if active else None,
            "active_frame_samples": sum(b - a for a, b in active),
            "amplitude_at_or_above_threshold_channel_samples": int(np.count_nonzero(np.abs(data) >= threshold))}


def edge_plan(pcm: np.ndarray, sr: int, speech: list[Interval], short: bool) -> dict[str, Any]:
    """Exactly the eight approved previews; not a word/phoneme safety guarantee."""
    validate_pcm(pcm, sr)
    validate_intervals(speech, len(pcm))
    n = len(pcm)
    primary, diagnostic = edge_activity(pcm, sr, -60), edge_activity(pcm, sr, -50)
    nonzero = np.flatnonzero(np.any(pcm != 0, axis=1))
    result: dict[str, Any] = {"primary": primary, "diagnostic": diagnostic,
        "exact_zero_head_samples": int(nonzero[0]) if len(nonzero) else n,
        "exact_zero_tail_samples": n - int(nonzero[-1]) - 1 if len(nonzero) else n,
        "nonzero_channel_samples": int(np.count_nonzero(pcm)),
        "head_guard_samples": edge_samples(100 if short else 40, sr),
        "tail_guard_samples": edge_samples(150 if short else 80, sr),
        "left_samples": 0, "right_samples": 0, "left_uncertain": False, "right_uncertain": False,
        "left_reason": "", "right_reason": "", "skip_reason": None}
    reason = ("unsupported_stereo_no_downmix" if pcm.shape[1] != 1 else
              "no_padded_vad_speech" if not speech else
              "all_rms_quiet_or_no_activity" if primary["first"] is None else None)
    if reason:
        result.update(skip_reason=reason, left_reason=reason, right_reason=reason)
        return result
    first, last = min(primary["first"], speech[0][0]), max(primary["last"], speech[-1][1])
    result.update(protected_first=first, protected_last=last)
    candidates = {"left": max(0, first - result["head_guard_samples"]),
                  "right": max(0, n - last - result["tail_guard_samples"])}
    for side, boundary in (("left", "first"), ("right", "last")):
        delta = (abs(primary[boundary] - diagnostic[boundary]) if diagnostic[boundary] is not None else None)
        uncertain = delta is None or delta * 1000 > 40 * sr
        result[side + "_threshold_delta_samples"] = delta
        result[side + "_uncertain"] = uncertain
        result[side + "_candidate_samples"] = candidates[side]
        reason = ("threshold_uncertain_keep_edge" if uncertain else
                  "below_minimum_40ms_keep_edge" if candidates[side] * 1000 < 40 * sr else
                  "guarded_union_cut")
        result[side + "_reason"] = reason
        if reason == "guarded_union_cut":
            result[side + "_samples"] = candidates[side]
    left, right = result["left_samples"], result["right_samples"]
    assert 0 <= left <= first <= last <= n - right <= n
    result.update(retained_rms_head_quiet_samples=primary["first"] - left,
                  retained_rms_tail_quiet_samples=n - right - primary["last"],
                  retained_union_head_samples=first - left, retained_union_tail_samples=n - right - last)
    return result


def process_edges(source: Path, output: Path, internal: dict[str, Any], *, available_ms: int,
                  chosen_text: str | None) -> dict[str, Any]:
    """Conditional, one-pass outer cuts using this file's existing internal receipt."""
    from utils.pipeline_text import json_sha256
    if type(available_ms) is not int:
        raise ValueError("Canonical edge window must be integer milliseconds")
    receipt_path = output.with_suffix(".edges.json")
    if (source.resolve() == output.resolve() or output.exists() or receipt_path.exists()
            or source.with_suffix(".edges.json").exists()):
        raise ValueError("Edge stage already attempted/applied or not distinct; use fresh output")
    if (internal.get("status") != "complete" or Path(internal["output_path"]).resolve() != source.resolve()
            or digest(source) != internal["output_sha256"]
            or digest(Path(internal["source_path"])) != internal["source_sha256"]):
        raise ValueError("Edge stage requires current source-bound internal receipt")
    if sf.info(source).subtype != "PCM_16":
        raise ValueError("Edge stage requires PCM_16 WAV")
    pcm, sr = sf.read(source, dtype="int16", always_2d=True)
    validate_pcm(pcm, sr)
    if sr != internal["sample_rate"] or len(pcm) != internal["after_frames"]:
        raise ValueError("Edge source PCM differs from internal receipt")
    speech = protected_after_cuts([tuple(s) for s in internal["speech_source_samples"]],
        [tuple(c) for c in internal["cut_intervals_source_samples"]], internal["before_frames"])
    if len(pcm) != internal["before_frames"] - sum(b - a for a, b in internal["cut_intervals_source_samples"]):
        raise ValueError("Internal sample counts do not conserve PCM")
    binding = edge_text_binding(chosen_text)
    # Missing pre-normalization text uses the larger guards, never accent-based words.
    short = (chosen_text is None or len(binding["punctuation_only_text"].split()) <= 2
             or internal["before_frames"] * 1000 <= 1500 * sr)
    receipt: dict[str, Any] = {"status": "started", "policy": dict(EDGE_POLICY),
        "internal_receipt_sha256": json_sha256(internal), "source_path": str(source.resolve()),
        "source_sha256": digest(source), "output_path": str(output.resolve()),
        "available_ms": available_ms, **binding, "short_protection": short,
        "before_frames": len(pcm), "sample_rate": sr, "channels": pcm.shape[1],
        "speech_source_samples": speech, "new_vad_passes": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    try:
        skip = ("invalid_window_keep_clip" if available_ms <= 0 else
                "already_fits" if len(pcm) * 1000 <= available_ms * sr else None)
        plan = ({"left_samples": 0, "right_samples": 0, "skip_reason": skip} if skip else
                edge_plan(pcm, sr, speech, short))
        left, right = plan["left_samples"], plan["right_samples"]
        after = pcm[left:len(pcm) - right]
        for a, b in speech + plan.get("primary", {}).get("active_intervals", []):
            if not left <= a < b <= len(pcm) - right:
                raise ValueError("Edge cut overlaps protected speech/activity")
        with tempfile.TemporaryDirectory(dir=output.parent) as directory:
            temporary = Path(directory) / "edges.wav"
            if left or right:
                sf.write(temporary, after, sr, format="WAV", subtype="PCM_16")
            else:
                shutil.copyfile(source, temporary)
            actual, actual_sr = sf.read(temporary, dtype="int16", always_2d=True)
            if actual_sr != sr or not np.array_equal(actual, after) or digest(source) != receipt["source_sha256"]:
                raise ValueError("Edge PCM/source verification failed")
            os.link(temporary, output)
        receipt.update(status="complete", decision=skip or ("applied" if left or right else "no_eligible_edge"),
                       plan=plan, after_frames=len(after), removed_frames=left + right,
                       saving_ms=(left + right) * 1000 / sr, output_sha256=digest(output))
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        temporary_receipt = receipt_path.with_suffix(".json.tmp")
        temporary_receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary_receipt.replace(receipt_path)
    logging.info("Silero edges %s: %s; removed %d frames / %.3f ms; window=%d ms",
                 source.name, receipt["decision"], receipt["removed_frames"], receipt["saving_ms"], available_ms)
    return receipt


def validate_edge_receipt(receipt: dict[str, Any], internal: dict[str, Any], *, available_ms: int,
                          chosen_text: str | None) -> None:
    """Current immutable stages plus exact original-text/window decision binding."""
    from utils.pipeline_text import json_sha256
    binding = edge_text_binding(chosen_text)
    if (receipt.get("status") != "complete" or receipt.get("policy") != EDGE_POLICY
            or receipt.get("internal_receipt_sha256") != json_sha256(internal)
            or receipt.get("available_ms") != available_ms
            or any(receipt.get(key) != value for key, value in binding.items())
            or Path(receipt["source_path"]).resolve() != Path(internal["output_path"]).resolve()
            or receipt["source_sha256"] != internal["output_sha256"]
            or not Path(receipt["source_path"]).is_file() or not Path(receipt["output_path"]).is_file()
            or digest(Path(receipt["source_path"])) != receipt["source_sha256"]
            or digest(Path(receipt["output_path"])) != receipt["output_sha256"]):
        raise ValueError("Full-track export blocked: stale edge text/window/processing provenance")
    info = sf.info(receipt["output_path"])
    if (info.frames != receipt["after_frames"] or info.samplerate != receipt["sample_rate"]
            or info.channels != receipt["channels"] or info.subtype != "PCM_16"
            or receipt["before_frames"] != internal["after_frames"]
            or receipt["before_frames"] != receipt["after_frames"] + receipt["removed_frames"]):
        raise ValueError("Full-track export blocked: stale edge PCM measurement")


class PauseProcessor:
    """One sequential ONNX instance; reset on every file, no inference cache."""

    def __init__(self) -> None:
        self.identity = runtime_identity()
        try:
            from silero_vad import load_silero_vad
            self.model: Any = load_silero_vad(onnx=True, opset_version=16)
        except (ImportError, OSError) as exc:
            raise RuntimeError("Silero pauses enabled but local ONNX/torch runtime cannot load") from exc
        if self.model.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Silero pauses require CPUExecutionProvider only")

    def detect(self, audio: np.ndarray) -> list[dict[str, int]]:
        import torch
        from silero_vad import get_speech_timestamps
        self.model.reset_states()
        return get_speech_timestamps(torch.from_numpy(audio), self.model, **PARAMETERS,
                                     return_seconds=False, visualize_probs=False)

    def process(self, source: Path, output: Path) -> dict[str, Any]:
        """Distinct exclusive output and intent receipt prevent repeating cuts."""
        if source.resolve() == output.resolve():
            raise ValueError("Silero pauses require distinct immutable PRETEMPO and PAUSES paths")
        receipt_path = output.with_suffix(".pauses.json")
        if output.exists() or receipt_path.exists() or source.with_suffix(".pauses.json").exists():
            raise ValueError("Pause stage already attempted/applied; use fresh PRETEMPO and output")
        if sf.info(source).subtype != "PCM_16":
            raise ValueError("Silero pauses require PCM_16 WAV")
        pcm, sr = sf.read(source, dtype="int16", always_2d=True)
        detector = detector_audio(pcm, sr)
        output.parent.mkdir(parents=True, exist_ok=True)
        receipt: dict[str, Any] = {"status": "started", "identity": self.identity,
            "source_path": str(source.resolve()), "source_sha256": digest(source), "output_path": str(output.resolve()),
            "before_frames": len(pcm), "sample_rate": sr, "channels": pcm.shape[1],
            "detector_frames": len(detector), "detector_float32_sha256": hashlib.sha256(detector.tobytes()).hexdigest(),
            "sequence_passes": 1}
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
        try:
            spans = self.detect(detector)
            cuts, speech, agreed = agreed_cuts(pcm, sr, spans, len(detector))
            after = cut_pcm(pcm, cuts)
            with tempfile.TemporaryDirectory(dir=output.parent) as directory:
                temporary = Path(directory) / "pauses.wav"
                if cuts:
                    sf.write(temporary, after, sr, format="WAV", subtype="PCM_16")
                else:
                    shutil.copyfile(source, temporary)
                actual, actual_sr = sf.read(temporary, dtype="int16", always_2d=True)
                if actual_sr != sr or not np.array_equal(actual, after) or digest(source) != receipt["source_sha256"]:
                    raise ValueError("Pause PCM/source verification failed")
                # Atomic exclusive publication: never overwrite another attempt.
                os.link(temporary, output)
            receipt.update(status="complete", after_frames=len(after), removed_frames=len(pcm) - len(after),
                           saving_ms=(len(pcm) - len(after)) * 1000 / sr,
                           speech_16k_samples=spans, speech_source_samples=speech, agreed_core_samples=agreed,
                           cut_intervals_source_samples=cuts, output_sha256=digest(output))
        except Exception as exc:
            receipt.update(status="failed", error=str(exc))
            raise
        finally:
            temporary_receipt = receipt_path.with_suffix(".json.tmp")
            temporary_receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
            temporary_receipt.replace(receipt_path)
        logging.info("Silero pauses %s: removed %d frames / %.3f ms", source.name,
                     receipt["removed_frames"], receipt["saving_ms"])
        return receipt


def process_segments(items: list[dict[str, Any]], segments: Path, stage: Path,
                     *, file_prefix: str = "", budget_items: list[dict[str, Any]] | None = None,
                     max_total_speed: float = 1.5,
                     edge_receipts: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Fresh CLI/UI directories only; retain pre-tempo stage copies and receipts."""
    if stage.exists():
        raise ValueError("Pause stage already exists; use a new run directory")
    budgets: dict[int, int] = {}
    chosen: dict[int, str] = {}
    if edge_receipts is not None:
        from utils.analyze_text import join_text_lines
        from utils.shortening_plan import PlannerConfig, plan_subtitles
        if budget_items is None or [(x.get("index"), x.get("start"), x.get("end")) for x in budget_items] != [
                (x.get("index"), x.get("start"), x.get("end")) for x in items]:
            raise ValueError("Edge stage requires full canonical budget/source items before normalization")
        budgets = {item["index"]: plan.available_ms for item, plan in zip(
            budget_items, plan_subtitles(budget_items, PlannerConfig(max_speed=max_total_speed)))}
        chosen = {row["index"]: join_text_lines(row.get("text", "")) for row in budget_items}
    processor = PauseProcessor()
    stage.mkdir(parents=True)
    receipts: list[dict[str, Any]] = []
    for item in items:
        name = f"{file_prefix}{item['index']}.wav"
        source = segments / name
        if not source.exists():
            continue  # The following bounded timing check records and blocks missing cues.
        pretempo = stage / f"{item['index']}.pretempo.wav"
        shutil.copyfile(source, pretempo)
        output = stage / f"{item['index']}.pauses.wav"
        internal = processor.process(pretempo, output)
        receipts.append(internal)
        if edge_receipts is not None:
            edge_output = stage / f"{item['index']}.edges.wav"
            edge_receipts.append(process_edges(output, edge_output, internal,
                available_ms=budgets[item["index"]], chosen_text=chosen[item["index"]]))
            output = edge_output
        with tempfile.TemporaryDirectory(dir=segments) as directory:
            replacement = Path(directory) / name
            shutil.copyfile(output, replacement)
            replacement.replace(source)
    return receipts


def validate_profile(config: dict[str, Any]) -> None:
    """Approval covers the selected model/voice only, not other default voices."""
    internal_enabled = config.get("silero_pauses", {}).get("enabled", False)
    if config.get("silero_edges", {}).get("enabled", False) and not internal_enabled:
        raise ValueError("silero_edges.enabled requires silero_pauses.enabled; no extra VAD pass is allowed")
    if not internal_enabled:
        return
    settings = config.get("tts", {})
    voices = [settings.get("voice", 0)] + [settings[key] for key in ("voice_male", "voice_female")
                                          if settings.get(key) is not None]
    if any(voice != 8 for voice in voices) or Path(str(settings.get("model_path", ""))).name != "vosk-model-tts-ru-0.10-multi":
        raise ValueError("Silero pauses approval covers vosk-model-tts-ru-0.10-multi voice 8 only; "
                         "disable silero_pauses.enabled for other profiles")
