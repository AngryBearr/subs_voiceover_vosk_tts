"""How much video memory (VRAM) the card has, and whether two GPU jobs may run at the same time on it.

The speed-ups of the pipeline overlap GPU work: the on-screen text scan beside diarization and recognition, the mix
(voice separation) of one engine beside the voice-over of the next, the video compression beside the voicing. They
were measured on a 16 GB card. ``gpu.parallel`` in the config decides what happens on others:

* ``auto`` (default): an overlap runs only when the card is known and big enough for it with a safety margin, and
  enough memory is free at that moment; otherwise the jobs go one after another (slower, but nothing fails);
* ``on``: always overlap; ``off``: never overlap (the order before the speed-ups).

The sizes below are peaks measured on an RTX 4070 Ti SUPER (``timings.json``), or estimates where marked; the
real peaks of the jobs we had no number for are written to ``logs/vram_peaks.json`` of the episode
(``record_peak``) so that the table can be tuned later.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

MODES = ("auto", "on", "off")
MARGIN_MB = 1536  # kept free on top of the jobs' own peaks: the desktop, allocator growth, CUDA contexts

# Peak VRAM of a job in MiB. Measured: diarize, asr (Whisper large-v3; Cohere 6412), voice, ocr.
# Estimates (to be replaced by the numbers of vram_peaks.json): f5 (3.1 GB alone + speech check + stress listener),
# teratts, demucs (Hybrid Demucs, 30 s chunks), nvenc (encoder sessions + CUDA context).
PEAK_MB: dict[str, int] = {"diarize": 4700, "asr": 8600, "voice": 1800, "ocr": 500,
                           "f5": 5500, "teratts": 2000, "demucs": 3000, "nvenc": 1536}


@dataclass(frozen=True)
class Overlap:
    """A job (``own``) that may start beside other GPU jobs (``partners``); ``sequential`` says what happens if not."""

    label: str
    own: str
    partners: tuple[str, ...]
    sequential: str


OVERLAPS: dict[str, Overlap] = {
    "ocr": Overlap("OCR", "ocr", ("diarize", "asr", "voice"), "OCR after ASR, one at a time"),
    "mix": Overlap("mix", "demucs", ("f5", "teratts"), "each mix finishes before the next engine starts"),
    "compress": Overlap("video compression", "nvenc", ("f5", "teratts", "demucs"),
                        "the video is compressed in the final mux, after the voicing"),
}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def normalize_mode(value: object) -> str:
    """``auto`` / ``on`` / ``off`` from a config value (also true/false); anything else means ``auto``."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    text = str(value or "auto").strip().lower()
    if text not in MODES:
        log.warning("gpu.parallel %r is not one of %s: using auto", value, ", ".join(MODES))
        return "auto"
    return text


_total_mb: float | None = None
_total_lock = threading.Lock()


def reset_cache() -> None:
    """Forget the cached card size (for tests)."""
    global _total_mb
    with _total_lock:
        _total_mb = None


def _device_index() -> int:
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


def query_vram() -> tuple[float, float] | None:
    """(total, free) VRAM in MiB of the first GPU, or None when there is no NVIDIA card or no way to ask."""
    index = _device_index()
    try:
        out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        total, free = (float(part) for part in out.stdout.strip().splitlines()[0].split(","))
        return total, free
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        import pynvml as nvml

        nvml.nvmlInit()
        try:
            info = nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(index))
            return info.total / 2**20, info.free / 2**20
        finally:
            nvml.nvmlShutdown()
    except Exception as error:  # noqa: BLE001 - no driver, no module: the card is unknown
        log.debug("GPU memory unknown: %s", error)
        return None


def gpu_memory() -> tuple[float, float] | None:
    """(total, free) in MiB; the total is asked once, the free memory every time."""
    global _total_mb
    found = query_vram()
    if found is None:
        return None
    with _total_lock:
        if _total_mb is None:
            _total_mb = found[0]
        return _total_mb, found[1]


def _gb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB"


def decide(name: str, mode: object = "auto", *, uses_gpu: bool = True) -> Decision:
    """May overlap ``name`` (a key of ``OVERLAPS``) start now? Static rule: the card must hold the heaviest partner,
    the job itself and ``MARGIN_MB``; and what is free at this moment must hold the job and the margin."""
    spec = OVERLAPS[name]
    mode = normalize_mode(mode)
    if mode == "on":
        return Decision(True, "gpu.parallel is on")
    if mode == "off":
        return Decision(False, f"gpu.parallel is off: {spec.sequential}")
    if not uses_gpu:
        return Decision(True, f"{spec.label} does not use the GPU")
    memory = gpu_memory()
    if memory is None:
        return Decision(False, f"GPU memory unknown: {spec.sequential}")
    total, free = memory
    own = PEAK_MB[spec.own]
    required = max(PEAK_MB[p] for p in spec.partners) + own + MARGIN_MB
    if total < required:
        return Decision(False, f"{_gb(total)} GPU, {spec.label} beside the other jobs needs {_gb(required)}: "
                               f"{spec.sequential}")
    if free < own + MARGIN_MB:
        return Decision(False, f"{_gb(free)} of GPU memory free now, {spec.label} needs {_gb(own + MARGIN_MB)}: "
                               f"{spec.sequential}")
    return Decision(True, f"{_gb(total)} GPU ({_gb(free)} free): {spec.label} runs beside the other jobs")


def allow_overlap(name: str, mode: object = "auto", *, uses_gpu: bool = True) -> bool:
    """``decide`` with the reason in the log; True if the overlap may start."""
    decision = decide(name, mode, uses_gpu=uses_gpu)
    log.info("GPU overlap '%s': %s", name, decision.reason)
    return decision.allowed


_peaks_lock = threading.Lock()


def record_peak(path: Path | None, name: str, peak_mb: float | None) -> None:
    """Keep the peak VRAM of a job (MiB, the larger of the old and the new value) in the JSON file ``path``."""
    if path is None or not peak_mb:
        return
    from utils.atomic_io import write_text_atomic

    with _peaks_lock:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data[name] = round(max(float(data.get(name) or 0), float(peak_mb)))
        write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True))
    log.info("peak VRAM of %s: %d MB", name, data[name])


def is_gpu_oom(error: BaseException) -> bool:
    from stt.worker import gpu_out_of_memory

    return gpu_out_of_memory(f"{type(error).__name__}: {error}") or "out of memory" in str(error).lower()


def free_gpu_memory() -> None:
    """Hand the cached but unused video memory of this process back (best effort)."""
    gc.collect()
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:  # noqa: BLE001
        log.debug("could not empty the CUDA cache: %s", error)


def run_with_cpu_fallback(run: Callable[[str], T], what: str, *, wait_s: float = 30.0,
                          sleep: Callable[[float], None] = time.sleep) -> T:
    """``run("cuda")``; if the card is out of memory, wait ``wait_s`` and try once more, then ``run("cpu")``
    (slower, same result). Other errors are raised as they are."""
    reason = ""
    for attempt in (1, 2):
        try:
            return run("cuda")
        except Exception as error:  # noqa: BLE001
            if not is_gpu_oom(error):
                raise
            reason = f"{type(error).__name__}: {str(error)[:200]}"
        free_gpu_memory()  # outside the except block: the failed model is released first
        if attempt == 1:
            log.warning("%s ran out of GPU memory (%s); trying again in %.0f s", what, reason, wait_s)
            sleep(wait_s)
    log.warning("%s ran out of GPU memory again (%s); doing it on the CPU (slower)", what, reason)
    return run("cpu")
