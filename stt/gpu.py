"""CUDA helpers: library preloading and per-process peak VRAM sampling."""

from __future__ import annotations

import logging
import os
import threading
from types import TracebackType

log = logging.getLogger(__name__)


def preload_cuda_libs() -> None:
    """Load cuDNN/cuBLAS from the pip `nvidia-*` wheels for ONNX Runtime and CTranslate2."""
    try:
        import onnxruntime as ort

        ort.preload_dlls()
    except Exception as exc:  # optional: torch-only environments do not need it
        log.debug("onnxruntime.preload_dlls unavailable: %s", exc)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class VramPeak:
    """Context manager sampling this process's GPU memory via NVML.

    Covers every allocator in the process (PyTorch, ONNX Runtime, CTranslate2),
    unlike `torch.cuda.max_memory_allocated`.
    """

    def __init__(self, interval_s: float = 0.05, device_index: int = 0) -> None:
        self.interval_s = interval_s
        self.device_index = device_index
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _used_mb(self, nvml: object, handle: object) -> float | None:
        pid = os.getpid()
        for proc in nvml.nvmlDeviceGetComputeRunningProcesses(handle):  # type: ignore[attr-defined]
            if proc.pid == pid and proc.usedGpuMemory is not None:
                return proc.usedGpuMemory / 2**20
        return None

    def _run(self) -> None:
        try:
            import pynvml as nvml

            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:
            log.debug("NVML unavailable, VRAM peak not measured: %s", exc)
            return
        try:
            while True:
                used = self._used_mb(nvml, handle)
                if used is not None:
                    self.peak_mb = max(self.peak_mb or 0.0, used)
                if self._stop.wait(self.interval_s):
                    break
        finally:
            nvml.nvmlShutdown()

    def __enter__(self) -> VramPeak:
        self._thread = threading.Thread(target=self._run, name="vram-peak", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
