"""Utility layer for latency-bench
================================

This module bundles **hardware discovery** and **micro-benchmark helpers** so
all higher-level scripts can rely on a single interface.

Key responsibilities
--------------------
1. Detect every compute backend available in the runtime process (CPU, CUDA
   GPU, Apple-Silicon MPS, AWS Inferentia v1/v2) and expose them as
   :class:`DeviceInfo` objects.
2. Provide small helper functions to time a *single* forward pass or to sweep
   candidate batch-sizes for a (model, device) pair.  These helpers do not try
   to be a full benchmark harness – they are intentionally lightweight so that
   the real benchmark driver can decide how many warm-ups / repeats / metrics
   to gather.

The code avoids importing heavyweight libraries (🤗 Transformers, Optimum,
Neuron SDK) unless strictly necessary.  It also protects every optional import
behind ``importlib.util.find_spec`` so that simply *importing* the module never
fails – instead the unavailable backends are silently ignored.
"""

from __future__ import annotations

import importlib.util
import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import psutil

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover – torch may be missing on CI
    torch = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    "DeviceInfo",
    "get_available_devices",
    "sweep_batch_size",
]

# ---------------------------------------------------------------------------
# ✨ Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DeviceInfo:
    """Normalized description of a compute device."""

    device_type: str  # cpu | cuda | mps | inf1 | inf2
    name: str
    vendor: str
    backend: str  # torch | onnxruntime | neuron
    device_id: Optional[int] = None  # GPU index, Neuron core index …
    total_memory_gb: Optional[float] = None
    arch: Optional[str] = None  # e.g. sm_86, arm64, graviton, ampere …

    # ───────── Convenience ─────────
    def torch_device(self):  # noqa: D401 – simple alias, not a property
        """Return a *torch.device* if the backend is PyTorch-compatible."""
        if self.backend != "torch":
            raise ValueError("This device is not a PyTorch backend → " f"{self.backend}")
        return (
            torch.device("cuda", self.device_id)
            if self.device_type == "cuda"
            else torch.device("mps")
            if self.device_type == "mps"
            else torch.device("cpu")
        )


# ---------------------------------------------------------------------------
# 🔍  Hardware discovery helpers
# ---------------------------------------------------------------------------


def _detect_cpu() -> DeviceInfo:
    """Always present – returns a single *DeviceInfo* for the host CPU."""

    cpu_info = platform.processor() or platform.machine()
    total_mem_gb = psutil.virtual_memory().total / 1024**3
    return DeviceInfo(
        device_type="cpu",
        name=cpu_info,
        vendor=platform.platform(),
        backend="torch",  # default engine for CPU
        device_id=None,
        total_memory_gb=round(total_mem_gb, 2),
        arch=platform.machine(),
    )


def _detect_cuda() -> List[DeviceInfo]:
    """Detect *all* CUDA GPUs via PyTorch (if available)."""

    if torch is None or not torch.cuda.is_available():
        return []

    gpus: List[DeviceInfo] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        gpus.append(
            DeviceInfo(
                device_type="cuda",
                name=props.name,
                vendor="NVIDIA",
                backend="torch",
                device_id=idx,
                total_memory_gb=round(props.total_memory / 1024**3, 2),
                arch=f"sm_{props.major}{props.minor}",
            )
        )
    return gpus


def _detect_mps() -> List[DeviceInfo]:
    """Detect Apple Silicon GPUs via the MPS backend."""

    if torch is None or not getattr(torch.backends, "mps", None):
        return []
    if not torch.backends.mps.is_available():
        return []

    # Apple does not expose multiple GPU IDs; treat it as a single logical device
    total_mem_gb = psutil.virtual_memory().total / 1024**3
    return [
        DeviceInfo(
            device_type="mps",
            name="Apple M-Series GPU",
            vendor="Apple",
            backend="torch",
            device_id=0,
            total_memory_gb=round(total_mem_gb, 2),
            arch="mps",
        )
    ]


def _detect_neuron() -> List[DeviceInfo]:
    """Detect AWS Inferentia/Trainium devices if the Neuron SDK is present."""

    devices: List[DeviceInfo] = []

    # Inferentia v1 (NeuronCore v1 – inf1 instances)
    if importlib.util.find_spec("torch_neuron"):
        import torch_neuron  # type: ignore # noqa: F401 – import triggers runtime init

        # *torch_neuron* does not expose per-core memory sizes → leave None
        devices.append(
            DeviceInfo(
                device_type="inf1",
                name="AWS Inferentia v1",
                vendor="AWS",
                backend="neuron",
                device_id=0,
            )
        )

    # Inferentia v2 / Trainium (NeuronCore v2 – inf2, trn1, trn1n, trn2)
    if importlib.util.find_spec("torch_neuronx"):
        import torch_neuronx  # type: ignore # noqa: F401

        # The Neuron Runtime CLI can list logical cores – fall back to *1* if it fails
        try:
            result = (
                Path("/opt/aws/neuron/bin/neu-smi").read_text()  # type: ignore
                if Path("/opt/aws/neuron/bin/neu-smi").exists()
                else "core_count:1"
            )
            core_count = int(result.strip().split(":")[-1])
        except Exception:  # noqa: BLE001 – any parsing failure → default 1 core
            core_count = 1
        for idx in range(core_count):
            devices.append(
                DeviceInfo(
                    device_type="inf2",
                    name="AWS Inferentia v2 / Trainium",
                    vendor="AWS",
                    backend="neuron",
                    device_id=idx,
                )
            )
    return devices


# Public façade --------------------------------------------------------------

def get_available_devices(refresh: bool = False) -> List[DeviceInfo]:
    """Return and cache the list of *DeviceInfo* objects for this process."""

    if not refresh and hasattr(get_available_devices, "_cache"):
        return getattr(get_available_devices, "_cache")  # type: ignore

    devices: List[DeviceInfo] = [_detect_cpu()]  # CPU is always present
    devices.extend(_detect_cuda())
    devices.extend(_detect_mps())
    devices.extend(_detect_neuron())

    logger.info("Detected %d devices: %s", len(devices), [d.device_type for d in devices])
    setattr(get_available_devices, "_cache", devices)  # simple memoization
    return devices


# ---------------------------------------------------------------------------
# 🚀  Micro-benchmark helpers
# ---------------------------------------------------------------------------

def _sync_if_needed(device: DeviceInfo):
    """Call the right backend-specific *synchronize* method (CUDA/MPS)."""

    if device.backend != "torch":
        return
    if device.device_type == "cuda":
        torch.cuda.synchronize(device.device_id)  # type: ignore[arg-type]
    elif device.device_type == "mps":  # no explicit sync API (as of PyTorch 2.2)
        pass


def time_single_forward(model, inputs, device: DeviceInfo, warmup: int = 2, repeat: int = 5) -> float:
    """Return **average latency (ms)** for a single forward pass.

    Parameters
    ----------
    model  : HuggingFace *PreTrainedModel* or *torch.nn.Module*
    inputs : Dict[str, torch.Tensor] already on the correct device
    device : Which *DeviceInfo* is running the model
    warmup : How many *extra* forward passes to ignore before timing
    repeat : How many measured repetitions to average
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for *time_single_forward*.")

    model.eval()
    with torch.no_grad():
        # Warm-up – helps stabilise GPU clocks & caches
        for _ in range(warmup):
            _ = model(**inputs)
        _sync_if_needed(device)

        elapsed: List[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(**inputs)
            _sync_if_needed(device)
            elapsed.append(time.perf_counter() - start)

    return sum(elapsed) / len(elapsed) * 1000  # → milliseconds


def sweep_batch_size(
    model, tokenizer, device: DeviceInfo, seq_len: int = 128, batch_sizes: List[int] | None = None
) -> Dict[int, float]:
    """Quickly sweep over candidate *batch_sizes* and return {bs: latency_ms}.

    This helper is intentionally simple: it only builds *random* token batches
    (no dataset decoding) and reports a single mean latency value per batch.
    The caller can decide if/when to compute throughput.
    """

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]

    if torch is None:
        raise RuntimeError("PyTorch is required for *sweep_batch_size*.")

    torch_device = device.torch_device()
    model.to(torch_device)

    # Build a dummy input once per batch-size
    results: Dict[int, float] = {}
    for bs in batch_sizes:
        dummy_input_ids = torch.randint(
            low=5, high=tokenizer.vocab_size, size=(bs, seq_len), dtype=torch.long, device=torch_device
        )
        attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.long, device=torch_device)
        inputs = {"input_ids": dummy_input_ids, "attention_mask": attention_mask}
        latency = time_single_forward(model, inputs, device)
        results[bs] = round(latency, 3)
        logger.debug("bs=%d  →  %.3f ms", bs, latency)

    return results
