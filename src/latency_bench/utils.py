"""Utility layer for latency-bench
================================

This module bundles **hardware discovery** and **micro-benchmark helpers** so
all higher-level scripts can rely on a single interface.
"""

from __future__ import annotations

import importlib.util
import logging
import math
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

# Define a reasonable, fixed batch size for hardware execution to avoid OOM errors.
HARDWARE_EXECUTION_BATCH_SIZE = 256
# Define the total number of items to simulate for a large batch job on MPS
MPS_SIMULATED_TOTAL_BATCH_SIZE = 4096

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

    def torch_device(self):
        """Return a *torch.device* if the backend is PyTorch-compatible."""
        if self.backend != "torch":
            raise ValueError(f"This device is not a PyTorch backend → {self.backend}")
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
        backend="torch",
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
    if importlib.util.find_spec("torch_neuronx"):
        try:
            import torch_xla.core.xla_model as xm
            if len(xm.get_xla_supported_devices()) > 0:
                devices.append(
                    DeviceInfo(
                        device_type="inf2",
                        name="AWS Inferentia v2 / Trainium",
                        vendor="AWS",
                        backend="neuron",
                        device_id=0,
                    )
                )
        except Exception as e:
            logger.warning("Neuron device detection failed: %s", e)
            pass
    return devices


def get_available_devices(refresh: bool = False) -> List[DeviceInfo]:
    """Return and cache the list of *DeviceInfo* objects for this process."""
    if not refresh and hasattr(get_available_devices, "_cache"):
        return getattr(get_available_devices, "_cache")
    devices: List[DeviceInfo] = [_detect_cpu()]
    devices.extend(_detect_cuda())
    devices.extend(_detect_mps())
    devices.extend(_detect_neuron())
    logger.info("Detected %d devices: %s", len(devices), [d.device_type for d in devices])
    setattr(get_available_devices, "_cache", devices)
    return devices


# ---------------------------------------------------------------------------
# 🚀  Micro-benchmark helpers
# ---------------------------------------------------------------------------

def _sync_if_needed(device: DeviceInfo):
    """Call the right backend-specific *synchronize* method (CUDA/MPS)."""
    if device.backend != "torch":
        return
    if device.device_type == "cuda":
        torch.cuda.synchronize(device.device_id)
    elif device.device_type == "mps":
        pass


def sweep_batch_size(
    model, tokenizer, device: DeviceInfo, seq_len: int = 128, batch_sizes: List[int] | None = None
) -> Dict[int, float]:
    """
    Sweep over candidate *batch_sizes* and return {bs: latency_ms}.
    This function contains special logic for MPS to simulate a large job
    using small, memory-safe mini-batches.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]

    if torch is None:
        raise RuntimeError("PyTorch is required for *sweep_batch_size*.")

    tensor_device = device.torch_device()
    model.to(tensor_device)
    model.eval()

    # --- Custom Logic for MPS ---
    if device.device_type == 'mps':
        # For MPS, we ignore the user-provided batch sizes to prevent OOM crashes.
        # Instead, we use a hardcoded list of small, safe batch sizes for the simulation.
        safe_mps_batch_sizes = [1, 2, 4, 8]
        logger.info(
            "  [MPS] Using safe batch sizes %s for simulation.", safe_mps_batch_sizes
        )
        batch_sizes_to_run = safe_mps_batch_sizes
    else:
        batch_sizes_to_run = batch_sizes


    results: Dict[int, float] = {}
    for user_batch_size in batch_sizes_to_run:
        # --- Mini-batching logic ---
        if device.device_type == 'mps':
            num_mini_batches = math.ceil(MPS_SIMULATED_TOTAL_BATCH_SIZE / user_batch_size)
            current_batch_size = user_batch_size
            logger.info(
                "  [MPS] Simulating a total job of %d items using %d mini-batches of size %d",
                MPS_SIMULATED_TOTAL_BATCH_SIZE,
                num_mini_batches,
                current_batch_size,
            )
        else:
            if user_batch_size <= HARDWARE_EXECUTION_BATCH_SIZE:
                num_mini_batches = 1
                current_batch_size = user_batch_size
            else:
                num_mini_batches = math.ceil(user_batch_size / HARDWARE_EXECUTION_BATCH_SIZE)
                current_batch_size = HARDWARE_EXECUTION_BATCH_SIZE
                logger.info(
                    "  Simulating user_batch_size=%d with %d mini-batches of %d",
                    user_batch_size,
                    num_mini_batches,
                    current_batch_size,
                )

        # Create a single dummy input tensor for one mini-batch
        dummy_input_ids = torch.randint(
            low=5, high=tokenizer.vocab_size, size=(current_batch_size, seq_len), dtype=torch.long, device=tensor_device
        )
        attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.long, device=tensor_device)
        
        inputs = {"input_ids": dummy_input_ids, "attention_mask": attention_mask}
        if 'token_type_ids' in tokenizer.model_input_names:
            inputs['token_type_ids'] = torch.zeros_like(dummy_input_ids, dtype=torch.long, device=tensor_device)

        # --- Benchmarking ---
        with torch.no_grad():
            # Warm-up runs
            for _ in range(2):
                for _ in range(num_mini_batches):
                    _ = model(**inputs)
            _sync_if_needed(device)

            # Timed runs
            total_elapsed = 0.0
            repeat = 5
            for _ in range(repeat):
                start = time.perf_counter()
                for _ in range(num_mini_batches):
                    _ = model(**inputs)
                _sync_if_needed(device)
                total_elapsed += (time.perf_counter() - start)
        
        # Calculate the average latency for the entire simulated job
        avg_total_latency_ms = (total_elapsed / repeat) * 1000
        
        # For MPS, we report the result against the simulated total size, but key it by the hardware batch size
        if device.device_type == 'mps':
            results[user_batch_size] = round(avg_total_latency_ms, 3)
            logger.info("  - HW bs=%-4d (Job Size: %d) → %.3f ms (total)", user_batch_size, MPS_SIMULATED_TOTAL_BATCH_SIZE, avg_total_latency_ms)
        else:
            results[user_batch_size] = round(avg_total_latency_ms, 3)
            logger.info("  - bs=%-4d  →  %.3f ms (total)", user_batch_size, avg_total_latency_ms)

    return results