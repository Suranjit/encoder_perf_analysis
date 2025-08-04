"""Model-loading helpers for *latency-bench*.

This wrapper hides the messy details of loading the same **encoder model** on
several very different runtimes:

* **PyTorch / CUDA** – regular `transformers`.
* **PyTorch / Apple-Silicon MPS** – same, just `.to("mps")`.
* **PyTorch / CPU** – likewise, but chooses `torch.float32` unless BF16 is
  available.
* **ONNXRuntime** – CPU **or** GPU EP, by leveraging `optimum.onnxruntime`.
* **AWS Neuron (Inferentia v1 & v2 / Trainium)** – via `optimum.neuron` which
  wraps Neuron compilation & loading.

The public façade is a single call:

```python
model, tokenizer, runtime = load_encoder(
    "bert-base-uncased",         # HF repo id **or** local path
    device=my_device,            # a utils.DeviceInfo
    dtype="bf16",               # "fp32"│"bf16"│"fp16" (only if supported)
    prefer_onnx=False,
)
```

If *model_name_or_path* is a directory **and** already contains pre-compiled
Neuron or ONNX files, they are loaded directly; otherwise the wrapper falls
back to compiling/exporting on-the-fly (with a console warning because that can
be slow).
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Tuple

from .utils import DeviceInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    "load_encoder",
]

# Optional heavy imports – defer until actually required
_TRANS_AVAILABLE = importlib.util.find_spec("transformers") is not None
_OPTIMUM_ONNX_AVAILABLE = importlib.util.find_spec("optimum.onnxruntime") is not None
_OPTIMUM_NEURON_AVAILABLE = importlib.util.find_spec("optimum.neuron") is not None

autocls_model = None  # will be populated lazily


def _lazy_hf_imports():
    """Import HuggingFace classes the first time we really need them."""
    global autocls_model
    if autocls_model is not None:
        return autocls_model

    if not _TRANS_AVAILABLE:
        raise ImportError("🤗 transformers not installed – add to requirements.")

    from transformers import (  # noqa: WPS433 – runtime import is intended
        AutoConfig,
        AutoModel,
        AutoModelForMaskedLM,
        AutoTokenizer,
    )

    autocls_model = (AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig)
    return autocls_model


# ---------------------------------------------------------------------------
# Helper utils
# ---------------------------------------------------------------------------


def _pick_dtype(dtype: str, torch) -> "torch.dtype":  # type: ignore[name-defined]
    mapping = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _has_bf16(device: DeviceInfo, torch) -> bool:  # type: ignore[name-defined]
    if device.device_type == "cuda":
        major = torch.cuda.get_device_capability(device.device_id)[0]
        return major >= 8  # Ampere+
    if device.device_type == "cpu":
        # A simple proxy for determining BF16 support on CPU
        try:
            # This will succeed on machines with AVX512_BF16
            _ = torch.randn(1, dtype=torch.bfloat16)
            return True
        except Exception:
            return False
    if device.device_type == "mps":
        return False
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_encoder(
    model_name_or_path: str,
    device: DeviceInfo,
    *,
    dtype: str = "fp32",
    prefer_onnx: bool | None = None,
    trust_remote_code: bool = False,
) -> Tuple[object, object, str]:
    """Return *(model, tokenizer, runtime_tag)* ready for inference.

    Parameters
    ----------
    model_name_or_path
        Either a HF *model-id* or a local directory.
    device
        One of the :class:`latency_bench.utils.DeviceInfo` objects.
    dtype
        "fp32" (default), "bf16", or "fp16" (if backend allows it).
    prefer_onnx
        Force ONNXRuntime even if PyTorch is available.  If *None* (default),
        auto-select ONNX **only** on CPU-only machines.
    trust_remote_code
        Passed through to HF *from_pretrained* for custom architectures.
    """

    # ------------------------------------------------------------------
    # Decide the runtime backend
    # ------------------------------------------------------------------
    if device.backend == "neuron":
        runtime = "neuron"
    else:
        # If caller explicitly wants ONNX or if we are on a CPU-only box
        cpu_only = device.device_type == "cpu"
        runtime = "onnx" if (prefer_onnx or (prefer_onnx is None and cpu_only)) else "torch"

    logger.info(
        "Loading model '%s' for backend=%s on %s (%s)",
        model_name_or_path,
        runtime,
        device.device_type,
        device.name,
    )

    # ------------------------------------------------------------------
    # Load tokenizer – always via 🌎/local HF repo
    # ------------------------------------------------------------------
    AutoModel, AutoModelForMaskedLM, AutoTokenizer, _ = _lazy_hf_imports()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    # ------------------------------------------------------------------
    # Runtime-specific model loading
    # ------------------------------------------------------------------
    if runtime == "torch":
        import torch

        desired_dtype = _pick_dtype(dtype, torch)
        if dtype == "bf16" and not _has_bf16(device, torch):
            logger.warning("Backend does not support bfloat16 – falling back to fp32")
            desired_dtype = torch.float32

        config_path = Path(model_name_or_path) / "config.json" if os.path.isdir(model_name_or_path) else None
        auto_cls = AutoModelForMaskedLM if config_path and "mlm" in config_path.read_text() else AutoModel
        model = auto_cls.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=desired_dtype,
        )
        model.to(device.torch_device())
        model.eval()

    elif runtime == "onnx":
        if not _OPTIMUM_ONNX_AVAILABLE:
            raise ImportError("Install `optimum[onnxruntime]` for ONNX support.")
        from optimum.onnxruntime import ORTModelForSequenceClassification

        onnx_dir = Path(model_name_or_path)
        if onnx_dir.is_dir() and (onnx_dir / "model.onnx").exists():
            model = ORTModelForSequenceClassification.from_pretrained(onnx_dir)
        else:
            logger.warning("Exporting model to ONNX – this may take a while …")
            model = ORTModelForSequenceClassification.from_pretrained(model_name_or_path, export=True)
            model.save_pretrained("./_exported_onnx")

        ep_list = ["CUDAExecutionProvider"] if device.device_type == "cuda" else ["CPUExecutionProvider"]
        model.set_providers(providers=ep_list)

    elif runtime == "neuron":
        if not _OPTIMUM_NEURON_AVAILABLE:
            raise ImportError("Install `optimum-neuron` for AWS Inferentia support.")
        from optimum.neuron import NeuronModelForSequenceClassification

        local_dir = Path(model_name_or_path)
        
        # Check for the modern 'model.neuron' artifact name.
        is_compiled = local_dir.is_dir() and (local_dir / "model.neuron").exists()
        
        if is_compiled:
            logger.info("✅ Found pre-compiled Neuron artifact in %s. Loading directly.", local_dir)
            # When loading a pre-compiled model, `export` must be False (the default).
            model = NeuronModelForSequenceClassification.from_pretrained(local_dir)
        else:
            # The benchmark.py script is responsible for compiling. If we get here and 
            # the model isn't compiled, it's a workflow error, so we fail fast.
            raise FileNotFoundError(
                f"Neuron benchmark requires a pre-compiled model directory, but "
                f"'model.neuron' was not found in '{local_dir}'. "
                f"Ensure `benchmark.py` ran its compilation step successfully."
            )

    else:
        raise RuntimeError(f"Unknown runtime selected: {runtime}")

    return model, tokenizer, runtime