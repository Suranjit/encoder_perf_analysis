"""latency_bench.compile
================================
CLI utility that **pre-compiles** a suite of HuggingFace encoder models to run
natively on AWS Inferentia v1 (Inf1) or v2 / Trainium (Inf2, Trn1/Trn2).

The compilation is handled via **optimum-neuron**, which wraps the Neuron SDK
and takes care of tracing, graph partitioning and saving the resulting
TorchScript artefact (`traced_model.pt`).  Once compiled, the models can be
loaded instantly by `latency_bench.loader.load_encoder()` without incurring the
(one-time) compile cost during the latency benchmark.

Usage
-----
```bash
python -m latency_bench.compile --suite base  # default BF16
python -m latency_bench.compile --suite all  --dtype fp32 --out compiled_models
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List

from latency_bench.utils import DeviceInfo, get_available_devices

# Optional heavy deps – import lazily so help text works on non-Neuron hosts
try:
    from transformers import AutoTokenizer  # noqa: WPS433
    from optimum.neuron import (
        NeuronModelForSequenceClassification,
        NeuronModelForSeq2SeqLM,
    )
except ModuleNotFoundError as err:  # pragma: no cover – will trigger in CI
    print("❌  This script requires `optimum-neuron` and the AWS Neuron SDK.")
    print("    Make sure you run it **inside** an Inf1/Inf2 DLAMI or after you")
    print("    installed the Neuron packages manually.")
    sys.exit(1)

logger = logging.getLogger("latency_bench.compile")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# 📦  Model suites
# ---------------------------------------------------------------------------

SUITES: dict[str, List[str]] = {
    "base": [
        "bert-base-uncased",
        "roberta-base",
        "t5-small",
    ],
    "large": [
        "bert-large-uncased",
        "roberta-large",
        "t5-large",
    ],
    "all": [
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
        "t5-small",
        "t5-large",
    ],
}

DTYPE_CHOICES = {"fp32", "bf16", "fp16"}

# ---------------------------------------------------------------------------
# 🛠  Helpers
# ---------------------------------------------------------------------------

def _neuron_device() -> DeviceInfo:
    """Return the first Neuron device found or abort."""

    for d in get_available_devices():
        if d.backend == "neuron":
            return d
    logger.error("No Inferentia/Trainium device detected – aborting.")
    sys.exit(1)


def _compile(model_id: str, out_dir: Path, dtype: str) -> None:
    """Compile *model_id* if not already compiled in *out_dir*."""

    out_dir.mkdir(parents=True, exist_ok=True)
    compiled_flag = out_dir / "traced_model.pt"
    if compiled_flag.exists():
        logger.info("✔  %s already compiled – skip", model_id)
        return

    logger.info("⚙️  Compiling %s  →  %s", model_id, out_dir)

    if "t5" in model_id.lower():
        ModelCls = NeuronModelForSeq2SeqLM
    else:
        ModelCls = NeuronModelForSequenceClassification

    # 1️⃣ Compile (export=True triggers tracing)
    model = ModelCls.from_pretrained(
        model_id,
        export=True,
        auto_cast=dtype if dtype in {"bf16", "fp16"} else None,
    )

    # 2️⃣ Save artefact + tokenizer
    model.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(out_dir)

    logger.info("✅  Compiled and saved %s", model_id)


# ---------------------------------------------------------------------------
# 🚀  Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401 – simple wrapper
    p = argparse.ArgumentParser(description="Bulk-compile HF models for AWS Neuron")
    p.add_argument(
        "--suite",
        choices=SUITES.keys(),
        default="base",
        help="Which pre-defined model set to compile (default: base)",
    )
    p.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="bf16",
        help="Numerical precision to try during compilation (default: bf16)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("compiled_models"),
        help="Destination directory for compiled artefacts (default: ./compiled_models)",
    )
    return p.parse_args()


def main() -> None:  # noqa: D103 – entry-point
    args = _parse_args()

    _neuron_device()  # abort early if no Neuron HW available

    models: Iterable[str] = SUITES[args.suite]
    for model_id in models:
        target_dir = args.out / model_id.replace("/", "__")
        try:
            _compile(model_id, target_dir, args.dtype)
        except Exception as exc:  # noqa: BLE001 – we want to continue other models
            logger.error("💥  Failed to compile %s: %s", model_id, exc)

    logger.info("🏁  Done. Compiled artefacts under %s", args.out.resolve())


if __name__ == "__main__":  # pragma: no cover – CLI entry
    main()
