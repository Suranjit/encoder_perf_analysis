"""latency_bench.benchmark
================================
End-to-end harness that measures **batch-inference latency** for any HuggingFace
encoder model across multiple devices (CPU, CUDA, MPS, Inferentia) and – if
requested – logs the results to **Weights & Biases** for easy graphing.

For Neuron (Inf1 / Inf2 / Trn1) devices the script can **auto-compile** models
on the fly so you don’t have to run `latency_bench.compile` beforehand.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import psutil

from latency_bench.loader import load_encoder
from latency_bench.utils import DeviceInfo, get_available_devices, sweep_batch_size

# ── Optional heavy deps (loaded lazily) ─────────────────────────────────────
try:
    import wandb

    _WandbAvailable = True
except ModuleNotFoundError:  # pragma: no cover
    _WandbAvailable = False

# Neuron-only deps (imported inside helper if needed)
try:
    # NeuronExportConfig and AutoCastMode are deprecated/removed.
    from transformers import AutoTokenizer, AutoConfig
except ModuleNotFoundError:  # pragma: no cover
    AutoTokenizer = AutoConfig = None  # type: ignore

logger = logging.getLogger("latency_bench.benchmark")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Define a reasonable, fixed batch size for hardware compilation to avoid OOM errors.
HARDWARE_COMPILE_BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# 🔖  Model suites
# ---------------------------------------------------------------------------

_SUITE_MODELS = {
    "base": [
        "bert-base-uncased",
        "roberta-base",
    ],
    "large": [
        "bert-large-uncased",
        "roberta-large",
    ],
    "all": [
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
    ],
}

# ---------------------------------------------------------------------------
# 🛠  CLI + helpers
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark batch-inference latency.")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--suite", choices=_SUITE_MODELS.keys())
    grp.add_argument("--models", nargs="+", metavar="PATH_OR_ID")

    p.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "cuda", "mps", "inf1", "inf2"],
        help="Which device types to benchmark (default: all detected)",
    )
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096],
        help="Batch sizes to test",
    )
    p.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[128],
        metavar="LEN",
        help="One or more sequence lengths to test",
    )
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--prefer-onnx", action="store_true")
    p.add_argument("--out", type=Path, default=Path("runs"))

    # Neuron helpers
    p.add_argument(
        "--auto-compile",
        dest="auto_compile",
        action="store_true",
        help="Auto-compile Neuron artefacts when missing (default on for inf1/inf2)",
    )
    p.add_argument("--no-auto-compile", dest="auto_compile", action="store_false")

    # Tracking
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-run")

    # default auto_compile=None so we can decide later
    p.set_defaults(auto_compile=None)
    return p.parse_args(argv)


def _select_devices(names: List[str] | None) -> List[DeviceInfo]:
    all_dev = get_available_devices()
    if not names:
        return all_dev
    picked = [d for d in all_dev if d.device_type in names]
    if not picked:
        logger.error("No matching devices for %s", names)
        sys.exit(1)
    return picked


def _wandb_init(args: argparse.Namespace):
    if not args.wandb_project:
        return None
    if not _WandbAvailable:
        logger.error("wandb not installed – omit --wandb-project or install package")
        sys.exit(1)
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run or f"bench-{dt.datetime.now():%Y%m%d-%H%M%S}",
        config={
            "batch_sizes": args.batch_sizes,
            "seq_lens": args.seq_lens,
            "dtype": args.dtype,
            "devices": args.devices or "auto",
        },
    )


# ---------------------------------------------------------------------------
# 🧩  Neuron auto-compile
# ---------------------------------------------------------------------------


def _ensure_neuron_artefact(
    model_id: str,
    artefact_root: Path,
    batch_size: int,
    seq_len: int,
    dtype: str,
) -> Path:
    """Compile `(model_id, bs, seq_len, dtype)` if artefact missing and return folder."""
    artefact_dir = (
        artefact_root
        / model_id.replace("/", "__")
        / f"bs{batch_size}_L{seq_len}_{dtype}"
    )
    if (artefact_dir / "model.neuron").exists(): # The new format saves a .neuron file
        return artefact_dir

    # We need to import the Neuron classes here, inside the function.
    try:
        if "t5" in model_id.lower():
            from optimum.neuron import NeuronModelForSeq2SeqLM as NeuronCls
        else:
            from optimum.neuron import NeuronModelForSequenceClassification as NeuronCls
    except ImportError:
        raise RuntimeError("optimum-neuron not available in environment")

    logger.info("⚙️  Compiling %s bs=%d L=%d (%s)", model_id, batch_size, seq_len, dtype)

    # Define compiler arguments for the new API
    input_shapes = {"batch_size": batch_size, "sequence_length": seq_len}
    
    compiler_args = ["--optlevel", "1"]
    
    # The `export` argument is now mandatory for compilation
    model = NeuronCls.from_pretrained(
        model_id,
        export=True,
        auto_cast_type='bf16' if dtype == 'bf16' else 'fp32',
        compiler_args=compiler_args,
        **input_shapes,
    )

    artefact_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artefact_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(artefact_dir)
    
    return artefact_dir


# ---------------------------------------------------------------------------
# 🚀  Benchmark helper
# ---------------------------------------------------------------------------


def _benchmark_model(
    model_or_path: str,
    device: DeviceInfo,
    batch_sizes: Iterable[int],
    seq_len: int,
    dtype: str,
    prefer_onnx: bool,
) -> dict[int, float]:
    model, tok, runtime = load_encoder(
        model_or_path,
        device,
        dtype=dtype,
        prefer_onnx=prefer_onnx,
        trust_remote_code=True,
    )
    logger.info("🏃  [%s] Benchmarking %s on %s...", runtime, model_or_path, device.device_type)
    return sweep_batch_size(model, tok, device, seq_len, list(batch_sizes))


# ---------------------------------------------------------------------------
# 🏁  Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None):
    args = _parse_args(argv)

    # decide default for auto-compile
    if args.auto_compile is None:
        args.auto_compile = bool({"inf1", "inf2"} & set(args.devices or []))

    models: Iterable[str] = args.models if args.models else _SUITE_MODELS[args.suite]
    devices = _select_devices(args.devices)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"bench-{dt.datetime.now():%Y%m%d-%H%M%S}.csv"

    wandb_run = _wandb_init(args)

    fieldnames = [
        "model",
        "device_type",
        "device_name",
        "runtime",
        "batch_size",
        "latency_ms",
        "dtype",
        "seq_len",
        "host",
    ]
    if wandb_run:
        wandb_table = wandb.Table(columns=fieldnames)

    host_info = f"{platform.node()} ({psutil.cpu_count(logical=False)}c/{psutil.cpu_count()}t)"

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for model_id in models:
            for dev in devices:
                for seq_len in args.seq_lens:
                    try:
                        # auto-compile if requested & device is Neuron
                        model_ref = model_id
                        if args.auto_compile and dev.backend == "neuron":
                            arte_root = Path("compiled")
                            # Compile for the fixed hardware batch size, not the max from user input
                            bs_for_compile = HARDWARE_COMPILE_BATCH_SIZE
                            
                            failed_dir = (
                                arte_root
                                / model_id.replace("/", "__")
                                / f"bs{bs_for_compile}_L{seq_len}_{args.dtype}"
                            )
                            if failed_dir.exists():
                                logger.warning("Removing previous failed artifact at %s", failed_dir)
                                import shutil
                                shutil.rmtree(failed_dir)

                            model_ref = str(
                                _ensure_neuron_artefact(
                                    model_id, arte_root, bs_for_compile, seq_len, args.dtype
                                )
                            )

                        res = _benchmark_model(
                            model_ref,
                            dev,
                            args.batch_sizes,
                            seq_len,
                            args.dtype,
                            args.prefer_onnx,
                        )
                    except Exception as exc:  # pragma: no cover – keep going
                        logger.error("💥  %s on %s (L=%d) failed: %s", model_id, dev.device_type, seq_len, exc)
                        continue

                    for bs, latency in res.items():
                        row = {
                            "model": model_id,
                            "device_type": dev.device_type,
                            "device_name": dev.name,
                            "runtime": dev.backend,
                            "batch_size": bs,
                            "latency_ms": latency,
                            "dtype": args.dtype,
                            "seq_len": seq_len,
                            "host": host_info,
                        }
                        writer.writerow(row)
                        if wandb_run:
                            wandb_table.add_data(*row.values())

    logger.info("📄  Results saved to %s", csv_path.resolve())

    if wandb_run:
        wandb_run.log({"latency_table": wandb_table})
        wandb_run.finish()
        logger.info("🚀  Logged to Weights & Biases project '%s'", args.wandb_project)


if __name__ == "__main__":  # pragma: no cover
    main()