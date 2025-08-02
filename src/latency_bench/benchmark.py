"""latency_bench.benchmark
================================
End-to-end harness that measures **batch-inference latency** for any HuggingFace
encoder model across multiple devices (CPU, CUDA, MPS, Inferentia) and – if
requested – logs the results to **Weights & Biases** for easy graphing.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import psutil
import platform

from latency_bench.loader import load_encoder
from latency_bench.utils import (
    DeviceInfo,
    get_available_devices,
    sweep_batch_size,
)

try:
    import wandb  # optional heavy import

    _WandbAvailable = True
except ModuleNotFoundError:  # pragma: no cover – wandb optional
    _WandbAvailable = False

logger = logging.getLogger("latency_bench.benchmark")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
# 🛠  Helpers
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark batch-inference latency.")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--suite", choices=_SUITE_MODELS.keys())
    group.add_argument("--models", nargs="+", metavar="PATH_OR_ID")

    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda", "mps", "inf1", "inf2"], help="Subset of device types")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
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

    p.add_argument("--wandb-project")
    p.add_argument("--wandb-run", default=None)
    return p.parse_args(argv)


def _select_devices(names: List[str] | None) -> List[DeviceInfo]:
    all_dev = get_available_devices()
    if not names:
        return all_dev
    picked = [d for d in all_dev if d.device_type in names]
    if not picked:
        logger.error("No matching devices: %s", names)
        sys.exit(1)
    return picked


def _wandb_init(args: argparse.Namespace):
    if not args.wandb_project:
        return None
    if not _WandbAvailable:
        logger.error("wandb not installed – `pip install wandb` or omit --wandb-project")
        sys.exit(1)
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run or f"bench-{dt.datetime.now():%Y%m%d-%H%M%S}",
        config={
            "batch_sizes": args.batch_sizes,
            "dtype": args.dtype,
            "seq_len": args.seq_lens,
            "devices": args.devices or "auto",
        },
    )


# ---------------------------------------------------------------------------
# 🚀  Benchmark helpers
# ---------------------------------------------------------------------------

def _benchmark_model(
    model_id: str,
    device: DeviceInfo,
    batch_sizes: Iterable[int],
    seq_len: int,
    dtype: str,
    prefer_onnx: bool,
) -> dict[int, float]:
    model, tok, runtime = load_encoder(
        model_id,
        device,
        dtype=dtype,
        prefer_onnx=prefer_onnx,
        trust_remote_code=True,
    )
    logger.info("🏃  [%s] %s on %s", runtime, model_id, device.device_type)
    return sweep_batch_size(model, tok, device, seq_len=seq_len, batch_sizes=list(batch_sizes))


# ---------------------------------------------------------------------------
# 🏁  Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):  # noqa: D103
    args = _parse_args(argv)
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
    if wandb_run is not None:
        wandb_table = wandb.Table(columns=fieldnames)

    host_info = f"{platform.node()} ({psutil.cpu_count(logical=False)}c/{psutil.cpu_count()}t)"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_id in models:
            for dev in devices:
                try:
                    for seq_len in args.seq_lens:
                        results = _benchmark_model(
                            model_id, dev, args.batch_sizes, seq_len, args.dtype, args.prefer_onnx
                        )
                        for bs, latency in results.items():
                            row = {
                                "model": model_id,
                                "device_type": dev.device_type,
                                "device_name": dev.name,
                                "runtime": dev.backend,
                                "batch_size": bs,
                                "latency_ms": latency,
                                "dtype": args.dtype,
                                "seq_len": seq_len,           # <- use loop variable
                                "host": host_info,
                            }
                            writer.writerow(row)
                            if wandb_run is not None:
                                wandb_table.add_data(*row.values())
                        # end inner latency loop
                    # end seq_len loop
                except Exception as exc:  # noqa: BLE001 – continue
                    logger.error("💥  %s on %s failed: %s", model_id, dev.device_type, exc)
                    continue
                
    logger.info("📄  Results saved to %s", csv_path.resolve())

    if wandb_run is not None:
        wandb_run.log({"latency_table": wandb_table})
        wandb_run.finish()
        logger.info("🚀  Logged to Weights & Biases project '%s'", args.wandb_project)


if __name__ == "__main__":  # pragma: no cover
    main()
