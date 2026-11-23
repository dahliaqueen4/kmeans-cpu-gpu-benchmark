"""Benchmark CPU-to-MPS transfers from pageable and pinned host memory.

Apple silicon uses unified memory, and PyTorch MPS currently does not provide the
CUDA pinned-memory transfer path on many releases. This script tries the real
``Tensor.pin_memory()`` API. If PyTorch rejects it, the CSV records pinned memory
as unsupported instead of substituting ordinary memory and mislabelling results.

Outputs:
    pageable_vs_pinned_results.csv
    pageable_vs_pinned_transfer_speed.png
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_SIZES_MIB = (1, 10, 100, 500, 1024)
DEFAULT_REPEATS = 5
BYTES_PER_MIB = 1024**2


def require_mps() -> torch.device:
    """Return an MPS device or stop with a clear environment error."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Run this script on an Apple-silicon Mac from "
            "an MPS-enabled PyTorch environment."
        )
    return torch.device("mps")


def transfer_once(source: torch.Tensor, device: torch.device) -> float:
    """Measure one host-to-MPS copy, including completion of queued GPU work."""
    torch.mps.synchronize()
    start = time.perf_counter()
    destination = source.to(device=device, non_blocking=True)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    # Keep the destination alive through the second synchronization.
    del destination
    return elapsed


def benchmark_source(
    source: torch.Tensor, device: torch.device, repeats: int
) -> list[float]:
    """Warm up once, then return transfer durations for recorded repetitions."""
    warmup = source[: min(source.numel(), 256 * 1024)]
    warmup_gpu = warmup.to(device=device, non_blocking=True)
    torch.mps.synchronize()
    del warmup_gpu

    timings = [transfer_once(source, device) for _ in range(repeats)]
    torch.mps.empty_cache()
    return timings


def gib_per_second(byte_count: int, seconds: float) -> float:
    """Convert a copy duration to binary gigabytes per second."""
    return (byte_count / 1024**3) / seconds


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    """Rewrite the accumulated averages so partial progress survives each size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict[str, object]], path: Path) -> None:
    """Create a grouped bar chart; unsupported pinned values remain visibly absent."""
    labels = [str(row["size_label"]) for row in rows]
    pageable = [float(row["pageable_average_gib_s"]) for row in rows]
    pinned = [
        float(row["pinned_average_gib_s"])
        if row["pinned_average_gib_s"] != ""
        else np.nan
        for row in rows
    ]

    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(positions - width / 2, pageable, width, label="Pageable")
    axis.bar(positions + width / 2, pinned, width, label="Pinned")
    axis.set_title("CPU to Apple MPS Transfer Speed")
    axis.set_xlabel("Transfer size")
    axis.set_ylabel("Average transfer speed (GiB/s)")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    if all(np.isnan(value) for value in pinned):
        axis.text(
            0.5,
            0.96,
            "Pinned host memory is unsupported by this PyTorch MPS backend",
            ha="center",
            va="top",
            transform=axis.transAxes,
            fontsize=9,
        )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes-mib",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES_MIB),
        help="Transfer sizes in MiB (default: 1 10 100 500 1024)",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pageable_vs_pinned_results.csv"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("pageable_vs_pinned_transfer_speed.png"),
    )
    args = parser.parse_args()

    if any(size <= 0 for size in args.sizes_mib) or args.repeats <= 0:
        parser.error("all sizes and repeats must be positive")
    device = require_mps()

    rows: list[dict[str, object]] = []
    pinned_supported = True
    pinned_error = ""

    for size_mib in args.sizes_mib:
        byte_count = size_mib * BYTES_PER_MIB
        element_count = byte_count // np.dtype(np.float32).itemsize
        print(f"\nBenchmarking {size_mib} MiB ({args.repeats} timed transfers)...")

        # np.ones touches every page, preventing lazy virtual-memory allocation from
        # making the first measured transfer fundamentally different from later ones.
        cpu_array = np.ones(element_count, dtype=np.float32)
        pageable_tensor = torch.from_numpy(cpu_array)  # Shares the NumPy allocation.
        pageable_times = benchmark_source(pageable_tensor, device, args.repeats)
        pageable_average = statistics.mean(pageable_times)

        pinned_times: list[float] = []
        if pinned_supported:
            try:
                # Pinning is preparation, so it is intentionally outside transfer timing.
                pinned_tensor = pageable_tensor.pin_memory()
                pinned_times = benchmark_source(pinned_tensor, device, args.repeats)
                del pinned_tensor
            except (RuntimeError, NotImplementedError) as error:
                pinned_supported = False
                pinned_error = " ".join(str(error).splitlines())
                print(f"Pinned memory unsupported: {pinned_error}")

        pinned_average = statistics.mean(pinned_times) if pinned_times else None
        rows.append(
            {
                "size_label": "1 GiB" if size_mib == 1024 else f"{size_mib} MiB",
                "size_mib": size_mib,
                "bytes": byte_count,
                "repeats": args.repeats,
                "pageable_average_seconds": f"{pageable_average:.9f}",
                "pageable_average_gib_s": f"{gib_per_second(byte_count, pageable_average):.6f}",
                "pinned_average_seconds": (
                    f"{pinned_average:.9f}" if pinned_average is not None else ""
                ),
                "pinned_average_gib_s": (
                    f"{gib_per_second(byte_count, pinned_average):.6f}"
                    if pinned_average is not None
                    else ""
                ),
                "pinned_status": "supported" if pinned_average is not None else "unsupported",
                "pinned_error": "" if pinned_average is not None else pinned_error,
            }
        )
        write_results(args.output, rows)
        print(
            f"Pageable average: {pageable_average * 1000:.3f} ms, "
            f"{gib_per_second(byte_count, pageable_average):.3f} GiB/s"
        )
        if pinned_average is not None:
            print(
                f"Pinned average:   {pinned_average * 1000:.3f} ms, "
                f"{gib_per_second(byte_count, pinned_average):.3f} GiB/s"
            )

        del pageable_tensor, cpu_array
        gc.collect()
        torch.mps.empty_cache()

    make_plot(rows, args.plot)
    print(f"\nSaved results to {args.output}")
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
