"""Measure CPU RAM and Apple-MPS memory around a K-Means run.

Example:
    python memory_tracker.py --samples 10000 --features 32 --clusters 10
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import psutil
import torch


MIB = 1024**2


def require_mps() -> torch.device:
    """Return the MPS device, or fail with a useful message."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Run on an Apple-silicon Mac with an MPS-enabled "
            "PyTorch build."
        )
    return torch.device("mps")


def synchronize() -> None:
    """Wait for queued MPS work so timings and memory readings are meaningful."""
    torch.mps.synchronize()


def memory_snapshot() -> dict[str, float]:
    """Read this process's resident RAM and live tensors allocated on MPS."""
    rss = psutil.Process(os.getpid()).memory_info().rss / MIB
    mps = torch.mps.current_allocated_memory() / MIB
    return {"cpu_ram_mib": rss, "gpu_allocated_mib": mps}


def kmeans(
    points: torch.Tensor, clusters: int, iterations: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A compact MPS K-Means implementation used by this memory benchmark."""
    if clusters <= 0 or clusters > len(points):
        raise ValueError("clusters must be between 1 and the number of samples")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_ids = torch.randperm(len(points), generator=generator)[:clusters]
    centers = points[initial_ids.to(points.device)].clone()

    for _ in range(iterations):
        # Squared Euclidean distance; sqrt is unnecessary for nearest-center choice.
        # Matrix-multiply form uses operations with reliable MPS support.
        distances = (
            points.square().sum(dim=1, keepdim=True)
            + centers.square().sum(dim=1).unsqueeze(0)
            - 2.0 * (points @ centers.T)
        )
        labels = distances.argmin(dim=1)

        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, points)
        # index_add avoids relying on torch.bincount, which older MPS builds lack.
        counts = torch.zeros(clusters, device=points.device, dtype=points.dtype)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=points.dtype))
        nonempty = counts > 0
        # Retaining the old center is a stable policy for an empty cluster.
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]

    return centers, labels


def append_result(path: Path, row: dict[str, object]) -> None:
    """Append one result while writing a header for a new/empty CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("memory_results.csv"))
    args = parser.parse_args()

    if args.samples <= 0 or args.features <= 0 or args.iterations <= 0:
        parser.error("samples, features, and iterations must be positive")
    device = require_mps()

    # Generate on CPU with a seeded generator for repeatability, then transfer once.
    cpu_generator = torch.Generator().manual_seed(args.seed)
    points = torch.randn(args.samples, args.features, generator=cpu_generator).to(device)
    synchronize()

    before = memory_snapshot()
    print(
        f"Before K-Means: CPU RAM={before['cpu_ram_mib']:.2f} MiB, "
        f"MPS allocated={before['gpu_allocated_mib']:.2f} MiB"
    )

    start = time.perf_counter()
    centers, labels = kmeans(points, args.clusters, args.iterations, args.seed)
    synchronize()
    elapsed = time.perf_counter() - start

    after = memory_snapshot()
    print(
        f"After K-Means:  CPU RAM={after['cpu_ram_mib']:.2f} MiB, "
        f"MPS allocated={after['gpu_allocated_mib']:.2f} MiB"
    )
    print(f"Elapsed: {elapsed:.6f} s")

    # Keep outputs alive until after the snapshot; otherwise their memory can vanish.
    _ = (centers, labels)
    append_result(
        args.output,
        {
            "samples": args.samples,
            "features": args.features,
            "clusters": args.clusters,
            "iterations": args.iterations,
            "elapsed_seconds": f"{elapsed:.6f}",
            "cpu_ram_before_mib": f"{before['cpu_ram_mib']:.3f}",
            "cpu_ram_after_mib": f"{after['cpu_ram_mib']:.3f}",
            "cpu_ram_delta_mib": f"{after['cpu_ram_mib'] - before['cpu_ram_mib']:.3f}",
            "gpu_before_mib": f"{before['gpu_allocated_mib']:.3f}",
            "gpu_after_mib": f"{after['gpu_allocated_mib']:.3f}",
            "gpu_delta_mib": f"{after['gpu_allocated_mib'] - before['gpu_allocated_mib']:.3f}",
        },
    )
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
