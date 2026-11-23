"""Benchmark manual MPS K-Means tiles at several block sizes, three runs each."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch

from tiled_kmeans_gpu import (
    initial_centers,
    require_mps,
    tiled_squared_distances,
    timed_kmeans,
)


BLOCK_SIZES = (16, 32, 64, 128, 256)
RUNS_PER_SIZE = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("block_sweep_results.csv")
    )
    args = parser.parse_args()

    if min(args.samples, args.features, args.iterations) <= 0:
        parser.error("samples, features, and iterations must be positive")
    device = require_mps()
    generator = torch.Generator().manual_seed(args.seed)
    points = torch.randn(args.samples, args.features, generator=generator).to(device)
    centers = initial_centers(points, args.clusters, args.seed)

    rows: list[dict[str, object]] = []
    for block_size in BLOCK_SIZES:
        distance_fn = lambda x, c, size=block_size: tiled_squared_distances(x, c, size)

        # Warm each shape once; warm-up is deliberately excluded from recorded runs.
        _ = distance_fn(points[: min(len(points), 256)], centers)
        torch.mps.synchronize()

        timings = []
        for run in range(1, RUNS_PER_SIZE + 1):
            seconds, _, _ = timed_kmeans(
                points, centers, args.iterations, distance_fn
            )
            timings.append(seconds)
            rows.append(
                {
                    "block_size": block_size,
                    "run": run,
                    "seconds": f"{seconds:.6f}",
                    "samples": args.samples,
                    "features": args.features,
                    "clusters": args.clusters,
                    "iterations": args.iterations,
                }
            )
            print(f"Block {block_size:3d}, run {run}: {seconds:.6f} s")
        print(
            f"Block {block_size:3d} mean: {statistics.mean(timings):.6f} s "
            f"(stdev {statistics.stdev(timings):.6f} s)"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
