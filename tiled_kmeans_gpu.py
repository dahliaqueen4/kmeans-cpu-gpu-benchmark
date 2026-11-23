"""Compare naive and manually tiled K-Means distance calculations on MPS."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable

import torch


DistanceFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def require_mps() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("This benchmark requires Apple MPS and an MPS-enabled PyTorch.")
    return torch.device("mps")


def naive_squared_distances(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Materialize the N x K x D difference tensor (simple, but memory hungry)."""
    return (points[:, None, :] - centers[None, :, :]).square().sum(dim=2)


def tiled_squared_distances(
    points: torch.Tensor, centers: torch.Tensor, tile_size: int = 64
) -> torch.Tensor:
    """Compute N x K distances in blocks without an N x K x D temporary.

    Both sample and center dimensions are tiled. The algebraic identity
    ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x@c keeps each temporary two-dimensional.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    result = torch.empty(
        (points.shape[0], centers.shape[0]), device=points.device, dtype=points.dtype
    )
    for row in range(0, points.shape[0], tile_size):
        x = points[row : row + tile_size]
        x_norm = x.square().sum(dim=1, keepdim=True)
        for column in range(0, centers.shape[0], tile_size):
            c = centers[column : column + tile_size]
            c_norm = c.square().sum(dim=1).unsqueeze(0)
            block = x_norm + c_norm - 2.0 * (x @ c.T)
            # Round-off can make a theoretically nonnegative value slightly negative.
            result[row : row + len(x), column : column + len(c)] = block.clamp_min_(0)
    return result


def initial_centers(
    points: torch.Tensor, clusters: int, seed: int
) -> torch.Tensor:
    if clusters <= 0 or clusters > len(points):
        raise ValueError("clusters must be between 1 and the number of samples")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randperm(len(points), generator=generator)[:clusters]
    return points[ids.to(points.device)].clone()


def run_kmeans(
    points: torch.Tensor,
    centers: torch.Tensor,
    iterations: int,
    distance_function: DistanceFunction,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run K-Means using an injected distance implementation."""
    centers = centers.clone()
    clusters = len(centers)
    for _ in range(iterations):
        labels = distance_function(points, centers).argmin(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, points)
        # index_add is supported on older MPS builds where bincount may not be.
        counts = torch.zeros(clusters, device=points.device, dtype=points.dtype)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=points.dtype))
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
    return centers, labels


def timed_kmeans(
    points: torch.Tensor,
    centers: torch.Tensor,
    iterations: int,
    distance_function: DistanceFunction,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Time asynchronous MPS work correctly by synchronizing at both boundaries."""
    torch.mps.synchronize()
    start = time.perf_counter()
    final_centers, labels = run_kmeans(points, centers, iterations, distance_function)
    torch.mps.synchronize()
    return time.perf_counter() - start, final_centers, labels


def write_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("tiled_comparison.csv"))
    args = parser.parse_args()

    if min(args.samples, args.features, args.iterations, args.tile_size) <= 0:
        parser.error("samples, features, iterations, and tile-size must be positive")
    device = require_mps()
    generator = torch.Generator().manual_seed(args.seed)
    points = torch.randn(args.samples, args.features, generator=generator).to(device)
    centers = initial_centers(points, args.clusters, args.seed)

    # A small warm-up removes one-time MPS graph/kernel setup from the comparison.
    warm_points = points[: min(len(points), 256)]
    _ = naive_squared_distances(warm_points, centers)
    _ = tiled_squared_distances(warm_points, centers, args.tile_size)
    torch.mps.synchronize()

    naive_time, naive_centers, naive_labels = timed_kmeans(
        points, centers, args.iterations, naive_squared_distances
    )
    tiled_fn = lambda x, c: tiled_squared_distances(x, c, args.tile_size)
    tiled_time, tiled_centers, tiled_labels = timed_kmeans(
        points, centers, args.iterations, tiled_fn
    )

    # The two mathematically equivalent routes can differ slightly from FP rounding.
    agreement = (naive_labels == tiled_labels).float().mean().item()
    center_max_error = (naive_centers - tiled_centers).abs().max().item()
    common = {
        "samples": args.samples,
        "features": args.features,
        "clusters": args.clusters,
        "iterations": args.iterations,
        "tile_size": args.tile_size,
    }
    rows = [
        {**common, "method": "naive", "seconds": f"{naive_time:.6f}", "speedup_vs_naive": "1.000000"},
        {**common, "method": "tiled", "seconds": f"{tiled_time:.6f}", "speedup_vs_naive": f"{naive_time / tiled_time:.6f}"},
    ]
    write_comparison(args.output, rows)
    print(f"Naive: {naive_time:.6f} s")
    print(f"Tiled ({args.tile_size}): {tiled_time:.6f} s")
    print(f"Tiled speedup: {naive_time / tiled_time:.3f}x")
    print(f"Final-label agreement: {agreement:.2%}; max center error: {center_max_error:.3g}")
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
