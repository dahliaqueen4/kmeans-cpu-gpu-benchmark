"""
Benchmark runner for CPU (Numba) vs GPU (PyTorch) K-Means implementations.

Generates synthetic datasets at multiple scales, runs repeated trials,
and saves aggregated timing results to CSV for analysis.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from hardware_detection import detect_hardware, print_hardware_summary
from kmeans_cpu import kmeans_cpu
from kmeans_gpu import kmeans_gpu


DEFAULT_SAMPLE_SIZES = [1_000, 5_000, 10_000, 50_000]
DEFAULT_FEATURE_DIMS = [2, 8, 16, 32]
DEFAULT_N_CLUSTERS = 8
DEFAULT_REPEATS = 3
RESULTS_DIR = Path("results")


def generate_dataset(
    n_samples: int,
    n_features: int,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """Create a clustered synthetic dataset for benchmarking."""
    points, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=random_state,
    )
    return points.astype(np.float64)


def run_single_trial(
    implementation: str,
    points: np.ndarray,
    n_clusters: int,
    max_iter: int,
    random_state: int,
) -> dict:
    """Execute one K-Means run and return timing metrics."""
    if implementation == "cpu":
        result = kmeans_cpu(
            points,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
        )
        device = "cpu-numba"
    elif implementation == "gpu":
        result = kmeans_gpu(
            points,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
        )
        device = result.device
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    return {
        "implementation": implementation,
        "device": device,
        "n_samples": points.shape[0],
        "n_features": points.shape[1],
        "n_clusters": n_clusters,
        "iterations": result.iterations,
        "inertia": result.inertia,
        "elapsed_seconds": result.elapsed_seconds,
        "random_state": random_state,
    }


def run_benchmark(
    sample_sizes: list[int] | None = None,
    feature_dims: list[int] | None = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    repeats: int = DEFAULT_REPEATS,
    max_iter: int = 100,
    implementations: list[str] | None = None,
    output_dir: Path = RESULTS_DIR,
) -> Path:
    """
    Run the full benchmark suite and write results to CSV.

    Returns:
        Path to the saved CSV file.
    """
    sample_sizes = sample_sizes or DEFAULT_SAMPLE_SIZES
    feature_dims = feature_dims or DEFAULT_FEATURE_DIMS
    implementations = implementations or ["cpu", "gpu"]

    print_hardware_summary()
    hardware = detect_hardware()

    rows: list[dict] = []
    total_runs = len(sample_sizes) * len(feature_dims) * len(implementations) * repeats
    run_index = 0

    print(f"\nStarting benchmark ({total_runs} total runs)...")

    for n_samples in sample_sizes:
        for n_features in feature_dims:
            for impl in implementations:
                for repeat in range(repeats):
                    run_index += 1
                    seed = 42 + repeat
                    print(
                        f"[{run_index}/{total_runs}] "
                        f"{impl.upper()} | n={n_samples:,} | d={n_features} | trial={repeat + 1}"
                    )

                    points = generate_dataset(n_samples, n_features, n_clusters, seed)
                    row = run_single_trial(impl, points, n_clusters, max_iter, seed)
                    row["repeat"] = repeat
                    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                    rows.append(row)

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    meta_path = output_dir / f"benchmark_{timestamp}_hardware.json"

    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(hardware, indent=2))

    print(f"\nSaved results to: {csv_path}")
    print(f"Saved hardware metadata to: {meta_path}")
    return csv_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run K-Means CPU vs GPU benchmarks")
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_SIZES,
        help="Sample sizes to benchmark",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=DEFAULT_FEATURE_DIMS,
        help="Feature dimensions to benchmark",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help="Number of clusters",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of repeated trials per configuration",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum K-Means iterations",
    )
    parser.add_argument(
        "--impl",
        choices=["cpu", "gpu", "both"],
        default="both",
        help="Which implementation(s) to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for CSV output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementations = ["cpu", "gpu"] if args.impl == "both" else [args.impl]
    run_benchmark(
        sample_sizes=args.samples,
        feature_dims=args.dims,
        n_clusters=args.clusters,
        repeats=args.repeats,
        max_iter=args.max_iter,
        implementations=implementations,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
