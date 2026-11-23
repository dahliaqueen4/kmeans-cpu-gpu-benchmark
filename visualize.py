"""
Visualization utilities for K-Means benchmark results.

Reads CSV output from benchmark.py and generates performance comparison plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def load_latest_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load the most recent benchmark CSV from the results directory."""
    csv_files = sorted(results_dir.glob("benchmark_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No benchmark CSV files found in {results_dir.resolve()}. "
            "Run benchmark.py first."
        )
    latest = csv_files[-1]
    print(f"Loading results from: {latest}")
    return pd.read_csv(latest)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average elapsed time across repeated trials."""
    group_cols = ["implementation", "n_samples", "n_features", "n_clusters"]
    aggregated = (
        df.groupby(group_cols, as_index=False)
        .agg(
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
            std_elapsed_seconds=("elapsed_seconds", "std"),
            mean_iterations=("iterations", "mean"),
        )
        .fillna(0.0)
    )
    return aggregated


def plot_runtime_vs_samples(
    df: pd.DataFrame, n_features: int, output_dir: Path
) -> Path:
    """Plot mean runtime as sample size increases for a fixed feature dimension."""
    subset = df[df["n_features"] == n_features]
    if subset.empty:
        raise ValueError(f"No data for n_features={n_features}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for impl, group in subset.groupby("implementation"):
        ordered = group.sort_values("n_samples")
        ax.errorbar(
            ordered["n_samples"],
            ordered["mean_elapsed_seconds"],
            yerr=ordered["std_elapsed_seconds"],
            marker="o",
            capsize=4,
            label=impl.upper(),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.set_title(f"K-Means runtime vs sample size (d={n_features})")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"runtime_vs_samples_d{n_features}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_runtime_vs_dimensions(
    df: pd.DataFrame, n_samples: int, output_dir: Path
) -> Path:
    """Plot mean runtime as feature dimension increases for a fixed sample size."""
    subset = df[df["n_samples"] == n_samples]
    if subset.empty:
        raise ValueError(f"No data for n_samples={n_samples}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for impl, group in subset.groupby("implementation"):
        ordered = group.sort_values("n_features")
        ax.errorbar(
            ordered["n_features"],
            ordered["mean_elapsed_seconds"],
            yerr=ordered["std_elapsed_seconds"],
            marker="s",
            capsize=4,
            label=impl.upper(),
        )

    ax.set_xlabel("Number of features")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.set_title(f"K-Means runtime vs dimensionality (n={n_samples:,})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"runtime_vs_dims_n{n_samples}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_speedup_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Plot CPU/GPU speedup ratio as a heatmap over sample size and dimension.

    Values > 1 mean GPU is faster; values < 1 mean CPU is faster.
    """
    pivot_cpu = df.pivot_table(
        index="n_samples",
        columns="n_features",
        values="mean_elapsed_seconds",
        aggfunc="mean",
    )
    pivot_gpu = df[df["implementation"] == "gpu"].pivot_table(
        index="n_samples",
        columns="n_features",
        values="mean_elapsed_seconds",
        aggfunc="mean",
    )

    if pivot_gpu.empty or pivot_cpu.empty:
        raise ValueError("Need both CPU and GPU results to compute speedup heatmap.")

    speedup = pivot_cpu / pivot_gpu

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(speedup.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(speedup.columns)))
    ax.set_xticklabels(speedup.columns)
    ax.set_yticks(range(len(speedup.index)))
    ax.set_yticklabels(speedup.index)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Number of samples")
    ax.set_title("Speedup (CPU time / GPU time)")
    fig.colorbar(im, ax=ax, label="Speedup (>1 = GPU faster)")
    fig.tight_layout()

    output_path = output_dir / "speedup_heatmap.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_all_plots(
    results_dir: Path = RESULTS_DIR,
    output_dir: Path = PLOTS_DIR,
) -> list[Path]:
    """Generate the full set of benchmark visualization plots."""
    raw = load_latest_results(results_dir)
    aggregated = _aggregate(raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    for n_features in sorted(aggregated["n_features"].unique()):
        saved.append(plot_runtime_vs_samples(aggregated, int(n_features), output_dir))

    for n_samples in sorted(aggregated["n_samples"].unique()):
        saved.append(plot_runtime_vs_dimensions(aggregated, int(n_samples), output_dir))

    if set(aggregated["implementation"]) >= {"cpu", "gpu"}:
        saved.append(plot_speedup_heatmap(aggregated, output_dir))

    print("\nSaved plots:")
    for path in saved:
        print(f"  - {path}")

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize K-Means benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing benchmark CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory to save generated plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_all_plots(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
