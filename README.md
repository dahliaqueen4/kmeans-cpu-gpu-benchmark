# K-Means Benchmark

A computer architecture benchmarking project comparing **CPU** and **GPU** K-Means clustering performance.

## Overview

This project implements K-Means (Lloyd's algorithm) in two ways:

| Module | Backend | Description |
|--------|---------|-------------|
| `kmeans_cpu.py` | Numba JIT | CPU-accelerated loops with parallel centroid updates |
| `kmeans_gpu.py` | PyTorch | GPU/MPS/CPU tensor operations via PyTorch |

Supporting scripts detect hardware, run experiments, and visualize results.

## Project Structure

```
kmeans-benchmark/
├── requirements.txt        # Python dependencies
├── hardware_detection.py   # CPU, RAM, and GPU detection
├── kmeans_cpu.py           # Numba CPU implementation
├── kmeans_gpu.py           # PyTorch GPU implementation
├── benchmark.py            # Experiment runner (saves CSV)
├── visualize.py            # Performance graph generator
├── results/                # Benchmark CSV output (created on run)
└── plots/                  # Generated charts (created on run)
```

## Setup

```bash
cd kmeans-benchmark
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Check hardware**

```bash
python hardware_detection.py
```

**2. Run benchmarks**

```bash
python benchmark.py
```

Optional flags:

```bash
python benchmark.py --samples 1000 10000 --dims 8 16 --repeats 5 --impl both
```

**3. Generate plots**

```bash
python visualize.py
```

Plots are saved to the `plots/` directory.

## Benchmark Metrics

Each run records:

- Sample count and feature dimensionality
- Implementation (CPU vs GPU)
- Iterations to convergence
- Total elapsed time (seconds)
- Within-cluster sum of squares (inertia)

Results are written to `results/benchmark_<timestamp>.csv` along with a hardware metadata JSON file.

## Notes

- The CPU path uses Numba `@njit` kernels; the first call includes JIT warmup (handled inside the benchmark).
- The GPU path automatically selects `cuda`, Apple `mps`, or `cpu` based on available hardware.
- For fair comparison, use the same random seed and dataset sizes across implementations.

## Course Context

Designed for a computer architecture class to explore how algorithm choice, parallelism, and hardware (CPU cores vs GPU) affect clustering performance at scale.
