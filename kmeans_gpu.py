"""
GPU K-Means implementation using PyTorch.

Runs Lloyd's algorithm on CUDA, Apple MPS, or CPU depending on hardware.
Distance computations use broadcasted tensor operations for parallelism.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from hardware_detection import detect_hardware


@dataclass
class KMeansResult:
    """Container for K-Means outputs and timing metadata."""

    labels: np.ndarray
    centroids: np.ndarray
    iterations: int
    inertia: float
    elapsed_seconds: float
    device: str


def get_device(preferred: str | None = None) -> torch.device:
    """
    Select the best available PyTorch device.

    Args:
        preferred: Optional override ('cuda', 'mps', or 'cpu')

    Returns:
        torch.device instance
    """
    if preferred is not None:
        return torch.device(preferred)

    hw = detect_hardware()
    return torch.device(hw["gpu"]["recommended_device"])


def _initialize_centroids(
    points: torch.Tensor, n_clusters: int, random_state: int
) -> torch.Tensor:
    """Randomly sample initial centroids from the input data."""
    generator = torch.Generator(device=points.device)
    generator.manual_seed(random_state)
    indices = torch.randperm(points.shape[0], generator=generator, device=points.device)[
        :n_clusters
    ]
    return points[indices].clone()


def _pairwise_squared_distances(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Compute squared Euclidean distances using broadcasting.

    Args:
        points: Shape (n_samples, n_features)
        centroids: Shape (n_clusters, n_features)

    Returns:
        Tensor of shape (n_samples, n_clusters)
    """
    # (n, 1, f) - (1, k, f) -> (n, k, f)
    diff = points.unsqueeze(1) - centroids.unsqueeze(0)
    return torch.sum(diff * diff, dim=2)


def kmeans_gpu(
    points: np.ndarray,
    n_clusters: int = 8,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    device: str | None = None,
) -> KMeansResult:
    """
    Run K-Means clustering using PyTorch (GPU when available).

    Args:
        points: Input data array of shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        max_iter: Maximum Lloyd iterations
        tol: Stop when centroid movement falls below this threshold
        random_state: Seed for centroid initialization
        device: Optional device string override

    Returns:
        KMeansResult with labels, centroids, and timing information
    """
    torch_device = get_device(device)
    data = torch.as_tensor(points, dtype=torch.float32, device=torch_device)
    centroids = _initialize_centroids(data, n_clusters, random_state)

    # Warm up GPU kernels / JIT compilation before timing.
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    warmup = data[: min(32, len(data))]
    _ = _pairwise_squared_distances(warmup, centroids[: min(n_clusters, centroids.shape[0])])
    if torch_device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    labels = torch.zeros(data.shape[0], dtype=torch.long, device=torch_device)
    iterations = 0

    for iteration in range(max_iter):
        iterations = iteration + 1
        distances = _pairwise_squared_distances(data, centroids)
        labels = torch.argmin(distances, dim=1)

        # Accumulate new centroids per cluster.
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_clusters, device=torch_device, dtype=torch.float32)

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            count = mask.sum()
            counts[cluster_id] = count
            if count > 0:
                new_centroids[cluster_id] = data[mask].mean(dim=0)

        # Keep old centroid if a cluster becomes empty during iteration.
        empty = counts == 0
        if torch.any(empty):
            new_centroids[empty] = centroids[empty]

        shift = torch.linalg.norm(new_centroids - centroids).item()
        centroids = new_centroids
        if shift < tol:
            break

    if torch_device.type == "cuda":
        torch.cuda.synchronize()

    assigned = data - centroids[labels]
    inertia = torch.sum(assigned * assigned).item()
    elapsed = time.perf_counter() - start

    return KMeansResult(
        labels=labels.detach().cpu().numpy(),
        centroids=centroids.detach().cpu().numpy(),
        iterations=iterations,
        inertia=float(inertia),
        elapsed_seconds=elapsed,
        device=str(torch_device),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(10_000, 16)).astype(np.float32)
    result = kmeans_gpu(samples, n_clusters=5)
    print(f"Device:     {result.device}")
    print(f"Iterations: {result.iterations}")
    print(f"Inertia:    {result.inertia:.2f}")
    print(f"Time:       {result.elapsed_seconds:.4f}s")
