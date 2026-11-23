"""
CPU K-Means implementation accelerated with Numba JIT.

Uses Lloyd's algorithm with vectorized distance computations inside
Numba-compiled kernels for better single-thread and multi-core performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange


@dataclass
class KMeansResult:
    """Container for K-Means outputs and timing metadata."""

    labels: np.ndarray
    centroids: np.ndarray
    iterations: int
    inertia: float
    elapsed_seconds: float


@njit(cache=True)
def _pairwise_squared_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distances between all points and centroids.

    Args:
        points: Shape (n_samples, n_features)
        centroids: Shape (n_clusters, n_features)

    Returns:
        Distance matrix of shape (n_samples, n_clusters)
    """
    n_samples = points.shape[0]
    n_clusters = centroids.shape[0]
    distances = np.empty((n_samples, n_clusters), dtype=np.float64)

    for i in range(n_samples):
        for k in range(n_clusters):
            dist = 0.0
            for j in range(points.shape[1]):
                diff = points[i, j] - centroids[k, j]
                dist += diff * diff
            distances[i, k] = dist

    return distances


@njit(cache=True)
def _assign_labels(distances: np.ndarray) -> np.ndarray:
    """Assign each sample to the nearest centroid index."""
    labels = np.empty(distances.shape[0], dtype=np.int64)
    for i in range(distances.shape[0]):
        best = 0
        best_dist = distances[i, 0]
        for k in range(1, distances.shape[1]):
            if distances[i, k] < best_dist:
                best_dist = distances[i, k]
                best = k
        labels[i] = best
    return labels


@njit(cache=True, parallel=True)
def _update_centroids(
    points: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Recompute centroid positions from current cluster assignments."""
    n_features = points.shape[1]
    centroids = np.zeros((n_clusters, n_features), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)

    for i in prange(points.shape[0]):
        cluster = labels[i]
        counts[cluster] += 1
        for j in range(n_features):
            centroids[cluster, j] += points[i, j]

    for k in range(n_clusters):
        if counts[k] > 0:
            for j in range(n_features):
                centroids[k, j] /= counts[k]

    return centroids


@njit(cache=True)
def _compute_inertia(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Sum of squared distances from each point to its assigned centroid."""
    inertia = 0.0
    for i in range(points.shape[0]):
        cluster = labels[i]
        for j in range(points.shape[1]):
            diff = points[i, j] - centroids[cluster, j]
            inertia += diff * diff
    return inertia


def _initialize_centroids(points: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """Pick random data points as initial centroids (Forgy initialization)."""
    rng = np.random.default_rng(random_state)
    indices = rng.choice(points.shape[0], size=n_clusters, replace=False)
    return points[indices].astype(np.float64, copy=True)


def kmeans_cpu(
    points: np.ndarray,
    n_clusters: int = 8,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> KMeansResult:
    """
    Run K-Means clustering on CPU using Numba-accelerated kernels.

    Args:
        points: Input data array of shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        max_iter: Maximum Lloyd iterations
        tol: Stop when centroid movement falls below this threshold
        random_state: Seed for centroid initialization

    Returns:
        KMeansResult with labels, centroids, and timing information
    """
    data = np.asarray(points, dtype=np.float64)
    centroids = _initialize_centroids(data, n_clusters, random_state)

    # Warm up JIT compilation so benchmark timing reflects steady-state performance.
    _pairwise_squared_distances(data[: min(32, len(data))], centroids)
    _assign_labels(_pairwise_squared_distances(data[: min(32, len(data))], centroids))
    _update_centroids(data[: min(32, len(data))], np.zeros(min(32, len(data)), dtype=np.int64), n_clusters)

    start = time.perf_counter()
    labels = np.zeros(data.shape[0], dtype=np.int64)
    iterations = 0

    for iteration in range(max_iter):
        iterations = iteration + 1
        distances = _pairwise_squared_distances(data, centroids)
        labels = _assign_labels(distances)
        new_centroids = _update_centroids(data, labels, n_clusters)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    inertia = _compute_inertia(data, labels, centroids)
    elapsed = time.perf_counter() - start

    return KMeansResult(
        labels=labels,
        centroids=centroids,
        iterations=iterations,
        inertia=float(inertia),
        elapsed_seconds=elapsed,
    )


if __name__ == "__main__":
    # Quick smoke test on synthetic data
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(10_000, 16))
    result = kmeans_cpu(samples, n_clusters=5)
    print(f"Iterations: {result.iterations}")
    print(f"Inertia:    {result.inertia:.2f}")
    print(f"Time:       {result.elapsed_seconds:.4f}s")
