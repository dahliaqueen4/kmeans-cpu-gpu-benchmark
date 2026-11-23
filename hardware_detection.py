"""
Hardware detection utilities for K-Means benchmarking.

Collects CPU, memory, and GPU information so benchmark results can be
reproduced and compared across machines.
"""

from __future__ import annotations

import os
import platform
from typing import Any

import psutil

try:
    import torch
except ImportError:  # pragma: no cover - torch optional at import time
    torch = None  # type: ignore[assignment]


def detect_hardware() -> dict[str, Any]:
    """
    Detect available system hardware.

    Returns:
        Dictionary with CPU, memory, and GPU metadata.
    """
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "processor": platform.processor() or "unknown",
        "architecture": platform.machine(),
    }

    virtual_memory = psutil.virtual_memory()
    memory_info = {
        "total_gb": round(virtual_memory.total / (1024**3), 2),
        "available_gb": round(virtual_memory.available / (1024**3), 2),
    }

    gpu_info = _detect_gpu()

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
    }


def _detect_gpu() -> dict[str, Any]:
    """Return GPU availability and device details when PyTorch is installed."""
    if torch is None:
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "reason": "PyTorch not installed",
        }

    cuda_available = torch.cuda.is_available()
    devices: list[dict[str, Any]] = []

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

    # Apple Silicon Metal backend
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    return {
        "available": cuda_available or mps_available,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "device_count": len(devices) if cuda_available else (1 if mps_available else 0),
        "devices": devices,
        "recommended_device": _recommended_device(cuda_available, mps_available),
    }


def _recommended_device(cuda_available: bool, mps_available: bool) -> str:
    """Pick the best PyTorch device string for this machine."""
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def print_hardware_summary() -> dict[str, Any]:
    """Print a human-readable hardware summary and return the raw dict."""
    info = detect_hardware()
    cpu = info["cpu"]
    memory = info["memory"]
    gpu = info["gpu"]

    print("=" * 60)
    print("HARDWARE SUMMARY")
    print("=" * 60)
    print(f"Platform: {info['platform']} {info['platform_release']}")
    print(f"Python:   {info['python_version']}")
    print(f"CPU:      {cpu['processor']} ({cpu['architecture']})")
    print(f"Cores:    {cpu['physical_cores']} physical / {cpu['logical_cores']} logical")
    print(f"RAM:      {memory['total_gb']} GB total, {memory['available_gb']} GB available")

    if gpu["available"]:
        print(f"GPU:      available ({gpu['recommended_device']})")
        for device in gpu["devices"]:
            print(
                f"  - [{device['index']}] {device['name']} "
                f"({device['total_memory_gb']} GB)"
            )
        if gpu.get("mps_available") and not gpu.get("cuda_available"):
            print("  - Apple Metal (MPS) backend enabled")
    else:
        reason = gpu.get("reason", "no compatible GPU detected")
        print(f"GPU:      not available ({reason})")

    print("=" * 60)
    return info


if __name__ == "__main__":
    print_hardware_summary()
