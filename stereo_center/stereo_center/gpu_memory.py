"""CUDA allocator peak-memory measurement helpers."""

from __future__ import annotations

import torch


def reset_gpu_peak_memory(device: str) -> bool:
    """Start a CUDA allocator peak-memory measurement for this process."""
    if not str(device).startswith("cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return True


def gpu_peak_memory_gib(device: str, enabled: bool) -> float | None:
    """Return the measured CUDA allocator peak in GiB, if tracking is enabled."""
    if not enabled:
        return None
    torch.cuda.synchronize(device)
    return round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 2)
