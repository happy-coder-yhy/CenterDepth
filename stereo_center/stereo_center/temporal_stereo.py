"""Temporal alignment primitives for video stereo disparity initialization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _target_to_source_grid(
    flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape (B, 2, H, W)")
    batch, _channels, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    source_x = xx.unsqueeze(0) + flow[:, 0]
    source_y = yy.unsqueeze(0) + flow[:, 1]
    valid = (
        (source_x >= 0.0)
        & (source_x <= width - 1)
        & (source_y >= 0.0)
        & (source_y <= height - 1)
    ).unsqueeze(1)
    grid = torch.stack(
        [
            2.0 * source_x / max(width - 1, 1) - 1.0,
            2.0 * source_y / max(height - 1, 1) - 1.0,
        ],
        dim=-1,
    )
    if grid.shape[0] != batch:
        raise RuntimeError("unexpected temporal sampling grid shape")
    return grid, valid


def backward_warp(
    source: torch.Tensor,
    target_to_source_flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``source`` into target coordinates using target-to-source flow."""
    if source.ndim != 4:
        raise ValueError("source must have shape (B, C, H, W)")
    if source.shape[0] != target_to_source_flow.shape[0]:
        raise ValueError("source and flow batch sizes must match")
    if source.shape[-2:] != target_to_source_flow.shape[-2:]:
        raise ValueError("source and flow spatial sizes must match")
    grid, valid = _target_to_source_grid(target_to_source_flow)
    warped = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, valid


def forward_backward_consistency_mask(
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    abs_tol: float = 0.5,
    rel_tol: float = 0.01,
) -> torch.Tensor:
    """Validate current->previous flow with previous->current round trips.

    ``backward_flow`` is defined on current-frame pixels and points into the
    previous frame. ``forward_flow`` is defined on previous-frame pixels.
    """
    if forward_flow.shape != backward_flow.shape:
        raise ValueError("forward and backward flows must have identical shapes")
    forward_at_previous, in_bounds = backward_warp(forward_flow, backward_flow)
    roundtrip = backward_flow + forward_at_previous
    error = torch.linalg.vector_norm(roundtrip, dim=1, keepdim=True)
    motion = (
        torch.linalg.vector_norm(backward_flow, dim=1, keepdim=True)
        + torch.linalg.vector_norm(forward_at_previous, dim=1, keepdim=True)
    )
    threshold = float(abs_tol) + float(rel_tol) * motion
    return in_bounds & (error <= threshold)


def temporal_alignment_mask(
    current_rgb: torch.Tensor,
    previous_rgb: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    photo_tol: float = 40.0,
    flow_abs_tol: float = 0.5,
    flow_rel_tol: float = 0.01,
) -> torch.Tensor:
    """Return a soft current-frame mask for trustworthy temporal priors."""
    if current_rgb.shape != previous_rgb.shape:
        raise ValueError("current and previous RGB tensors must have identical shapes")
    previous_at_current, in_bounds = backward_warp(previous_rgb, backward_flow)
    flow_valid = forward_backward_consistency_mask(
        forward_flow,
        backward_flow,
        abs_tol=flow_abs_tol,
        rel_tol=flow_rel_tol,
    )
    photo_error = (current_rgb - previous_at_current).abs().mean(dim=1, keepdim=True)
    photo_weight = (1.0 - photo_error / max(float(photo_tol), 1e-6)).clamp(0.0, 1.0)
    return photo_weight * in_bounds.to(photo_weight.dtype) * flow_valid.to(photo_weight.dtype)
