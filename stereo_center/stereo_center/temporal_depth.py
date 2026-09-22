"""Conservative output-side temporal fusion for metric depth experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .temporal_stereo import backward_warp, temporal_alignment_mask


@dataclass(frozen=True)
class TemporalDepthFusion:
    depth: torch.Tensor
    weight: torch.Tensor
    accepted: torch.Tensor


def _valid_depth(
    depth: torch.Tensor, valid: torch.Tensor | None, max_depth: float | None
) -> torch.Tensor:
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape (B, 1, H, W)")
    mask = torch.isfinite(depth) & (depth > 0.0)
    if valid is not None:
        if valid.shape != depth.shape:
            raise ValueError("depth valid mask must match depth shape")
        mask &= valid.bool()
    if max_depth is not None and float(max_depth) > 0.0:
        mask &= depth <= float(max_depth)
    return mask


def fuse_previous_depth(
    current_depth: torch.Tensor,
    previous_depth: torch.Tensor,
    current_rgb: torch.Tensor,
    previous_rgb: torch.Tensor,
    previous_to_current_flow: torch.Tensor,
    current_to_previous_flow: torch.Tensor,
    *,
    current_valid: torch.Tensor | None = None,
    previous_valid: torch.Tensor | None = None,
    max_prior_weight: float = 0.2,
    photo_tol: float = 25.0,
    flow_abs_tol: float = 0.5,
    flow_rel_tol: float = 0.01,
    depth_abs_tol: float = 0.05,
    depth_rel_tol: float = 0.05,
    max_depth: float | None = 20.0,
) -> TemporalDepthFusion:
    """Warp raw previous depth into current coordinates and weakly fuse it."""
    if not 0.0 <= float(max_prior_weight) <= 1.0:
        raise ValueError("max_prior_weight must be in [0, 1]")
    if previous_depth.shape != current_depth.shape:
        raise ValueError("current and previous depth must have identical shapes")
    if current_rgb.shape != previous_rgb.shape:
        raise ValueError("current and previous RGB must have identical shapes")
    if current_rgb.shape[0] != current_depth.shape[0] or current_rgb.shape[-2:] != current_depth.shape[-2:]:
        raise ValueError("RGB and depth must share batch and spatial dimensions")

    current_ok = _valid_depth(current_depth, current_valid, max_depth)
    previous_ok = _valid_depth(previous_depth, previous_valid, max_depth)
    previous_inverse = torch.where(
        previous_ok, previous_depth.reciprocal(), torch.zeros_like(previous_depth)
    )
    warped_inverse_sum, in_bounds = backward_warp(previous_inverse, current_to_previous_flow)
    warped_valid, _ = backward_warp(previous_ok.float(), current_to_previous_flow)
    sampled_ok = in_bounds & (warped_valid >= 1.0 - 1e-5)
    prior_inverse = warped_inverse_sum / warped_valid.clamp_min(1e-6)
    prior_depth = prior_inverse.reciprocal()

    alignment = temporal_alignment_mask(
        current_rgb,
        previous_rgb,
        previous_to_current_flow,
        current_to_previous_flow,
        photo_tol=photo_tol,
        flow_abs_tol=flow_abs_tol,
        flow_rel_tol=flow_rel_tol,
    )
    residual = (prior_depth - current_depth).abs()
    depth_threshold = torch.maximum(
        torch.full_like(current_depth, float(depth_abs_tol)),
        current_depth * float(depth_rel_tol),
    )
    accepted = (
        current_ok
        & sampled_ok
        & torch.isfinite(prior_depth)
        & (prior_depth > 0.0)
        & (residual <= depth_threshold)
    )
    weight = alignment * accepted.to(alignment.dtype) * float(max_prior_weight)
    current_inverse = torch.where(current_ok, current_depth.reciprocal(), torch.zeros_like(current_depth))
    fused_inverse = (1.0 - weight) * current_inverse + weight * prior_inverse
    fused = torch.where(current_ok, fused_inverse.reciprocal(), current_depth)
    return TemporalDepthFusion(depth=fused, weight=weight, accepted=accepted)
