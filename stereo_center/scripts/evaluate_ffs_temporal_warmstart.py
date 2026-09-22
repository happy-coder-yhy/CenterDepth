#!/usr/bin/env python
"""Evaluate SEA-RAFT disparity warm starts inside native FFS refinement."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stereo_center import calib, ffs_inference
from stereo_center.orbbec import load_pts_us, match_left_to_right_pts, pts_sidecar_path
from stereo_center.sea_raft_flow import flow_between, load_sea_raft
from stereo_center.temporal_stereo import backward_warp, temporal_alignment_mask
from stereo_center.visualize import colorize_depth_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-right", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--ffs-root", required=True)
    parser.add_argument("--sea-raft-root", required=True)
    parser.add_argument("--sea-raft-weights", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--sea-raft-iters", type=int, default=4)
    parser.add_argument("--sea-raft-input-scale", type=float, default=0.5)
    parser.add_argument("--prior-blend", type=float, default=0.75)
    parser.add_argument("--stats-spatial-stride", type=int, default=8)
    parser.add_argument("--photo-tol", type=float, default=25.0)
    parser.add_argument("--flow-abs-tol", type=float, default=0.5)
    parser.add_argument("--flow-rel-tol", type=float, default=0.01)
    parser.add_argument("--disp-abs-tol", type=float, default=3.0)
    parser.add_argument("--disp-rel-tol", type=float, default=0.15)
    return parser.parse_args()


def read_index(cap: cv2.VideoCapture, previous_index: int | None, target_index: int) -> np.ndarray:
    reads = target_index + 1 if previous_index is None else target_index - previous_index
    if reads < 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        reads = 1
    frame = None
    for _ in range(reads):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Unable to decode source frame {target_index}")
    assert frame is not None
    return frame


def native_ffs(
    wrapper: ffs_inference.FFSModel,
    left: torch.Tensor,
    right: torch.Tensor,
    init_disp_full: torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    """Run the native FFS model and optionally replace its 1/4-scale initializer."""
    padder = ffs_inference._padder(left.shape)
    if init_disp_full is None:
        left_pad, right_pad = padder.pad(left, right)
        init_disp = None
    else:
        left_pad, right_pad, init_full_pad = padder.pad(left, right, init_disp_full)
        init_disp = F.interpolate(
            init_full_pad, scale_factor=0.25, mode="bilinear", align_corners=True
        ) * 0.25
    left_pad = left_pad.cuda().contiguous()
    right_pad = right_pad.cuda().contiguous()
    if init_disp is not None:
        init_disp = init_disp.cuda().contiguous()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = wrapper.model.forward(
            left_pad,
            right_pad,
            iters=wrapper.valid_iters,
            test_mode=True,
            low_memory=False,
            init_disp=init_disp,
            optimize_build_volume=wrapper.volume_backend,
        )
    torch.cuda.synchronize()
    disparity = ffs_inference._extract_disparity(output).unsqueeze(1)
    return padder.unpad(disparity).float(), time.perf_counter() - started


def depth_from_disparity(disparity: torch.Tensor, fx: float, baseline: float) -> torch.Tensor:
    return float(fx * baseline) / disparity.clamp_min(1e-3)


def describe(values: list[torch.Tensor]) -> dict[str, float | int]:
    samples = torch.cat([value.flatten().cpu() for value in values])
    stride = max(1, samples.numel() // 4_000_000)
    quantile_samples = samples[::stride]
    return {
        "mean": float(samples.mean()),
        "median": float(torch.quantile(quantile_samples, 0.5)),
        "p95": float(torch.quantile(quantile_samples, 0.95)),
        "p99": float(torch.quantile(quantile_samples, 0.99)),
        "count": int(samples.numel()),
        "quantile_sample_count": int(quantile_samples.numel()),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.prior_blend <= 1.0:
        raise ValueError("--prior-blend must be in [0, 1]")
    if args.stats_spatial_stride < 1:
        raise ValueError("--stats-spatial-stride must be positive")
    calibration = (
        calib.load_orbbec_calibration(args.calib)
        if Path(args.calib).suffix.lower() in {".yaml", ".yml"}
        else calib.load_vdego_calibration(args.calib)
    )
    output_size = (
        max(32, int(calibration["resolution"][0] * args.scale)),
        max(32, int(calibration["resolution"][1] * args.scale)),
    )
    rectification = calib.compute_rectification_maps(calibration, output_size=output_size)
    fx, baseline = float(rectification["fx"]), float(rectification["baseline"])

    left_pts = load_pts_us(pts_sidecar_path(args.video))
    right_pts = load_pts_us(pts_sidecar_path(args.video_right))
    pairs, _ = match_left_to_right_pts(left_pts, right_pts, 1000)
    pairs = pairs[: args.max_frames]
    if len(pairs) < 2:
        raise ValueError("At least two PTS-paired frames are required")

    ffs = ffs_inference.load_ffs(
        weights_dir=args.weights,
        ffs_root=args.ffs_root,
        device="cuda",
        valid_iters=args.iters,
    )
    sea_raft = load_sea_raft(
        args.sea_raft_weights,
        args.sea_raft_root,
        Path(args.sea_raft_root) / "config/eval/spring-M.json",
        "cuda",
    )
    cap_left, cap_right = cv2.VideoCapture(args.video), cv2.VideoCapture(args.video_right)
    if not cap_left.isOpened() or not cap_right.isOpened():
        raise RuntimeError("Unable to open stereo videos")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fps = cap_left.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer = cv2.VideoWriter(
        str(outdir / "raw_ffs_depth.mp4"), fourcc, fps, output_size
    )
    warm_writer = cv2.VideoWriter(
        str(outdir / "ffs_warmstart_depth.mp4"), fourcc, fps, output_size
    )
    if not raw_writer.isOpened() or not warm_writer.isOpened():
        raise RuntimeError("Unable to create depth video writers")

    raw_changes: list[torch.Tensor] = []
    warm_changes: list[torch.Tensor] = []
    updates: list[torch.Tensor] = []
    accepted_pixels = 0
    candidate_pixels = 0
    ffs_raw_seconds = 0.0
    ffs_warm_seconds = 0.0
    flow_seconds = 0.0
    previous_left_index = previous_right_index = None
    previous_rgb = previous_raw_disparity = previous_raw_depth = previous_warm_depth = None

    for output_index, (left_index, right_index) in enumerate(pairs):
        left = read_index(cap_left, previous_left_index, left_index)
        right = read_index(cap_right, previous_right_index, right_index)
        previous_left_index, previous_right_index = left_index, right_index
        rectified_left, rectified_right = calib.rectify_pair(left, right, rectification)
        current_rgb = torch.from_numpy(
            cv2.cvtColor(rectified_left, cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1).float().unsqueeze(0).cuda()
        current_right = torch.from_numpy(
            cv2.cvtColor(rectified_right, cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1).float().unsqueeze(0)

        raw_disparity, elapsed = native_ffs(ffs, current_rgb.cpu(), current_right, None)
        ffs_raw_seconds += elapsed
        raw_depth = depth_from_disparity(raw_disparity, fx, baseline)
        warm_depth = raw_depth
        if previous_rgb is not None:
            started = time.perf_counter()
            previous_to_current, _ = flow_between(
                sea_raft, previous_rgb, current_rgb,
                iters=args.sea_raft_iters, input_scale=args.sea_raft_input_scale,
            )
            current_to_previous, _ = flow_between(
                sea_raft, current_rgb, previous_rgb,
                iters=args.sea_raft_iters, input_scale=args.sea_raft_input_scale,
            )
            torch.cuda.synchronize()
            flow_seconds += time.perf_counter() - started
            previous_disp = previous_raw_disparity.cuda()
            warped_disp, in_bounds = backward_warp(previous_disp, current_to_previous)
            alignment = temporal_alignment_mask(
                current_rgb, previous_rgb, previous_to_current, current_to_previous,
                photo_tol=args.photo_tol,
                flow_abs_tol=args.flow_abs_tol,
                flow_rel_tol=args.flow_rel_tol,
            )
            current_disp = raw_disparity.cuda()
            residual = (warped_disp - current_disp).abs()
            threshold = torch.maximum(
                torch.full_like(current_disp, args.disp_abs_tol),
                current_disp * args.disp_rel_tol,
            )
            accepted = in_bounds & torch.isfinite(warped_disp) & (warped_disp > 0.0) & (residual <= threshold)
            weight = accepted.to(alignment.dtype) * alignment * args.prior_blend
            init_full = (1.0 - weight) * current_disp + weight * warped_disp
            warm_disparity, elapsed = native_ffs(ffs, current_rgb.cpu(), current_right, init_full)
            ffs_warm_seconds += elapsed
            warm_depth = depth_from_disparity(warm_disparity, fx, baseline)
            accepted_pixels += int((weight > 0).sum().item())
            candidate_pixels += int(weight.numel())

            valid = (
                torch.isfinite(raw_depth)
                & torch.isfinite(previous_raw_depth)
                & torch.isfinite(warm_depth)
                & (raw_depth >= 0.3)
                & (raw_depth <= 20.0)
                & (previous_raw_depth >= 0.3)
                & (previous_raw_depth <= 20.0)
                & (warm_depth >= 0.3)
                & (warm_depth <= 20.0)
            )
            sampled_valid = valid[..., :: args.stats_spatial_stride, :: args.stats_spatial_stride]
            raw_changes.append(
                (raw_depth - previous_raw_depth).abs()[..., :: args.stats_spatial_stride, :: args.stats_spatial_stride][sampled_valid]
            )
            warm_changes.append(
                (warm_depth - previous_warm_depth).abs()[..., :: args.stats_spatial_stride, :: args.stats_spatial_stride][sampled_valid]
            )
            updates.append(
                (warm_depth - raw_depth).abs()[..., :: args.stats_spatial_stride, :: args.stats_spatial_stride][sampled_valid]
            )

        previous_rgb = current_rgb.detach()
        previous_raw_disparity = raw_disparity.detach()
        previous_raw_depth = raw_depth.detach()
        previous_warm_depth = warm_depth.detach()
        raw_np = raw_depth[0, 0].cpu().numpy()
        warm_np = warm_depth[0, 0].cpu().numpy()
        raw_writer.write(colorize_depth_log(raw_np, np.isfinite(raw_np) & (raw_np > 0.0)))
        warm_writer.write(colorize_depth_log(warm_np, np.isfinite(warm_np) & (warm_np > 0.0)))
        if (output_index + 1) % 24 == 0:
            print(f"[progress] {output_index + 1}/{len(pairs)}")

    cap_left.release()
    cap_right.release()
    raw_writer.release()
    warm_writer.release()
    result = {
        "frames": len(pairs),
        "prior_blend": args.prior_blend,
        "stats_spatial_stride": args.stats_spatial_stride,
        "accepted_weight_ratio": accepted_pixels / candidate_pixels,
        "same_pixel_adjacent_depth_change_m": {
            "raw_ffs": describe(raw_changes),
            "ffs_warmstart": describe(warm_changes),
        },
        "warmstart_output_minus_raw_depth_m": describe(updates),
        "timing_seconds": {
            "raw_ffs": ffs_raw_seconds,
            "warmstart_ffs": ffs_warm_seconds,
            "sea_raft_flow": flow_seconds,
        },
    }
    result["same_pixel_adjacent_depth_change_m"]["reduction_pct"] = 100.0 * (
        1.0 - result["same_pixel_adjacent_depth_change_m"]["ffs_warmstart"]["mean"]
        / result["same_pixel_adjacent_depth_change_m"]["raw_ffs"]["mean"]
    )
    (outdir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
