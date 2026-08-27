#!/usr/bin/env python
"""批量生成中心视角深度视频：校正 → 批量 WAFT 双向视差 → 中心融合 → 深度帧 → mp4。

用法示例（在 stereo_center/ 目录下）：
    conda run -n waft python scripts/run_depth_video.py \
        --video ../dataset/xxx/output.mp4 --calib ../dataset/xxx/calibration.json \
        --scale 0.5 --batch-size 4 --outdir outputs/depth_video

- 深度值恒为米制：depth = fx * baseline / disparity。
- 色阶为固定对数米制映射（0.3~20m，可调）：场景深度范围大时线性色阶
  顾此失彼，对数映射让近场/背景都有可分辨色带，且跨帧同深度同色。
- 时间平滑默认关闭（逐帧深度）：时间滤波虽能压抖动，但会引入拖影/重影。
- 遮挡填充与融合都在 GPU 上批量执行。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "third_party/s2m2/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stereo_center import calib, softsplat, stereo_backend  # noqa: E402
from stereo_center.gpu_memory import (  # noqa: E402
    gpu_peak_memory_gib,
    reset_gpu_peak_memory,
)
from stereo_center.orbbec import (  # noqa: E402
    forward_decode_read_count,
    load_pts_us,
    match_left_to_right_pts,
    pts_metadata_mismatch,
    pts_sidecar_path,
)
from stereo_center.pipeline import photometric_align_right  # noqa: E402
from stereo_center.guided_filter import guided_filter  # noqa: E402
from stereo_center.left_hole_fill import fill_small_left_holes  # noqa: E402
from stereo_center.raft_flow import flow_between, load_raft  # noqa: E402
from stereo_center.visualize import (  # noqa: E402
    colorize_depth_log,
    make_depth_colorbar_log,
)


def resolve_weights_dir(explicit: str | None, backend: str) -> Path:
    if backend in ("opencv_bm", "opencv_sgbm"):
        return Path(".")
    if explicit:
        return Path(explicit)
    if backend == "stereonet":
        return PROJECT_ROOT.parent / "weights" / "stereonet"
    env_map = {
        "waft": "WAFT_WEIGHTS_DIR",
        "s2m2": "S2M2_WEIGHTS_DIR",
        "las2": "LAS2_WEIGHTS_DIR",
        "ffs": "FFS_WEIGHTS_DIR",
    }
    env = env_map[backend]
    if env in os.environ:
        return Path(os.environ[env])
    repo_root = PROJECT_ROOT.parent
    if backend == "las2":
        subs = ["las2", "pretrain_weights"]
    elif backend == "waft":
        subs = ["waft"]
    elif backend == "ffs":
        subs = ["fast_foundation_stereo", "pretrain_weights"]
    else:
        subs = ["pretrain_weights"]
    for sub in subs:
        for c in (repo_root / "weights" / sub, PROJECT_ROOT / "weights" / sub):
            if c.exists():
                return c
    # 兜底：找目录下含对应权重文件的位置
    fname = f"LAS2_{'M'}.pth" if backend == "las2" else None
    for c in (repo_root / "weights", PROJECT_ROOT / "weights"):
        if c.exists():
            if backend != "las2" or any(c.rglob(fname)):
                return c
    raise FileNotFoundError(f"未找到权重目录（{backend}），请用 --weights 或环境变量 {env}")


def timing_artifact_name(backend: str) -> str:
    """Return backend-specific timing artifact filename."""
    return f"{backend}_timing.json"


def add_model_iteration_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the public model refinement iteration flag and legacy aliases."""
    parser.add_argument(
        "--iters", type=int, default=None,
        help=(
            "模型测试阶段视差迭代细化轮数；WAFT 默认取模型配置，"
            "FFS 默认 8"
        ),
    )
    parser.add_argument("--waft-iters", dest="iters", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--ffs-valid-iters", dest="iters", type=int, help=argparse.SUPPRESS)


def add_stereonet_arguments(parser: argparse.ArgumentParser) -> None:
    """Add pinned StereoNet source and resize controls."""
    parser.add_argument(
        "--stereonet-root", type=str, default=None,
        help="StereoNet_PyTorch 固定源码根目录（默认 third_party/StereoNet_PyTorch）",
    )
    parser.add_argument(
        "--stereonet-max-side", type=int, default=625,
        help="StereoNet 输入最长边（Scene Flow RGB 权重默认 625）",
    )


def add_opencv_bm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add OpenCV StereoBM parameters in OpenCV's native units."""
    parser.add_argument("--bm-num-disparities", type=int, default=128)
    parser.add_argument("--bm-block-size", type=int, default=31)
    parser.add_argument("--bm-uniqueness-ratio", type=int, default=10)
    parser.add_argument("--bm-speckle-window-size", type=int, default=100)
    parser.add_argument("--bm-speckle-range", type=int, default=2)
    parser.add_argument("--bm-disp12-max-diff", type=int, default=1)


def add_opencv_sgbm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add OpenCV StereoSGBM parameters in OpenCV's native units."""
    parser.add_argument("--sgbm-min-disparity", type=int, default=0)
    parser.add_argument("--sgbm-num-disparities", type=int, default=128)
    parser.add_argument("--sgbm-block-size", type=int, default=5)
    parser.add_argument("--sgbm-p1", type=int, default=None)
    parser.add_argument("--sgbm-p2", type=int, default=None)
    parser.add_argument("--sgbm-disp12-max-diff", type=int, default=1)
    parser.add_argument("--sgbm-uniqueness-ratio", type=int, default=10)
    parser.add_argument("--sgbm-speckle-window-size", type=int, default=100)
    parser.add_argument("--sgbm-speckle-range", type=int, default=2)
    parser.add_argument(
        "--sgbm-mode", choices=("sgbm", "hh", "3way", "hh4"), default="3way"
    )


def add_left_hole_fill_arguments(parser: argparse.ArgumentParser) -> None:
    """Add conservative left-view visualization hole-fill controls."""
    parser.add_argument(
        "--left-hole-fill", type=int, default=0, choices=[0, 1],
        help="左视角小孔洞 RGB 门控补全（仅美化可视化，默认关闭）",
    )
    parser.add_argument(
        "--left-hole-fill-max-area", type=int, default=256,
        help="允许补全的无效连通域最大面积（像素）",
    )
    parser.add_argument(
        "--left-hole-fill-color-tol", type=float, default=20.0,
        help="孔洞与有效边界的灰度差容差（0-255）",
    )


def opencv_bm_parameters(args) -> dict[str, int]:
    """Return the StereoBM configuration stored with an experiment artifact."""
    return {
        "num_disparities": int(args.bm_num_disparities),
        "block_size": int(args.bm_block_size),
        "uniqueness_ratio": int(args.bm_uniqueness_ratio),
        "speckle_window_size": int(args.bm_speckle_window_size),
        "speckle_range": int(args.bm_speckle_range),
        "disp12_max_diff": int(args.bm_disp12_max_diff),
    }


def opencv_sgbm_parameters(args) -> dict[str, int | str]:
    """Return the effective StereoSGBM configuration stored with an artifact."""
    block_size = int(args.sgbm_block_size)
    p1 = args.sgbm_p1
    p2 = args.sgbm_p2
    return {
        "min_disparity": int(args.sgbm_min_disparity),
        "num_disparities": int(args.sgbm_num_disparities),
        "block_size": block_size,
        "p1": int(p1) if p1 is not None else 8 * 3 * block_size * block_size,
        "p2": int(p2) if p2 is not None else 32 * 3 * block_size * block_size,
        "disp12_max_diff": int(args.sgbm_disp12_max_diff),
        "uniqueness_ratio": int(args.sgbm_uniqueness_ratio),
        "speckle_window_size": int(args.sgbm_speckle_window_size),
        "speckle_range": int(args.sgbm_speckle_range),
        "mode": args.sgbm_mode,
    }


def left_hole_fill_parameters(args) -> dict[str, bool | int | float]:
    """Return left-view small-hole fill configuration stored with an artifact."""
    return {
        "enabled": bool(args.left_hole_fill),
        "max_area": int(args.left_hole_fill_max_area),
        "color_tol": float(args.left_hole_fill_color_tol),
    }


def resolve_model_iters(backend: str, iters: int | None) -> int | None:
    """Resolve backend defaults while keeping the public CLI name as --iters."""
    if iters is not None:
        return int(iters)
    if backend == "ffs":
        return 8
    return None


def validate_backend_mode(args) -> None:
    """Keep one-way baselines on their intended left-view route."""
    if args.stereo_backend in ("opencv_bm", "opencv_sgbm", "stereonet") and (
        args.output_view != "left" or args.bi != 0
    ):
        raise ValueError(
            f"{args.stereo_backend} only supports --output-view left with --bi=0"
        )


def validate_left_hole_fill_mode(args) -> None:
    """Prevent a left-reference visualization filter from touching center depth."""
    if args.left_hole_fill and args.output_view != "left":
        raise ValueError("--left-hole-fill is available only for left-view output")


def resolve_processing_end(
    n_total: int, start_frame: int, requested_end: int, max_frames: int
) -> int:
    """Limit the requested range to frames that actually have stereo pairs."""
    end = n_total if requested_end < 0 else min(requested_end, n_total)
    if max_frames > 0:
        end = min(end, start_frame + max_frames)
    return end


def validate_waft_temporal_mode(args, hiera_mode: str) -> None:
    """Reject combinations that cannot preserve WAFT temporal semantics."""
    if not args.waft_temporal_init:
        return
    if args.stereo_backend != "waft":
        raise ValueError("--waft-temporal-init is available only for the WAFT backend")
    if not args.bi:
        raise ValueError("--waft-temporal-init requires --bi 1")
    if hiera_mode != "direct":
        raise ValueError("--waft-temporal-init supports direct WAFT inference only")
    if args.temporal_raft:
        raise ValueError(
            "--waft-temporal-init cannot be combined with final-depth --temporal-raft smoothing"
        )


def add_waft_temporal_arguments(parser: argparse.ArgumentParser) -> None:
    """Add WAFT warm-start controls without changing the default pipeline."""
    parser.add_argument(
        "--waft-temporal-init", type=int, default=0, choices=[0, 1],
        help="用相邻帧视差作为 WAFT 粗视差初始化（不平滑最终深度）",
    )
    parser.add_argument(
        "--waft-temporal-flow-iters", type=int, default=12,
        help="WAFT 时序初始化的 RAFT 光流迭代次数",
    )
    parser.add_argument(
        "--waft-temporal-blend", type=float, default=0.75,
        help="有效时序先验在 WAFT 粗初始化中的权重",
    )
    parser.add_argument(
        "--waft-temporal-photo-tol", type=float, default=40.0,
        help="时序先验光度一致性容差（RGB 0-255）",
    )
    parser.add_argument(
        "--waft-temporal-flow-abs-tol", type=float, default=0.5,
        help="前后向光流一致性绝对容差（像素）",
    )
    parser.add_argument(
        "--waft-temporal-flow-rel-tol", type=float, default=0.01,
        help="前后向光流一致性相对容差",
    )
    parser.add_argument(
        "--waft-temporal-disp-abs-tol", type=float, default=3.0,
        help="历史先验与当前粗视差的一致性绝对容差（像素）",
    )
    parser.add_argument(
        "--waft-temporal-disp-rel-tol", type=float, default=0.15,
        help="历史先验与当前粗视差的一致性相对容差",
    )


def waft_temporal_kwargs(args, temporal_flow_model, temporal_state) -> dict:
    """Build optional WAFT arguments so the non-temporal path stays unchanged."""
    if not args.waft_temporal_init:
        return {}
    return {
        "temporal_flow_model": temporal_flow_model,
        "temporal_state": temporal_state,
        "temporal_flow_iters": args.waft_temporal_flow_iters,
        "temporal_blend": args.waft_temporal_blend,
        "temporal_photo_tol": args.waft_temporal_photo_tol,
        "temporal_flow_abs_tol": args.waft_temporal_flow_abs_tol,
        "temporal_flow_rel_tol": args.waft_temporal_flow_rel_tol,
        "temporal_disp_abs_tol": args.waft_temporal_disp_abs_tol,
        "temporal_disp_rel_tol": args.waft_temporal_disp_rel_tol,
    }


def weighted_temporal_valid_ratio(records: list[dict]) -> float | None:
    """Average valid-prior coverage by frame count, including short final batches."""
    weighted = 0.0
    frames = 0
    for record in records:
        ratio = record.get("temporal_valid_ratio")
        if ratio is None:
            continue
        batch_size = int(record["batch_size"])
        weighted += float(ratio) * batch_size
        frames += batch_size
    return weighted / frames if frames else None


def warp_with_flow(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """src: (1,1,H,W)；flow: (1,2,H,W) 像素位移 -> 采样结果 (1,1,H,W)。"""
    B, C, H, W = src.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=src.device),
        torch.arange(W, device=src.device),
        indexing="ij",
    )
    gx = ((xx.float() + flow[:, 0]) / max(W - 1, 1) * 2 - 1).clamp(-1, 1)
    gy = ((yy.float() + flow[:, 1]) / max(H - 1, 1) * 2 - 1).clamp(-1, 1)
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(
        src, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )


def left_view_depth_from_disparity(
    disp: torch.Tensor,
    occ: torch.Tensor,
    fx: float,
    baseline: float,
    device: str,
    valid_mode: str = "strict",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert left-reference disparity to metric left-camera depth.

    disp/occ are (B,H,W) CPU or GPU tensors.  Output depth/valid are
    (B,1,H,W) on ``device``.  Invalid pixels remain in depth for color scaling,
    but ``valid`` marks them out for visualization.
    """
    disp_d = disp.to(device).float()
    occ_d = occ.to(device).float()
    if valid_mode == "strict":
        valid = torch.isfinite(disp_d) & (disp_d > 1e-6) & (occ_d > 0.5)
    elif valid_mode == "paper":
        # Paper/demo-style visualization: show the predicted disparity map itself.
        # Do not black out pixels only because the one-way visibility check says
        # their correspondence falls outside the image.  This is a visualization
        # choice, not a change to WAFT inference.
        valid = torch.isfinite(disp_d) & (disp_d > 1e-6)
    else:
        raise ValueError(f"unknown left-view valid mode: {valid_mode}")
    depth = float(fx) * float(baseline) / disp_d.clamp_min(1e-6)
    return depth.unsqueeze(1), valid.unsqueeze(1)


def process_batch(
    model,
    bgr_pairs: list,
    rect: dict,
    fx: float,
    baseline: float,
    args,
    temporal_flow_model=None,
    temporal_state: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """校正后 BGR 对列表 → GPU 批量推理 + 融合 + 遮挡填充。

    Returns:
        dep_b: (B, 1, H, W) float32 GPU 深度（米）；
        valid_b: (B, 1, H, W) bool GPU 有效掩码；
        rgb_b: (B, 3, H, W) float32 GPU RGB（0-255）。
    """
    timing = {
        "stereo_forward": 0.0,
        "stereo_total": 0.0,
        "stereo_detail": None,
        "photo_align": 0.0,
        "center_fusion": 0.0,
        "fill": 0.0,
        "depth_gf": 0.0,
    }

    B = len(bgr_pairs)
    H, W = bgr_pairs[0][0].shape[:2]
    left_t = torch.zeros(B, 3, H, W)
    right_t = torch.zeros(B, 3, H, W)
    for b, (rL, rR) in enumerate(bgr_pairs):
        left_t[b] = torch.from_numpy(cv2.cvtColor(rL, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        right_t[b] = torch.from_numpy(cv2.cvtColor(rR, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()

    if args.stereo_backend == "waft":
        from stereo_center import waft_inference  # noqa: E402  # 懒加载：仅 waft 路径需要 peft 等

        if args.bi:
            waft_detail = {}
            dL, dR, occL, occR, confL, confR, _ = waft_inference.run_stereo_matching_bi_batch(
                model, left_t, right_t, args.device,
                hiera=args.hiera, conf_mode=args.conf, occ_mode=args.occ,
                timing_out=waft_detail,
                **waft_temporal_kwargs(args, temporal_flow_model, temporal_state),
            )
            timing["stereo_forward"] += waft_detail.get("model_forward_seconds", 0.0)
            timing["stereo_total"] += waft_detail.get("waft_total_seconds", 0.0)
            timing["stereo_detail"] = waft_detail
        else:
            waft_detail = {}
            dL, occL, confL, _ = waft_inference.run_stereo_matching_batch(
                model, left_t, right_t, args.device,
                hiera=args.hiera, conf_mode="ones", occ_mode="visibility",
                timing_out=waft_detail,
            )
            timing["stereo_forward"] += waft_detail.get("model_forward_seconds", 0.0)
            timing["stereo_total"] += waft_detail.get("waft_total_seconds", 0.0)
            timing["stereo_detail"] = waft_detail
            dR = occR = confR = None
    else:
        if args.bi:
            t0 = time.perf_counter()
            dL, dR, occL, occR, confL, confR, _ = stereo_backend.run_bi_batch(
                args.stereo_backend, model, left_t, right_t, args.device,
                max_disp=args.max_disp, conf_mode=args.conf, occ_mode=args.occ,
                hiera=args.hiera,
            )
            elapsed = time.perf_counter() - t0
            timing["stereo_forward"] += elapsed
            timing["stereo_total"] += elapsed
        else:
            t0 = time.perf_counter()
            stereo_detail = {}
            dL, occL, confL, elapsed = stereo_backend.run(
                args.stereo_backend, model, left_t, right_t, args.device,
                max_disp=args.max_disp, conf_mode=args.conf, occ_mode=args.occ,
                hiera=args.hiera,
                max_side=args.stereonet_max_side,
                stereonet_root=args.stereonet_root,
                timing_out=stereo_detail,
                bm_num_disparities=args.bm_num_disparities,
                bm_block_size=args.bm_block_size,
                bm_uniqueness_ratio=args.bm_uniqueness_ratio,
                bm_speckle_window_size=args.bm_speckle_window_size,
                bm_speckle_range=args.bm_speckle_range,
                bm_disp12_max_diff=args.bm_disp12_max_diff,
            )
            measured = time.perf_counter() - t0
            timing["stereo_forward"] += stereo_detail.get(
                "model_forward_seconds", elapsed
            )
            timing["stereo_total"] += stereo_detail.get(
                "stereo_total_seconds", measured
            )
            timing["stereo_detail"] = stereo_detail or {
                "stereo_forward_seconds": elapsed,
                "stereo_total_seconds": measured,
            }
            dR = occR = confR = None
    if args.guided_filter:
        from stereo_center.guided_filter import guided_filter_batch  # noqa: E402

        dL, dR = guided_filter_batch(
            bgr_pairs, dL.numpy(), dR.numpy() if dR is not None else None,
            args.gf_radius, args.gf_eps,
        )
        dL = torch.from_numpy(dL)
        dR = torch.from_numpy(dR) if dR is not None else None
    dev = args.device
    if args.output_view == "left":
        dep_b, valid_b = left_view_depth_from_disparity(
            dL, occL, fx, baseline, dev, valid_mode=args.left_vis_mode
        )
        rgb_b = left_t.to(dev)
        return dep_b, valid_b, rgb_b, timing

    # 保持逐帧光度校正，但将中心视角融合本身一次性按 B 帧执行，
    # 避免每帧重复创建投影网格、权重和 z-buffer 中间张量。
    t0 = time.perf_counter()
    right_f_cpu = torch.stack([
        torch.from_numpy(
            cv2.cvtColor(
                photometric_align_right(rL_bgr, rR_bgr), cv2.COLOR_BGR2RGB
            )
        ).permute(2, 0, 1).float()
        for rL_bgr, rR_bgr in bgr_pairs
    ])
    timing["photo_align"] += time.perf_counter() - t0
    left_f = left_t.to(dev)
    right_f = right_f_cpu.to(dev)
    dl = dL.unsqueeze(1).to(dev)
    cl = confL.unsqueeze(1).to(dev)
    ol = occL.unsqueeze(1).to(dev)
    if dR is not None:
        dr = dR.unsqueeze(1).to(dev)
        cr = confR.unsqueeze(1).to(dev)
        orr = occR.unsqueeze(1).to(dev)
    else:
        dr = cr = orr = None
    t0 = time.perf_counter()
    rgb_b, dep_b, valid_b = softsplat.center_view(
        left_f, right_f, dl, cl, ol, fx=fx, baseline=baseline,
        disp_right=dr, conf_right=cr, occ_right=orr,
        edge_k=1.5, blend="softz", weight_mode="expdecay", weight_k=4.0,
        median_k=args.median_k,
        depth_z=bool(args.depth_z), depth_z_thresh=0.05, depth_z_power=2.0,
        color_tol=args.color_tol,
    )
    timing["center_fusion"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    rgb_b, dep_b, valid_b = softsplat.fill_disocclusion_torch(rgb_b, dep_b, valid_b)
    timing["fill"] += time.perf_counter() - t0
    if args.depth_gf:
        # 中心 RGB 引导滤波：深度边缘对齐到图像边缘，提升锐度
        dev = args.device
        t0 = time.perf_counter()
        for b in range(B):
            c_rgb = rgb_b[b].permute(1, 2, 0).cpu().numpy()
            c_gray = cv2.cvtColor(c_rgb, cv2.COLOR_RGB2GRAY)
            dep_np = dep_b[b, 0].cpu().numpy()
            q = guided_filter(c_gray, dep_np, args.depth_gf_radius, args.depth_gf_eps)
            if args.depth_unsharp > 0:
                # 边缘保留 unsharp：仅在有图像边缘处增强
                q = dep_np + args.depth_unsharp * (dep_np - q)
            dep_b[b, 0] = torch.from_numpy(q).to(dev)
        timing["depth_gf"] += time.perf_counter() - t0
    return dep_b, valid_b, rgb_b, timing


def main() -> None:
    parser = argparse.ArgumentParser(description="批量中心深度视频生成（米制色阶）")
    parser.add_argument("--video", type=str, required=True, help="SBS 视频，或独立左相机视频")
    parser.add_argument("--video-right", type=str, default=None, help="独立右相机视频；提供后按 PTS 配对")
    parser.add_argument("--left-pts", type=str, default=None, help="左视频 timestamp_us CSV（默认自动查找）")
    parser.add_argument("--right-pts", type=str, default=None, help="右视频 timestamp_us CSV（默认自动查找）")
    parser.add_argument(
        "--pts-match-tolerance-ms", type=float, default=8.0,
        help="独立双目流的最近 PTS 配对容差（毫秒）",
    )
    parser.add_argument("--calib", type=str, required=True, help="VDEgo JSON 或 Orbbec 相机 YAML")
    parser.add_argument("--scale", type=float, default=0.5, help="校正输出缩放（建议 0.5）")
    parser.add_argument("--batch-size", type=int, default=4, help="模型前向 batch（s0.5 建议 4）")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="-1=到视频末尾")
    parser.add_argument("--max-frames", type=int, default=0, help=">0 时最多处理 N 帧（调试用）")
    parser.add_argument("--fps", type=float, default=0.0, help="输出视频 fps（默认取源视频）")
    parser.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "outputs/depth_video"))
    parser.add_argument("--stereo-backend", type=str, default="waft", choices=["waft", "s2m2", "las2", "ffs", "stereonet", "opencv_bm", "opencv_sgbm"])
    parser.add_argument("--model-type", type=str, default="DAv2L-5")
    parser.add_argument("--max-disp", type=int, default=192, help="LAS2 最大视差（默认 192）")
    parser.add_argument("--las-root", type=str, default=None, help="LiteAnyStereo 仓库根目录（LAS2）")
    parser.add_argument("--ffs-root", type=str, default=None, help="Fast-FoundationStereo 仓库根目录（FFS）")
    add_stereonet_arguments(parser)
    add_model_iteration_arguments(parser)
    add_opencv_bm_arguments(parser)
    add_opencv_sgbm_arguments(parser)
    add_left_hole_fill_arguments(parser)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hiera", type=str, default="auto", choices=["auto", "direct", "hiera"])
    parser.add_argument("--bi", type=int, default=1, choices=[0, 1], help="双向视差（1=开，0=仅左方向，速度快约 2 倍）")
    parser.add_argument("--conf", type=str, default="lr", choices=["info", "ones", "lr"])
    parser.add_argument("--occ", type=str, default="lr", choices=["lr", "visibility"])
    parser.add_argument("--color-tol", type=float, default=15.0, help="softz 颜色冲突阈值")
    parser.add_argument("--median-k", type=int, default=0, help="融合前视差中值滤波核（奇数，0/1 关闭；3 可压抖动）")
    parser.add_argument("--guided-filter", type=int, default=0, choices=[0, 1], help="RGB 引导滤波锐化视差")
    parser.add_argument("--gf-radius", type=int, default=8, help="引导滤波窗口半径")
    parser.add_argument("--gf-eps", type=float, default=300.0, help="引导滤波正则项（0-255 灰度尺度）")
    parser.add_argument("--depth-gf", type=int, default=0, choices=[0, 1], help="融合后中心 RGB 引导滤波锐化深度")
    parser.add_argument("--depth-gf-radius", type=int, default=6, help="深度引导滤波窗口半径")
    parser.add_argument("--depth-gf-eps", type=float, default=200.0, help="深度引导滤波正则项")
    parser.add_argument("--depth-unsharp", type=float, default=0.0, help="深度边缘保留 unsharp 强度（0=关）")
    parser.add_argument("--temporal-raft", type=int, default=0, choices=[0, 1], help="RAFT 光流 + 一致性门控时间稳定（防闪）")
    parser.add_argument("--temporal-alpha", type=float, default=0.6, help="RAFT 时间 EMA 融合系数")
    parser.add_argument("--temporal-photo-tol", type=float, default=25.0, help="光度一致性门控容差（灰度差）")
    parser.add_argument("--temporal-depth-tol", type=float, default=0.3, help="深度一致性门控绝对容差（米）")
    parser.add_argument("--temporal-depth-rel", type=float, default=0.15, help="深度一致性门控相对容差")
    parser.add_argument(
        "--raft-weights", type=str, default=None,
        help="raft-things.pth 路径（RAFT 深度平滑或 WAFT 时序初始化）",
    )
    parser.add_argument("--raft-root", type=str, default=None, help="RAFT 仓库代码根目录")
    add_waft_temporal_arguments(parser)
    parser.add_argument("--depth-z", type=int, default=1, choices=[0, 1], help="深度 hard z-buffer")
    parser.add_argument(
        "--dmin-m", type=float, default=0.3,
        help="对数米制色阶下限（米，默认 0.3）",
    )
    parser.add_argument(
        "--dmax-m", type=float, default=20.0,
        help="对数米制色阶上限（米，默认 20；超出部分饱和为红色）",
    )
    parser.add_argument(
        "--temporal-median", type=int, default=1,
        help="时间维中值滤波窗口（奇数，1=关闭；时间平滑易引入拖影，默认关闭）",
    )
    parser.add_argument(
        "--temporal-ema", type=float, default=1.0,
        help="时间维 EMA：1=关闭（默认）；0=光流运动补偿（实验性，有重影风险）；"
        ">0 且 <1=固定系数",
    )
    parser.add_argument(
        "--flow-alpha", type=float, default=0.2,
        help="光流 EMA 融合系数（仅 --temporal-ema 0 时生效）",
    )
    parser.add_argument(
        "--spatial-median", type=int, default=1,
        help="逐帧深度空间 3x3 中值滤波核（奇数，1=关闭；默认关闭）",
    )
    parser.add_argument("--save-frames-every", type=int, default=50, help="每隔 N 帧存一张深度 PNG（0=不存）")
    parser.add_argument("--save-depth-npy", type=int, default=0, help="每帧额外保存米制深度 npy（供后处理时间平滑）")
    parser.add_argument("--video-name", type=str, default="depth_video.mp4")
    parser.add_argument(
        "--output-view",
        type=str,
        default="center",
        choices=["center", "left"],
        help="输出视角：center=中心视角合成；left=左相机视角深度（不做中心合成）",
    )
    parser.add_argument(
        "--left-vis-mode",
        type=str,
        default="strict",
        choices=["strict", "paper"],
        help="左视角可视化有效掩码：strict=显示严格有效区域；paper=论文/demo式展示预测图，减少黑点",
    )
    args = parser.parse_args()
    t_program = time.perf_counter()
    gpu_memory_tracking = reset_gpu_peak_memory(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    backend = args.stereo_backend
    validate_backend_mode(args)
    validate_left_hole_fill_mode(args)
    if backend in ("opencv_bm", "opencv_sgbm") and args.model_type == "DAv2L-5":
        args.model_type = "StereoBM" if backend == "opencv_bm" else "StereoSGBM"
    elif backend == "stereonet" and args.model_type == "DAv2L-5":
        args.model_type = "stereonet_sceneflow_rgb"
    model_iters = resolve_model_iters(backend, args.iters)

    t_model_load = 0.0
    t_temporal_flow_model_load = 0.0
    t_decode = 0.0
    t_rectify = 0.0
    t_waft = 0.0
    t_align = 0.0
    t_fusion = 0.0
    t_fill = 0.0
    t_left_hole_fill = 0.0
    t_depth_gf = 0.0
    t_color = 0.0
    t_write = 0.0
    t_png_write = 0.0
    left_hole_fill_components = 0
    left_hole_fill_pixels = 0
    waft_timing_records = []

    calib_loader = calib.load_orbbec_calibration if Path(args.calib).suffix.lower() in {".yaml", ".yml"} else calib.load_vdego_calibration
    cal = calib_loader(args.calib)
    out_size = (
        max(32, int(cal["resolution"][0] * args.scale)),
        max(32, int(cal["resolution"][1] * args.scale)),
    )
    rect = calib.compute_rectification_maps(cal, output_size=out_size)
    fx = rect["P1"][0, 0]
    baseline = rect["baseline"]
    H, W = out_size[1], out_size[0]
    print(f"[rect] {W}x{H}, fx={fx:.1f}, baseline={baseline:.4f} m")
    hiera_mode = args.hiera
    if hiera_mode == "auto":
        hiera_mode = "hiera" if max(H, W) > 1080 else "direct"
    validate_waft_temporal_mode(args, hiera_mode)

    weights_dir = resolve_weights_dir(args.weights, backend)
    print(f"[weights] {weights_dir}")
    t0 = time.perf_counter()
    model = stereo_backend.load(
        backend, args.model_type, str(weights_dir), args.device,
        num_refine=3, max_disp=args.max_disp, las_root=args.las_root,
        ffs_root=args.ffs_root,
        stereonet_root=args.stereonet_root,
        max_side=args.stereonet_max_side,
        valid_iters=model_iters,
        hiera=args.hiera,
        iters=model_iters,
        bm_num_disparities=args.bm_num_disparities,
        bm_block_size=args.bm_block_size,
        bm_uniqueness_ratio=args.bm_uniqueness_ratio,
        bm_speckle_window_size=args.bm_speckle_window_size,
        bm_speckle_range=args.bm_speckle_range,
        bm_disp12_max_diff=args.bm_disp12_max_diff,
        sgbm_min_disparity=args.sgbm_min_disparity,
        sgbm_num_disparities=args.sgbm_num_disparities,
        sgbm_block_size=args.sgbm_block_size,
        sgbm_p1=args.sgbm_p1,
        sgbm_p2=args.sgbm_p2,
        sgbm_disp12_max_diff=args.sgbm_disp12_max_diff,
        sgbm_uniqueness_ratio=args.sgbm_uniqueness_ratio,
        sgbm_speckle_window_size=args.sgbm_speckle_window_size,
        sgbm_speckle_range=args.sgbm_speckle_range,
        sgbm_mode=args.sgbm_mode,
    )
    t_model_load += time.perf_counter() - t0
    stereonet_metadata = None
    if backend == "stereonet":
        stereonet_metadata = {
            "max_side": int(args.stereonet_max_side),
            "checkpoint": model.checkpoint.name,
            "source_revision": model.source_revision,
        }
    raft = None
    if args.temporal_raft or args.waft_temporal_init:
        raft_weights = (
            Path(args.raft_weights).expanduser().resolve()
            if args.raft_weights
            else Path.home() / "BothEyesDepth/stereoanyvideo/third_party/RAFT/models/raft-things.pth"
        )
        t0 = time.perf_counter()
        raft = load_raft(raft_weights, args.raft_root, args.device)
        t_temporal_flow_model_load += time.perf_counter() - t0
    temporal_state = {} if args.waft_temporal_init else None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开左/SBS 视频: {args.video}")
    cap_right = cv2.VideoCapture(args.video_right) if args.video_right else None
    if cap_right is not None and not cap_right.isOpened():
        raise RuntimeError(f"无法打开右视频: {args.video_right}")
    paired_indices = None
    n_left_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cap_right is not None:
        left_pts_path = Path(args.left_pts) if args.left_pts else pts_sidecar_path(args.video)
        right_pts_path = Path(args.right_pts) if args.right_pts else pts_sidecar_path(args.video_right)
        if not left_pts_path.is_file() or not right_pts_path.is_file():
            raise FileNotFoundError(
                "独立左右视频必须提供 PTS CSV；可用 --left-pts/--right-pts 显式指定"
            )
        left_pts = load_pts_us(left_pts_path)
        right_pts = load_pts_us(right_pts_path)
        n_right_total = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
        if pts_metadata_mismatch(len(left_pts), len(right_pts), n_left_total, n_right_total):
            # Some segmented H.264 files advertise a frame count larger than
            # their decodable frames. Device PTS is the recording's frame index.
            print(
                f"[sync] 容器帧数 {n_left_total}/{n_right_total} 与 PTS "
                f"{len(left_pts)}/{len(right_pts)} 不一致，按 PTS 配对"
            )
        paired_indices, sync_offsets_us = match_left_to_right_pts(
            left_pts, right_pts, round(args.pts_match_tolerance_ms * 1000),
        )
        if not paired_indices:
            raise ValueError("没有 PTS 差值在容差内的双目帧对")
        n_total = len(paired_indices)
        print(
            f"[sync] PTS 配对 {n_total}/{n_left_total} 左帧，"
            f"中位偏差={np.median(sync_offsets_us) / 1000:.3f}ms，"
            f"最大偏差={sync_offsets_us.max() / 1000:.3f}ms"
        )
    else:
        n_total = n_left_total
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps > 0 else float(src_fps)
    end = resolve_processing_end(
        n_total, args.start_frame, args.end_frame, args.max_frames
    )
    print(f"[video] 共 {n_total} 帧，处理 [{args.start_frame}, {end})，输出 fps={fps:.2f}")

    cv2.imwrite(
        str(outdir / "colorbar.png"),
        make_depth_colorbar_log(args.dmin_m, args.dmax_m, height=H),
    )
    print(f"[color] 对数米制色阶 {args.dmin_m}~{args.dmax_m}m（全片固定，colorbar.png 已保存）")

    writer = cv2.VideoWriter(
        str(outdir / args.video_name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    t_all = time.time()
    depth_tbuf = deque(maxlen=max(1, args.temporal_median))
    win = max(1, args.temporal_median)
    prev_gray = None
    prev_depth = None
    prev_left_t = None
    prev_depth_t = None
    split_left_previous_idx = None
    split_right_previous_idx = None
    split_right_cached = None
    flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                       poly_n=5, poly_sigma=1.2, flags=0)
    frame_idx = args.start_frame
    processed = 0
    while frame_idx < end:
        batch_frames = []
        source_frame_indices = []
        t0 = time.perf_counter()
        if paired_indices is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            for source_idx in range(frame_idx, frame_idx + min(args.batch_size, end - frame_idx)):
                ok, img = cap.read()
                if not ok:
                    break
                batch_frames.append(img)
                source_frame_indices.append(source_idx)
        else:
            requested_pairs = paired_indices[
                frame_idx: frame_idx + min(args.batch_size, end - frame_idx)
            ]
            for left_idx, right_idx in requested_pairs:
                left_reads = forward_decode_read_count(split_left_previous_idx, left_idx)
                if left_reads is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, left_idx)
                    left_reads = 1
                ok_left = False
                for _ in range(left_reads):
                    ok_left, left_img = cap.read()
                    if not ok_left:
                        break
                split_left_previous_idx = left_idx

                right_reads = forward_decode_read_count(split_right_previous_idx, right_idx)
                if right_reads is None:
                    cap_right.set(cv2.CAP_PROP_POS_FRAMES, right_idx)
                    right_reads = 1
                ok_right = True
                if right_reads:
                    for _ in range(right_reads):
                        ok_right, right_img = cap_right.read()
                        if not ok_right:
                            break
                    split_right_cached = right_img if ok_right else None
                else:
                    right_img = split_right_cached
                    ok_right = right_img is not None
                split_right_previous_idx = right_idx
                if not ok_left or not ok_right:
                    break
                batch_frames.append((left_img, right_img))
                source_frame_indices.append(left_idx)
            if len(batch_frames) != len(requested_pairs):
                raise RuntimeError(
                    "PTS 配对帧无法完整解码："
                    f"请求 {len(requested_pairs)} 帧，仅读取 {len(batch_frames)} 帧"
                )
        t_decode += time.perf_counter() - t0
        if not batch_frames:
            break
        B = len(batch_frames)

        bgr_pairs = []
        t0 = time.perf_counter()
        for img in batch_frames:
            if paired_indices is None:
                l_bgr, r_bgr = img[:, : img.shape[1] // 2], img[:, img.shape[1] // 2 :]
            else:
                l_bgr, r_bgr = img
            bgr_pairs.append(calib.rectify_pair(l_bgr, r_bgr, rect))
        t_rectify += time.perf_counter() - t0
        dep_b, valid_b, _rgb_b, timing = process_batch(
            model, bgr_pairs, rect, fx, baseline, args,
            temporal_flow_model=raft if args.waft_temporal_init else None,
            temporal_state=temporal_state,
        )
        t_waft += timing["stereo_forward"]
        t_align += timing["photo_align"]
        t_fusion += timing["center_fusion"]
        t_fill += timing["fill"]
        t_depth_gf += timing["depth_gf"]
        timing_detail = timing.get("stereo_detail")
        if timing_detail is not None:
            stages_seconds = {
                    key: round(float(value), 6)
                    for key, value in timing_detail.items()
                    if isinstance(value, (int, float)) and key != "temporal_valid_ratio"
                }
        else:
            stages_seconds = {
                "stereo_forward_seconds": round(float(timing["stereo_forward"]), 6)
            }
        timing_record = {
            "batch_index": len(waft_timing_records),
            "start_frame": source_frame_indices[0],
            "end_frame": source_frame_indices[-1] + 1,
            "batch_size": B,
            "model_samples": 2 * B if args.bi else B,
            "stages_seconds": stages_seconds,
        }
        if timing_detail is not None and "temporal_valid_ratio" in timing_detail:
            timing_record["temporal_valid_ratio"] = round(
                float(timing_detail["temporal_valid_ratio"]), 6
            )
        waft_timing_records.append(timing_record)

        for b in range(B):
            rL_bgr, rR_bgr = bgr_pairs[b]
            d_cur = dep_b[b].unsqueeze(0)  # (1,1,H,W) GPU
            v_cur = valid_b[b].unsqueeze(0)
            if win > 1:
                depth_tbuf.append(d_cur)
                if len(depth_tbuf) == win:
                    d_cur = torch.median(torch.stack(list(depth_tbuf)), dim=0).values
            if args.temporal_raft and raft is not None:
                # RAFT 光流 + 光度/深度一致性门控的时间 EMA（防闪不拖影）
                cur_left_t = (
                    torch.from_numpy(cv2.cvtColor(rL_bgr, cv2.COLOR_BGR2RGB))
                    .permute(2, 0, 1).float().unsqueeze(0).to(args.device)
                )
                if prev_left_t is not None and prev_depth_t is not None:
                    flow = flow_between(raft, prev_left_t, cur_left_t)
                    prev_w = warp_with_flow(prev_depth_t, flow)
                    prev_w_gray = warp_with_flow(
                        prev_left_t.mean(dim=1, keepdim=True), flow
                    )
                    cur_gray_t = cur_left_t.mean(dim=1, keepdim=True)
                    mae = (prev_w_gray - cur_gray_t).abs()
                    g_photo = torch.clamp(
                        1.0 - mae / max(args.temporal_photo_tol, 1e-3), 0.0, 1.0
                    )
                    d_diff = (prev_w - d_cur).abs()
                    g_depth = (
                        d_diff
                        <= torch.maximum(
                            torch.tensor(args.temporal_depth_tol, device=args.device),
                            args.temporal_depth_rel * d_cur,
                        )
                    ).float()
                    g = g_photo * g_depth
                    g = F.avg_pool2d(g, 7, stride=1, padding=3)
                    if os.environ.get("SAV_DEBUG_GATE"):
                        print(
                            f"[gate] frame={source_frame_indices[b]} g_photo={g_photo.mean().item():.3f} "
                            f"g_depth={g_depth.mean().item():.3f} g={g.mean().item():.3f}",
                            flush=True,
                        )
                    d_cur = g * (
                        args.temporal_alpha * prev_w
                        + (1.0 - args.temporal_alpha) * d_cur
                    ) + (1.0 - g) * d_cur
                prev_left_t = cur_left_t
                prev_depth_t = d_cur.detach()
            dep_np = d_cur[0, 0].cpu().numpy()
            valid_np = v_cur[0, 0].cpu().numpy().astype(bool)
            if args.left_hole_fill:
                t0 = time.perf_counter()
                dep_np, valid_np, fill_stats = fill_small_left_holes(
                    dep_np, valid_np, rL_bgr,
                    max_area=args.left_hole_fill_max_area,
                    color_tol=args.left_hole_fill_color_tol,
                )
                t_left_hole_fill += time.perf_counter() - t0
                left_hole_fill_components += fill_stats["filled_components"]
                left_hole_fill_pixels += fill_stats["filled_pixels"]
            if args.spatial_median > 1:
                dep_np = cv2.medianBlur(dep_np, args.spatial_median)
            # 时间 EMA：默认用光流把上一帧深度 warp 到当前帧再融合（运动补偿）
            if args.temporal_ema != 1.0:
                cur_gray = cv2.cvtColor(rL_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if prev_depth is not None and prev_gray is not None:
                    if args.temporal_ema > 0.0:
                        alpha = args.temporal_ema
                        warped = prev_depth
                    else:
                        alpha = args.flow_alpha
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, cur_gray, None, **flow_params
                        )
                        yy, xx = np.mgrid[0 : dep_np.shape[0], 0 : dep_np.shape[1]]
                        # Farneback(prev,cur) 的 flow 把 cur 坐标映射回 prev，
                        # 因此 warp prev 深度到 cur 坐标要加 flow
                        mapx = (xx + flow[..., 0]).astype(np.float32)
                        mapy = (yy + flow[..., 1]).astype(np.float32)
                        warped = cv2.remap(
                            prev_depth, mapx, mapy,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                        )
                    smoothed = alpha * dep_np + (1.0 - alpha) * warped
                    dep_np = np.where(valid_np, smoothed, dep_np)
                prev_gray = cur_gray
                prev_depth = dep_np
            t0 = time.perf_counter()
            depth_img = colorize_depth_log(dep_np, valid_np, args.dmin_m, args.dmax_m)
            t_color += time.perf_counter() - t0
            t0 = time.perf_counter()
            writer.write(depth_img)
            if args.save_frames_every > 0 and processed % args.save_frames_every == 0:
                t_png0 = time.perf_counter()
                cv2.imwrite(str(outdir / f"frame_{source_frame_indices[b]:05d}.png"), depth_img)
                t_png_write += time.perf_counter() - t_png0
            if args.save_depth_npy:
                np.save(
                    str(outdir / f"depth_{frame_idx + b:06d}.npy"),
                    dep_np.astype(np.float32),
                )
            t_write += time.perf_counter() - t0
            processed += 1

        frame_idx += B
        if processed % (args.batch_size * 10) == 0 or frame_idx >= end:
            rate = processed / max(time.time() - t_all, 1e-6)
            eta = (end - frame_idx) / rate if rate > 0 else 0
            print(
                f"[progress] {processed}/{end - args.start_frame} 帧，"
                f"吞吐 {rate:.2f} 帧/s，ETA {eta/60:.1f} min",
                flush=True,
            )

    writer.release()
    cap.release()
    if cap_right is not None:
        cap_right.release()
    total_s = time.time() - t_all
    e2e_s = time.perf_counter() - t_program
    waft_stage_seconds = {}
    for record in waft_timing_records:
        for key, value in record["stages_seconds"].items():
            waft_stage_seconds[key] = waft_stage_seconds.get(key, 0.0) + value
    stereo_total_s = waft_stage_seconds.get(
        "stereo_total_seconds",
        waft_stage_seconds.get(
            "waft_total_seconds", waft_stage_seconds.get("stereo_forward_seconds", 0.0)
        ),
    )
    stereo_model_s = waft_stage_seconds.get(
        "model_forward_seconds",
        waft_stage_seconds.get("stereo_forward_seconds", 0.0),
    )
    proc_s = (
        stereo_total_s + t_align + t_fusion + t_fill + t_left_hole_fill
        + t_depth_gf + t_color + t_write
    )
    temporal_valid_ratio = weighted_temporal_valid_ratio(waft_timing_records)
    peak_gpu_memory_gib = gpu_peak_memory_gib(args.device, gpu_memory_tracking)
    peak_gpu_memory_source = (
        "torch.cuda.max_memory_reserved" if gpu_memory_tracking else None
    )
    timing_filename = timing_artifact_name(backend)
    stereo_timing = {
        "video": str(args.video),
        "backend": backend,
        "model_type": args.model_type,
        "bidirectional": bool(args.bi),
        "output_view": args.output_view,
        "batch_size": args.batch_size,
        "iters": model_iters,
        "peak_gpu_memory_gib": peak_gpu_memory_gib,
        "peak_gpu_memory_source": peak_gpu_memory_source,
        "stereonet": stereonet_metadata,
        "bm_parameters": opencv_bm_parameters(args) if backend == "opencv_bm" else None,
        "sgbm_parameters": opencv_sgbm_parameters(args) if backend == "opencv_sgbm" else None,
        "left_hole_fill": {
            **left_hole_fill_parameters(args),
            "seconds": round(t_left_hole_fill, 6),
            "filled_components": left_hole_fill_components,
            "filled_pixels": left_hole_fill_pixels,
        },
        "n_frames": processed,
        "n_batches": len(waft_timing_records),
        "model_samples": sum(record["model_samples"] for record in waft_timing_records),
        "waft_temporal_init": bool(args.waft_temporal_init),
        "temporal_valid_ratio": (
            round(temporal_valid_ratio, 6) if temporal_valid_ratio is not None else None
        ),
        "temporal_configuration": {
            "flow_iters": args.waft_temporal_flow_iters,
            "blend": args.waft_temporal_blend,
            "photo_tol": args.waft_temporal_photo_tol,
            "flow_abs_tol": args.waft_temporal_flow_abs_tol,
            "flow_rel_tol": args.waft_temporal_flow_rel_tol,
            "disp_abs_tol": args.waft_temporal_disp_abs_tol,
            "disp_rel_tol": args.waft_temporal_disp_rel_tol,
        } if args.waft_temporal_init else None,
        "stage_seconds": {key: round(value, 6) for key, value in waft_stage_seconds.items()},
        "average_seconds_per_batch": {
            key: round(value / max(len(waft_timing_records), 1), 6)
            for key, value in waft_stage_seconds.items()
        },
        "average_seconds_per_frame": {
            key: round(value / max(processed, 1), 6)
            for key, value in waft_stage_seconds.items()
        },
        "model_forward_fps": round(
            (sum(record["model_samples"] for record in waft_timing_records) / stereo_model_s)
            if stereo_model_s > 0 else 0.0, 3
        ),
        "batch_records": waft_timing_records,
    }
    (outdir / timing_filename).write_text(
        json.dumps(stereo_timing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats = {
        "video": str(args.video),
        "scale": args.scale,
        "batch_size": args.batch_size,
        "stereo_backend": backend,
        "model_type": args.model_type,
        "iters": model_iters,
        "peak_gpu_memory_gib": peak_gpu_memory_gib,
        "peak_gpu_memory_source": peak_gpu_memory_source,
        "stereonet": stereonet_metadata,
        "bidirectional": bool(args.bi),
        "bm_parameters": opencv_bm_parameters(args) if backend == "opencv_bm" else None,
        "sgbm_parameters": opencv_sgbm_parameters(args) if backend == "opencv_sgbm" else None,
        "left_hole_fill": {
            **left_hole_fill_parameters(args),
            "filled_components": left_hole_fill_components,
            "filled_pixels": left_hole_fill_pixels,
        },
        "max_disp": args.max_disp if backend in ("las2", "ffs") else None,
        "start_frame": args.start_frame,
        "end_frame": frame_idx,
        "n_frames": processed,
        "fps": fps,
        "size": [W, H],
        "total_seconds": round(total_s, 2),
        "end_to_end_seconds": round(e2e_s, 2),
        "avg_seconds_per_frame": round(total_s / max(processed, 1), 4),
        "color_tol": args.color_tol,
        "median_k": args.median_k,
        "guided_filter": bool(args.guided_filter),
        "gf_radius": args.gf_radius,
        "gf_eps": args.gf_eps,
        "depth_gf": bool(args.depth_gf),
        "depth_gf_radius": args.depth_gf_radius,
        "depth_gf_eps": args.depth_gf_eps,
        "depth_unsharp": args.depth_unsharp,
        "temporal_raft": bool(args.temporal_raft),
        "temporal_alpha": args.temporal_alpha,
        "waft_temporal_init": bool(args.waft_temporal_init),
        "waft_temporal_flow_iters": args.waft_temporal_flow_iters,
        "waft_temporal_blend": args.waft_temporal_blend,
        "waft_temporal_photo_tol": args.waft_temporal_photo_tol,
        "waft_temporal_flow_abs_tol": args.waft_temporal_flow_abs_tol,
        "waft_temporal_flow_rel_tol": args.waft_temporal_flow_rel_tol,
        "waft_temporal_disp_abs_tol": args.waft_temporal_disp_abs_tol,
        "waft_temporal_disp_rel_tol": args.waft_temporal_disp_rel_tol,
        "temporal_valid_ratio": (
            round(temporal_valid_ratio, 6) if temporal_valid_ratio is not None else None
        ),
        "depth_z": bool(args.depth_z),
        "output_view": args.output_view,
        "left_vis_mode": args.left_vis_mode,
        "colormap": "log_metric",
        "dmin_m": args.dmin_m,
        "dmax_m": args.dmax_m,
        "temporal_median": win,
        "temporal_ema": args.temporal_ema,
        "stage_model_load_seconds": round(t_model_load, 2),
        "stage_temporal_flow_model_load_seconds": round(t_temporal_flow_model_load, 2),
        "stage_temporal_flow_seconds": round(waft_stage_seconds.get("temporal_flow", 0.0), 2),
        "stage_temporal_initialization_seconds": round(
            waft_stage_seconds.get("temporal_initialization", 0.0), 2
        ),
        "stage_video_decode_locate_seconds": round(t_decode, 2),
        "stage_stereo_rectify_seconds": round(t_rectify, 2),
        "stage_waft_forward_seconds": round(t_waft, 2),
        "stage_waft_pipeline_seconds": round(stereo_total_s, 2),
        "stage_stereo_forward_seconds": round(t_waft, 2),
        "stage_stereo_pipeline_seconds": round(stereo_total_s, 2),
        "stereo_timing_file": timing_filename,
        "stage_photometric_align_seconds": round(t_align, 2),
        "stage_center_fusion_seconds": round(t_fusion, 2),
        "stage_disocclusion_fill_seconds": round(t_fill, 2),
        "stage_left_hole_fill_seconds": round(t_left_hole_fill, 2),
        "stage_depth_gf_seconds": round(t_depth_gf, 2),
        "stage_depth_colorize_seconds": round(t_color, 2),
        "stage_video_write_seconds": round(t_write, 2),
        "stage_png_write_seconds": round(t_png_write, 2),
        "stage_stereo_fusion_fill_seconds": round(
            stereo_total_s + t_align + t_fusion + t_fill + t_left_hole_fill + t_depth_gf, 2
        ),
        "stage_main_process_seconds": round(proc_s, 2),
        "stage_other_seconds": round(max(total_s - proc_s, 0), 2),
    }
    (outdir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {processed} 帧，总耗时 {total_s:.1f}s（{total_s/max(processed,1):.2f}s/帧）")
    print(f"[timing] 模型加载 {t_model_load:.2f}s")
    if raft is not None:
        print(f"[timing] RAFT 光流模型加载 {t_temporal_flow_model_load:.2f}s")
    print(f"[timing] 视频解码与定位 {t_decode:.2f}s")
    print(f"[timing] 双目校正 {t_rectify:.2f}s")
    backend_label = backend.upper()
    direction_label = "双向" if args.bi else "单向"
    print(f"[timing] {backend_label} {direction_label}前向 {t_waft:.2f}s")
    print(f"[timing] {backend_label} 输入到输出管线 {stereo_total_s:.2f}s（细分见 {timing_filename}）")
    if args.waft_temporal_init:
        print(f"[timing] 时序双向光流 {waft_stage_seconds.get('temporal_flow', 0.0):.2f}s")
        print(
            f"[timing] WAFT 时序初始化 {waft_stage_seconds.get('temporal_initialization', 0.0):.2f}s，"
            f"有效先验比例 {temporal_valid_ratio or 0.0:.3f}"
        )
    print(f"[timing] 双目光度对齐 {t_align:.2f}s")
    print(f"[timing] 中心视角融合 {t_fusion:.2f}s")
    print(f"[timing] 遮挡补洞 {t_fill:.2f}s")
    print(f"[timing] 左视角小孔洞补全 {t_left_hole_fill:.2f}s")
    print(f"[timing] 深度着色 {t_color:.2f}s")
    print(f"[timing] 视频写盘 {t_write:.2f}s（其中 PNG {t_png_write:.2f}s）")
    print(f"[timing] 脚本总耗时 {total_s:.2f}s")
    print(f"[timing] 端到端耗时 {e2e_s:.2f}s")
    print(f"[out] {outdir / args.video_name}")
    print(f"[out] {outdir / timing_filename}")


if __name__ == "__main__":
    main()
