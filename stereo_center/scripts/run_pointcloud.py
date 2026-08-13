#!/usr/bin/env python
"""点云路线：双向视差 -> 3D 点云 -> 中心虚拟相机 z-buffer 渲染 -> Center RGB + Depth。

用法示例（在 stereo_center/ 目录下）：
    ../.venv/bin/python scripts/run_pointcloud.py \
        --video ../vdego-c2-48b749_2026-07-28_10-27-26_30fps/output.mp4 \
        --calib ../vdego-c2-48b749_2026-07-28_10-27-26_30fps/calibration.json \
        --frame 60 --scale 0.5 --outdir outputs/pointcloud/f60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / "third_party/s2m2/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stereo_center import calib, pointcloud, pytorch3d_rasterizer, stereo_backend  # noqa: E402


def resolve_weights_dir(explicit: str | None, backend: str = "s2m2") -> str:
    """权重目录解析（与 run_pipeline 一致）：--weights > 环境变量 > 仓库根 weights/<backend>/。"""
    if explicit:
        return explicit
    if backend == "waft":
        env = os.environ.get("WAFT_WEIGHTS_DIR")
        subdir = "waft"
    else:
        env = os.environ.get("S2M2_WEIGHTS_DIR")
        subdir = "pretrain_weights"
    if env:
        return env
    for cand in (REPO_ROOT / "weights" / subdir, PROJECT_ROOT / "weights" / subdir):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"未找到权重目录（后端 {backend}），请用 --weights 或设置环境变量")


def main() -> None:
    parser = argparse.ArgumentParser(description="点云 + 中心虚拟相机渲染")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--frame", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--stereo-backend", type=str, default="waft", choices=["s2m2", "waft"],
        help="立体匹配后端（默认 waft；s2m2 可回退）",
    )
    parser.add_argument(
        "--model-type", type=str, default="DAv2L-5",
        choices=["S", "M", "L", "XL", "DAv2S-4", "DAv2B-4", "DAv2L-5"],
        help="模型类型：s2m2 用 S/M/L/XL；waft 用 DAv2S-4/DAv2B-4/DAv2L-5",
    )
    parser.add_argument("--num-refine", type=int, default=3)
    parser.add_argument(
        "--weights", type=str, default=None,
        help="权重目录（默认按后端解析：waft->weights/waft 或 $WAFT_WEIGHTS_DIR）",
    )
    parser.add_argument(
        "--waft-mode", type=str, default="auto", choices=["auto", "direct", "hiera"],
        help="WAFT 推理模式：auto 在 >1080 时用 0.5->1.0 分层",
    )
    parser.add_argument(
        "--disp-left-npy", type=str, default=None,
        help="复用已保存的左视差 npy（跳过立体匹配；与 waft 环境解耦）",
    )
    parser.add_argument(
        "--disp-right-npy", type=str, default=None,
        help="复用已保存的右视差 npy（跳过立体匹配；与 waft 环境解耦）",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "outputs/pointcloud"))
    parser.add_argument("--max-points", type=int, default=500_000, help="点云下采样上限")
    parser.add_argument("--stride", type=int, default=2, help="深度图采样步长（>1 减少点数）")
    parser.add_argument("--z-max", type=float, default=10.0, help="可视化深度截断（米）")
    parser.add_argument(
        "--backend", type=str, default="auto", choices=["auto", "pytorch3d", "fallback"],
        help="渲染后端：pytorch3d / fallback（纯 numpy z-buffer）",
    )
    parser.add_argument("--radius-px", type=int, default=2, help="点半径（像素）")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 读帧
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"读取视频第 {args.frame} 帧失败")
    left_bgr, right_bgr = img[:, : img.shape[1] // 2], img[:, img.shape[1] // 2 :]

    # 标定 + 校正
    cal = calib.load_vdego_calibration(args.calib)
    rect = calib.compute_rectification_maps(
        cal, output_size=(int(cal["resolution"][0] * args.scale), int(cal["resolution"][1] * args.scale))
    )
    rL, rR = calib.rectify_pair(left_bgr, right_bgr, rect)
    H, W = rL.shape[:2]
    fx, fy = rect["P1"][0, 0], rect["P1"][1, 1]
    cx, cy = rect["P1"][0, 2], rect["P1"][1, 2]
    B = rect["baseline"]
    print(f"[rect] 校正后 {W}x{H}, fx={fx:.1f}, baseline={B:.4f} m")

    # 双向视差：优先复用 npy（跨环境解耦），否则跑立体匹配后端
    stereo_name = args.stereo_backend
    if args.disp_left_npy and args.disp_right_npy:
        dL = np.load(args.disp_left_npy)
        dR = np.load(args.disp_right_npy)
        if dL.shape != (H, W) or dR.shape != (H, W):
            raise ValueError(
                f"npy 视差形状 {dL.shape}/{dR.shape} 与校正尺寸 {H}x{W} 不一致"
            )
        t1 = t2 = 0.0
        print(f"[stereo] 复用视差 npy: {args.disp_left_npy}, {args.disp_right_npy}")
    else:
        if stereo_name == "s2m2" and args.model_type not in ("S", "M", "L", "XL"):
            raise ValueError(f"s2m2 后端不支持模型类型 {args.model_type}")
        if stereo_name == "waft" and args.model_type not in ("DAv2S-4", "DAv2B-4", "DAv2L-5"):
            raise ValueError(
                f"waft 后端不支持模型类型 {args.model_type}（可选 DAv2S-4/DAv2B-4/DAv2L-5）"
            )
        weights_dir = resolve_weights_dir(args.weights, stereo_name)
        print(f"[weights] 权重目录: {weights_dir}")
        model = stereo_backend.load(
            stereo_name, args.model_type, weights_dir, args.device,
            num_refine=args.num_refine,
        )
        lt = torch.from_numpy(cv2.cvtColor(rL, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)
        rt = torch.from_numpy(cv2.cvtColor(rR, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)
        # 点云路线只需要双向视差：waft 用 visibility 模式避免每个方向重复双向推理
        backend_kwargs = (
            {"hiera": args.waft_mode, "conf_mode": "ones", "occ_mode": "visibility"}
            if stereo_name == "waft"
            else {}
        )
        dL, _, _, t1 = stereo_backend.run(
            stereo_name, model, lt, rt, args.device, **backend_kwargs
        )
        dR, _, _, t2 = stereo_backend.run(
            stereo_name, model, rt, lt, args.device, **backend_kwargs
        )
        print(f"[{stereo_name}] 双向推理 {t1 + t2:.2f}s")
    dL = np.asarray(dL)
    dR = np.asarray(dR)
    np.save(str(outdir / "disp_left.npy"), dL)
    np.save(str(outdir / "disp_right.npy"), dR)

    # 深度 -> 点云
    depthL = fx * B / dL.clip(min=0.5)
    depthR = fx * B / dR.clip(min=0.5)
    ptsL, colL = pointcloud.depth_to_pointcloud(
        rL, depthL, fx, fy, cx, cy, max_points=args.max_points, stride=args.stride
    )
    ptsR, colR = pointcloud.depth_to_pointcloud(
        rR, depthR, fx, fy, cx, cy, max_points=args.max_points, stride=args.stride
    )
    ptsR = pointcloud.transform_right_to_left(ptsR, B)
    pts = np.concatenate([ptsL, ptsR], axis=0)
    col = np.concatenate([colL, colR], axis=0)
    print(f"[cloud] 点数 {len(pts)}（左 {len(ptsL)} + 右 {len(ptsR)}）")

    # 中心虚拟相机（中点，cam_tx = B/2）z-buffer 渲染
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    backend = args.backend
    if backend == "auto":
        backend = "pytorch3d" if pytorch3d_rasterizer.pytorch3d_available() else "fallback"
    print(f"[render] 后端: {backend}")
    center_rgb, center_depth, center_valid = pytorch3d_rasterizer.render_center_view(
        pts, col, K, H, W, cam_tx=B / 2.0,
        radius_px=args.radius_px, device=args.device, backend=backend,
    )
    print(
        f"[center] 有效像素 {center_valid.mean():.1%}，深度 "
        f"{center_depth[center_valid].min():.2f}~{center_depth[center_valid].max():.2f} m"
    )

    # 保存
    center_rgb_u8 = np.clip(center_rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(outdir / "center_rgb.png"), center_rgb_u8)
    np.save(str(outdir / "center_depth.npy"), center_depth)
    np.savez(str(outdir / "pointcloud.npz"), points=pts, colors=col)
    pointcloud.save_ply(pts, col, outdir / "pointcloud.ply")
    pointcloud.visualize_pointcloud(pts, col, outdir / "pointcloud_3d.png", z_max=args.z_max)
    cv2.imwrite(str(outdir / "center_depth.png"), pointcloud_depth_png(center_depth, center_valid))

    stats = {
        "frame": args.frame,
        "scale": args.scale,
        "stereo_backend": stereo_name if not args.disp_left_npy else "npy",
        "model_type": args.model_type,
        "render_backend": backend,
        "num_points": int(len(pts)),
        "valid_pixel_fraction": round(float(center_valid.mean()), 4),
        "depth_min_m": round(float(center_depth[center_valid].min()), 4),
        "depth_max_m": round(float(center_depth[center_valid].max()), 4),
        "s2m2_inference_seconds": round(t1 + t2, 3),
    }
    if stereo_name == "waft" and not args.disp_left_npy:
        stats["waft_mode"] = args.waft_mode
    (outdir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] 结果已保存到 {outdir}")


def pointcloud_depth_png(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """深度 -> jet 伪彩图（与 run_pipeline 风格一致：p98 + gamma 0.6）。"""
    d = np.where(valid, depth, np.nan)
    if not np.isfinite(d).any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmax = np.nanpercentile(d, 98)
    norm = np.clip(np.nan_to_num(d / max(vmax, 1e-6), nan=0.0), 0, 1)
    norm = norm**0.6
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~valid] = 0
    return img


if __name__ == "__main__":
    main()
