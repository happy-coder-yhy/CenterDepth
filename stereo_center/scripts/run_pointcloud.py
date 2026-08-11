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

from stereo_center import calib, pointcloud, pytorch3d_rasterizer, s2m2_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="点云 + 中心虚拟相机渲染")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--frame", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--model-type", type=str, default="S", choices=["S", "M", "L", "XL"])
    parser.add_argument("--num-refine", type=int, default=3)
    parser.add_argument(
        "--weights", type=str, default=None,
        help="权重目录（默认：仓库根 weights/pretrain_weights 或 $S2M2_WEIGHTS_DIR）",
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
    parser.add_argument("--radius-px", type=int, default=1, help="点半径（像素）")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 权重解析（与 run_pipeline 一致）
    weights_dir = args.weights
    if not weights_dir:
        weights_dir = os.environ.get("S2M2_WEIGHTS_DIR")
    if not weights_dir:
        for cand in (REPO_ROOT / "weights/pretrain_weights", PROJECT_ROOT / "weights/pretrain_weights"):
            if cand.exists():
                weights_dir = str(cand)
                break
    if not weights_dir:
        raise FileNotFoundError("未找到权重目录，请用 --weights 或设置 S2M2_WEIGHTS_DIR")
    print(f"[weights] 权重目录: {weights_dir}")

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

    # 双向 S²M²
    model = s2m2_inference.load_s2m2(args.model_type, weights_dir, args.num_refine, args.device)
    lt = torch.from_numpy(cv2.cvtColor(rL, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)
    rt = torch.from_numpy(cv2.cvtColor(rR, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)
    dL, _, cL, t1 = s2m2_inference.run_stereo_matching(model, lt, rt, args.device)
    dR, _, cR, t2 = s2m2_inference.run_stereo_matching(model, rt, lt, args.device)
    print(f"[s2m2] 双向推理 {t1 + t2:.2f}s")

    # 深度 -> 点云
    depthL = fx * B / dL.numpy().clip(min=0.5)
    depthR = fx * B / dR.numpy().clip(min=0.5)
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
        "model_type": args.model_type,
        "render_backend": backend,
        "num_points": int(len(pts)),
        "valid_pixel_fraction": round(float(center_valid.mean()), 4),
        "depth_min_m": round(float(center_depth[center_valid].min()), 4),
        "depth_max_m": round(float(center_depth[center_valid].max()), 4),
        "s2m2_inference_seconds": round(t1 + t2, 3),
    }
    (outdir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] 结果已保存到 {outdir}")


def pointcloud_depth_png(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """深度 -> jet 伪彩图（与 run_pipeline 风格一致）。"""
    d = np.where(valid, depth, np.nan)
    if not np.isfinite(d).any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmax = np.nanpercentile(d, 95)
    norm = np.clip(np.nan_to_num(d / max(vmax, 1e-6), nan=0.0), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~valid] = 0
    return img


if __name__ == "__main__":
    main()
