#!/usr/bin/env python
"""客观评估中心视角合成质量（无 GT 时使用）。

指标：
- reproj_ssim_{left,right}: center_rgb 反投影回左/右相机后与 rect_left/right 的 SSIM
- reproj_mae_{left,right}: 同位置 MAE
- hole_fraction: 中心视图无效像素占比

用法（在 stereo_center/ 目录下）：
    ../.venv/bin/python scripts/eval_center_quality.py --outdir outputs/fusion_local/improved
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def photometric_align_right(
    left_bgr: np.ndarray, right_bgr: np.ndarray
) -> np.ndarray:
    """与 pipeline 相同的光度对齐（把右图对齐到左图），保证对比公平。"""
    gL = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gR = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = (gL > 30) & (gR > 30)
    out = right_bgr.copy()
    if not mask.any():
        return out
    for c in range(3):
        lc = left_bgr[:, :, c].astype(np.float32)
        rc = right_bgr[:, :, c].astype(np.float32)
        a = lc[mask].std() / max(rc[mask].std(), 1e-6)
        b = lc[mask].mean() - a * rc[mask].mean()
        out[:, :, c] = np.clip(a * rc + b, 0, 255)
    return out


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """灰度 SSIM（11x11 高斯窗，与 skimage 默认接近）。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    win = cv2.getGaussianKernel(11, 1.5)
    win = win @ win.T
    mu_a = cv2.filter2D(a, -1, win, borderType=cv2.BORDER_REFLECT)
    mu_b = cv2.filter2D(b, -1, win, borderType=cv2.BORDER_REFLECT)
    mu_aa = cv2.filter2D(a * a, -1, win, borderType=cv2.BORDER_REFLECT)
    mu_bb = cv2.filter2D(b * b, -1, win, borderType=cv2.BORDER_REFLECT)
    mu_ab = cv2.filter2D(a * b, -1, win, borderType=cv2.BORDER_REFLECT)
    var_a = mu_aa - mu_a * mu_a
    var_b = mu_bb - mu_b * mu_b
    cov = mu_ab - mu_a * mu_b
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2) + 1e-12
    return float(num.mean() / den.mean())


def reproject_consistency(
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    center_rgb: np.ndarray,
    center_depth: np.ndarray,
    fx: float,
    baseline: float,
) -> dict:
    """中心 RGB 按中心深度反投影到左/右相机，与输入校正图比较。"""
    H, W = center_rgb.shape[:2]
    disp = fx * baseline / np.where(center_depth > 1e-4, center_depth, np.inf)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # 只评估校正图内有真实内容的区域（排除鱼眼圆外的黑色死区）
    content = (
        (cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY) > 5)
        & (cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY) > 5)
    )
    valid = content & np.isfinite(disp) & (disp > 1e-4)

    def sample(src: np.ndarray, disp_map: np.ndarray) -> np.ndarray:
        map_x = np.clip(xx + disp_map, 0, W - 1).astype(np.float32)
        map_y = np.clip(yy, 0, H - 1).astype(np.float32)
        return cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    syn_l = sample(rect_left, disp / 2.0)
    syn_r = sample(rect_right, -disp / 2.0)
    res = {}
    for name, syn, ref in (
        ("left", syn_l, rect_left),
        ("right", syn_r, rect_right),
    ):
        m = valid
        if not m.any():
            res[f"reproj_ssim_{name}"] = 0.0
            res[f"reproj_mae_{name}"] = float("nan")
            continue
        gs = cv2.cvtColor(syn, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        res[f"reproj_ssim_{name}"] = round(ssim(gs[m], gr[m]), 4)
        mae = np.abs(syn.astype(np.float32) - ref.astype(np.float32))
        res[f"reproj_mae_{name}"] = round(float(mae[m].mean()), 3)
    res["hole_fraction"] = round(float((~valid).mean()), 5)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--baseline", type=float, default=None)
    args = parser.parse_args()

    out = Path(args.outdir)
    rL = cv2.imread(str(out / "rect_left.png"))
    rR = cv2.imread(str(out / "rect_right.png"))
    rR_aligned = photometric_align_right(rL, rR)
    rgb = cv2.imread(str(out / "center_rgb.png"))
    dep = np.load(str(out / "center_depth.npy"))
    stats = {}
    sp = out / "stats.json"
    if sp.exists():
        stats = json.loads(sp.read_text())
    fx = args.fx or stats.get("fx")
    baseline = args.baseline or stats.get("baseline")
    if fx is None or baseline is None:
        raise SystemExit("需要 --fx/--baseline 或在 stats.json 中提供 fx/baseline")
    res = reproject_consistency(
        rL, rR_aligned, rgb, dep, float(fx), float(baseline)
    )
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
