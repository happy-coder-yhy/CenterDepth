"""3D 点云重建与 z-buffer 虚拟相机渲染（纯 numpy，无额外依赖）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def depth_to_pointcloud(
    rgb_bgr: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_points: int = 300_000,
    stride: int = 1,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """RGB-D -> 相机坐标系 3D 点云。

    Args:
        rgb_bgr: (H, W, 3) uint8 BGR 图。
        depth: (H, W) float32 深度（米）。
        fx/fy/cx/cy: 相机内参（与图像尺寸对应）。
        max_points: 随机下采样上限。
        stride: 步长采样（>1 可快速降采样）。

    Returns:
        points: (N, 3) float32，X 向右、Y 向下、Z 向前（相机坐标系）。
        colors: (N, 3) float32，0~1 RGB。
    """
    H, W = depth.shape
    valid = (depth > 0) & np.isfinite(depth)
    if stride > 1:
        yy, xx = np.mgrid[0:H, 0:W]
        valid &= (yy % stride == 0) & (xx % stride == 0)
    v, u = np.nonzero(valid)
    Z = depth[v, u].astype(np.float64)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    colors = rgb_bgr[v, u][:, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    if len(X) > max_points:
        idx = np.random.default_rng(rng_seed).choice(len(X), max_points, replace=False)
        X, Y, Z, colors = X[idx], Y[idx], Z[idx], colors[idx]
    points = np.stack([X, Y, Z], axis=1).astype(np.float32)
    return points, colors


def transform_right_to_left(points_right: np.ndarray, baseline: float) -> np.ndarray:
    """平行校正假设：右相机坐标 -> 左相机坐标（沿 X 平移 +B）。"""
    p = points_right.copy()
    p[:, 0] += baseline
    return p


def render_zbuffer(
    points: np.ndarray,
    colors: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    H: int,
    W: int,
    cam_tx: float = 0.0,
    point_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """把点云投影到虚拟相机并 z-buffer 渲染。

    Args:
        points: (N, 3) 相机坐标系（左相机）。
        colors: (N, 3) 0~1 RGB。
        cam_tx: 虚拟相机相对左相机沿 X 的平移（中点相机 = baseline/2）。
        point_radius: 每个点填充的邻域半径（像素），用于填补投影空洞。

    Returns:
        rgb: (H, W, 3) float32 0~255；
        depth: (H, W) float32，无投影处为 0。
    """
    X = points[:, 0] - cam_tx
    Z = points[:, 2]
    ok = Z > 0.05
    u = fx * X[ok] / Z[ok] + cx
    v = fy * points[ok, 1] / Z[ok] + cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, Zv = ui[m], vi[m], Z[ok][m]
    col = colors[ok][m]

    # 点半径 splat：每个点写 (2r+1)^2 邻域，带 z-buffer
    if point_radius > 0:
        r = point_radius
        offs = np.arange(-r, r + 1)
        du, dv = np.meshgrid(offs, offs)
        n_off = du.size
        ui = (ui[:, None] + du.ravel()[None, :]).ravel()
        vi = (vi[:, None] + dv.ravel()[None, :]).ravel()
        Zv = np.repeat(Zv, n_off)
        col = np.repeat(col, n_off, axis=0)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi, Zv, col = ui[inb], vi[inb], Zv[inb], col[inb]

    order = np.argsort(Zv)[::-1]  # 远 -> 近，近点后写入覆盖远点
    rgb = np.zeros((H, W, 3), np.float32)
    depth = np.zeros((H, W), np.float32)
    rgb[vi[order], ui[order]] = col[order] * 255.0
    depth[vi[order], ui[order]] = Zv[order]
    return rgb, depth


def save_ply(points: np.ndarray, colors: np.ndarray, path: str | Path) -> None:
    """保存 PLY（ASCII，含 RGB），可用 MeshLab / CloudCompare 打开。"""
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def visualize_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    out_path: str | Path,
    z_max: float = 10.0,
    title: str = "3D Point Cloud",
) -> None:
    """用 matplotlib 渲染 3D 点云（三个视角并排），保存 PNG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = points[:, 2] <= z_max
    pts = points[keep]
    col = colors[keep]

    views = [
        ("front", 0, -90),
        ("perspective", 20, -60),
        ("top", 90, 0),
    ]
    fig = plt.figure(figsize=(18, 6))
    for i, (name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=0.6, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(name)
        # 等比例
        lims = np.percentile(pts, [1, 99], axis=0)
        span = max((lims[1] - lims[0]).max() / 2, 0.1)
        centers = (lims[0] + lims[1]) / 2
        for j, c in enumerate(centers):
            getattr(ax, "set_xlim" if j == 0 else "set_ylim" if j == 1 else "set_zlim")(
                c - span, c + span
            )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
