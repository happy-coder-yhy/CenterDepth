"""PyTorch3D PointsRasterizer 渲染中心视角（无 PyTorch3D 时用 numpy z-buffer 兜底）。

与导师方案文档中的路线一致：
    fish-eye 校正 -> S²M² disparity -> metric depth -> 3D 点云
    -> 头部中心虚拟相机 -> PointsRasterizer -> Center RGB + Center Depth

PyTorch3D 可用时使用真正的 PointsRasterizer；不可用时退回纯 numpy 的
z-buffer 渲染（语义一致：每个像素取最近点）。
"""

from __future__ import annotations

import numpy as np

from . import pointcloud


def pytorch3d_available() -> bool:
    try:
        import pytorch3d  # noqa: F401

        return True
    except Exception:
        return False


def render_center_view(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    cam_tx: float,
    radius_px: int = 1,
    device: str = "cpu",
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用虚拟相机（左相机坐标 +X 平移 cam_tx）z-buffer 渲染点云。

    Args:
        points: (N, 3) 世界坐标（左校正相机坐标系）。
        colors: (N, 3) 0~1 RGB。
        K: 3x3 相机内参。
        H/W: 输出图像尺寸。
        cam_tx: 中心虚拟相机相对左相机沿 X 的平移（中点 = baseline/2）。
        radius_px: 点半径（像素），填投影空洞。
        device: pytorch3d 后端使用的设备。
        backend: auto / pytorch3d / fallback。

    Returns:
        rgb: (H, W, 3) float32 0~255；
        depth: (H, W) float32（米），无投影处 0；
        valid: (H, W) bool。
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if backend == "auto":
        backend = "pytorch3d" if pytorch3d_available() else "fallback"

    if backend == "pytorch3d":
        return _render_pytorch3d(points, colors, K, H, W, cam_tx, radius_px, device)
    if backend == "fallback":
        rgb, depth = pointcloud.render_zbuffer(
            points, colors, fx, fy, cx, cy, H, W, cam_tx=cam_tx, point_radius=radius_px
        )
        return rgb, depth, depth > 0
    raise ValueError(f"未知后端: {backend}")


def _render_pytorch3d(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    cam_tx: float,
    radius_px: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from pytorch3d.renderer import (
        PerspectiveCameras,
        PointsRasterizationSettings,
        PointsRasterizer,
    )
    from pytorch3d.structures import Pointclouds

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # PyTorch3D 相机坐标系为 +X 左、+Y 上（与图像坐标相反），故 X/Y 取负；
    # 同时把投影主点移到图像中心，避免 NDC 视锥外（|x|>1）点被裁剪
    shift_x = cx - W / 2.0
    shift_y = cy - H / 2.0
    Z = points[:, 2]
    X_cam = points[:, 0] - cam_tx
    X = -(X_cam + shift_x * Z / fx)
    Y = -(points[:, 1] + shift_y * Z / fy)
    pts_cam = np.stack([X, Y, Z], axis=1)
    pts = torch.from_numpy(pts_cam).float().to(device)
    feat = torch.from_numpy(colors).float().to(device)

    cameras = PerspectiveCameras(
        R=torch.eye(3, device=device)[None],
        T=torch.zeros(1, 3, device=device),
        focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[W / 2.0, H / 2.0]], device=device, dtype=torch.float32),
        image_size=((H, W),),
        in_ndc=False,
        device=device,
    )
    radius_ndc = 2.0 * radius_px / W  # 像素 -> NDC
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=radius_ndc,
        points_per_pixel=3,
        bin_size=None,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(Pointclouds(points=[pts], features=[feat]))

    idx = fragments.idx[0]  # (H, W, points_per_pixel)
    nearest = idx[..., 0]
    valid = nearest >= 0
    rgb = torch.zeros(H, W, 3, device=device)
    dep = torch.zeros(H, W, device=device)
    rgb[valid] = feat[nearest[valid]] * 255.0
    # 米制深度：直接取最近点的世界 Z（zbuf 是 NDC 深度，不可直接用作米制）
    dep[valid] = torch.from_numpy(points[:, 2]).float().to(device)[nearest[valid]]
    return (
        rgb.cpu().numpy().astype(np.float32),
        dep.cpu().numpy().astype(np.float32),
        valid.cpu().numpy(),
    )
