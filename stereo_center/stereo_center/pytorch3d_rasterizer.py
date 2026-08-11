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

    pts = torch.from_numpy(points).float().to(device)
    col = torch.from_numpy(colors).float().to(device)
    K_t = torch.from_numpy(K).float().to(device)[None]
    R = torch.eye(3, device=device)[None]
    T = torch.tensor([[-cam_tx, 0.0, 0.0]], device=device)  # world -> center cam

    cameras = PerspectiveCameras(
        R=R, T=T, K=K_t, image_size=((H, W),), in_ndc=False, device=device
    )
    # radius 为 NDC 单位；1 像素 ≈ 2/W，乘系数略放大以覆盖亚像素间隙
    radius_ndc = (radius_px + 0.5) / (W / 2.0)
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=radius_ndc,
        points_per_pixel=3,
        bin_size=None,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(Pointclouds(points=[pts], features=[col]))

    idx = fragments.idx[0]  # (H, W, points_per_pixel)
    nearest = idx[..., 0]
    valid = nearest >= 0
    rgb = torch.zeros(H, W, 3, device=device)
    dep = torch.zeros(H, W, device=device)
    rgb[valid] = col[nearest[valid]] * 255.0
    dep[valid] = pts[nearest[valid], 2]  # 世界 Z == 中心相机 Z（仅沿 X 平移）
    return (
        rgb.cpu().numpy().astype(np.float32),
        dep.cpu().numpy().astype(np.float32),
        valid.cpu().numpy(),
    )
