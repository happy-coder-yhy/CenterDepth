"""组合管线：双目帧 -> 鱼眼校正 -> 立体匹配（s2m2|waft）-> SoftSplat -> 中心视角 RGB + Depth。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from . import calib, softsplat, stereo_backend


@dataclass
class PipelineResult:
    rect_left: np.ndarray  # BGR
    rect_right: np.ndarray  # BGR
    disp: np.ndarray  # (H, W) float32 左视差（像素）
    occ: np.ndarray  # (H, W) float32
    conf: np.ndarray  # (H, W) float32
    center_rgb: np.ndarray  # (H, W, 3) uint8
    center_depth: np.ndarray  # (H, W) float32（米），无效处为 0
    center_valid: np.ndarray  # (H, W) bool
    elapsed_s2m2: float


def process_stereo_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    cal: dict,
    model,
    device: str = "cpu",
    scale: float = 0.5,
    backend: str = "waft",
    backend_kwargs: dict | None = None,
) -> PipelineResult:
    """处理一帧双目图，输出中心视角 RGB + Depth。"""
    out_size = (
        max(32, int(cal["resolution"][0] * scale)),
        max(32, int(cal["resolution"][1] * scale)),
    )
    rect = calib.compute_rectification_maps(cal, output_size=out_size)
    rL, rR = calib.rectify_pair(left_bgr, right_bgr, rect)

    left_rgb = cv2.cvtColor(rL, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(rR, cv2.COLOR_BGR2RGB)
    left_t = torch.from_numpy(left_rgb).permute(2, 0, 1).float().unsqueeze(0)
    right_t = torch.from_numpy(right_rgb).permute(2, 0, 1).float().unsqueeze(0)

    mod = stereo_backend.get_backend(backend)
    kwargs = dict(backend_kwargs or {})
    disp, occ, conf, elapsed = mod.run_stereo_matching(
        model, left_t, right_t, device, **kwargs
    )

    center_rgb, center_depth, valid = softsplat.center_view(
        left_t, right_t,
        disp.unsqueeze(0).unsqueeze(0),
        conf.unsqueeze(0).unsqueeze(0),
        occ.unsqueeze(0).unsqueeze(0),
        fx=rect["fx"],
        baseline=rect["baseline"],
    )

    center_rgb_np = (
        center_rgb[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
    )
    center_depth_np = center_depth[0, 0].numpy()
    valid_np = valid[0, 0].numpy().astype(bool)

    return PipelineResult(
        rect_left=rL,
        rect_right=rR,
        disp=disp.numpy(),
        occ=occ.numpy(),
        conf=conf.numpy(),
        center_rgb=cv2.cvtColor(center_rgb_np, cv2.COLOR_RGB2BGR),
        center_depth=center_depth_np,
        center_valid=valid_np,
        elapsed_s2m2=elapsed,
    )
