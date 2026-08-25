"""OpenCV StereoSGBM backend for calibrated left-reference stereo matching."""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np
import torch


SGBM_MODES = {
    "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
    "hh": cv2.STEREO_SGBM_MODE_HH,
    "3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    "hh4": cv2.STEREO_SGBM_MODE_HH4,
}


@dataclass(frozen=True)
class OpenCVSGBMModel:
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    p1: int = 600
    p2: int = 2400
    disp12_max_diff: int = 1
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY


def _validate(model: OpenCVSGBMModel) -> None:
    if model.num_disparities <= 0 or model.num_disparities % 16:
        raise ValueError("sgbm_num_disparities must be a positive multiple of 16")
    if model.block_size < 1 or model.block_size > 255 or model.block_size % 2 == 0:
        raise ValueError("sgbm_block_size must be an odd integer from 1 to 255")
    if model.p1 < 0 or model.p2 <= model.p1:
        raise ValueError("sgbm_p2 must be greater than p1, and p1 must be non-negative")
    if model.disp12_max_diff < -1:
        raise ValueError("sgbm_disp12_max_diff must be at least -1")
    if model.uniqueness_ratio < 0:
        raise ValueError("sgbm_uniqueness_ratio must be non-negative")
    if model.speckle_window_size < 0 or model.speckle_range < 0:
        raise ValueError("SGBM speckle parameters must be non-negative")
    if model.mode not in SGBM_MODES.values():
        raise ValueError("sgbm_mode must be one of: sgbm, hh, 3way, hh4")


def load_opencv_sgbm(
    _model_type: str = "StereoSGBM",
    _weights_dir: str = "",
    _device: str = "cpu",
    **kwargs,
) -> OpenCVSGBMModel:
    """Create a parameter-only StereoSGBM model; no checkpoint or GPU is used."""
    block_size = int(kwargs.get("sgbm_block_size", 5))
    p1 = kwargs.get("sgbm_p1")
    p2 = kwargs.get("sgbm_p2")
    mode_name = str(kwargs.get("sgbm_mode", "3way"))
    if mode_name not in SGBM_MODES:
        raise ValueError("sgbm_mode must be one of: sgbm, hh, 3way, hh4")
    model = OpenCVSGBMModel(
        min_disparity=int(kwargs.get("sgbm_min_disparity", 0)),
        num_disparities=int(kwargs.get("sgbm_num_disparities", 128)),
        block_size=block_size,
        p1=int(p1) if p1 is not None else 8 * 3 * block_size * block_size,
        p2=int(p2) if p2 is not None else 32 * 3 * block_size * block_size,
        disp12_max_diff=int(kwargs.get("sgbm_disp12_max_diff", 1)),
        uniqueness_ratio=int(kwargs.get("sgbm_uniqueness_ratio", 10)),
        speckle_window_size=int(kwargs.get("sgbm_speckle_window_size", 100)),
        speckle_range=int(kwargs.get("sgbm_speckle_range", 2)),
        mode=SGBM_MODES[mode_name],
    )
    _validate(model)
    return model


def _gray_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().float().permute(1, 2, 0).numpy()
    rgb = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _matcher(model: OpenCVSGBMModel) -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(
        minDisparity=model.min_disparity,
        numDisparities=model.num_disparities,
        blockSize=model.block_size,
        P1=model.p1,
        P2=model.p2,
        disp12MaxDiff=model.disp12_max_diff,
        uniquenessRatio=model.uniqueness_ratio,
        speckleWindowSize=model.speckle_window_size,
        speckleRange=model.speckle_range,
        mode=model.mode,
    )


def _visibility(disp: torch.Tensor) -> torch.Tensor:
    _, _, width = disp.shape
    x = torch.arange(width, device=disp.device).view(1, 1, width)
    return ((disp >= 0.5) & (x - disp >= 0) & (disp < width - 1)).float()


def run_stereo_matching(
    model: OpenCVSGBMModel,
    left: torch.Tensor,
    right: torch.Tensor,
    _device: str = "cpu",
    **_unused,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Run CPU StereoSGBM for each item of a ``(B,3,H,W)`` RGB batch."""
    if left.ndim != 4 or right.ndim != 4 or left.shape != right.shape:
        raise ValueError("StereoSGBM expects equal left/right tensors shaped (B, 3, H, W)")
    if left.shape[1] != 3:
        raise ValueError("StereoSGBM expects RGB tensors with three channels")

    t0 = time.perf_counter()
    disparities = []
    for left_image, right_image in zip(left, right):
        raw = _matcher(model).compute(_gray_uint8(left_image), _gray_uint8(right_image))
        disparities.append(torch.from_numpy(raw.astype(np.float32) / 16.0))
    elapsed = time.perf_counter() - t0
    disp = torch.stack(disparities, dim=0)
    occ = _visibility(disp)
    conf = torch.ones_like(disp)
    return disp, occ, conf, elapsed
