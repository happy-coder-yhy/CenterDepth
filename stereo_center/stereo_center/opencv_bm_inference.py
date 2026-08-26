"""OpenCV StereoBM backend for calibrated left-reference stereo matching."""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class OpenCVBMModel:
    num_disparities: int = 128
    block_size: int = 31
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1


def _validate(model: OpenCVBMModel) -> None:
    if model.num_disparities <= 0 or model.num_disparities % 16:
        raise ValueError("bm_num_disparities must be a positive multiple of 16")
    if model.block_size < 5 or model.block_size > 255 or model.block_size % 2 == 0:
        raise ValueError("bm_block_size must be an odd integer from 5 to 255")
    if model.uniqueness_ratio < 0:
        raise ValueError("bm_uniqueness_ratio must be non-negative")
    if model.speckle_window_size < 0 or model.speckle_range < 0:
        raise ValueError("BM speckle parameters must be non-negative")


def load_opencv_bm(
    _model_type: str = "StereoBM",
    _weights_dir: str = "",
    _device: str = "cpu",
    **kwargs,
) -> OpenCVBMModel:
    """Create a parameter-only StereoBM model; no checkpoint or GPU is used."""
    model = OpenCVBMModel(
        num_disparities=int(kwargs.get("bm_num_disparities", 128)),
        block_size=int(kwargs.get("bm_block_size", 31)),
        uniqueness_ratio=int(kwargs.get("bm_uniqueness_ratio", 10)),
        speckle_window_size=int(kwargs.get("bm_speckle_window_size", 100)),
        speckle_range=int(kwargs.get("bm_speckle_range", 2)),
        disp12_max_diff=int(kwargs.get("bm_disp12_max_diff", 1)),
    )
    _validate(model)
    return model


def _gray_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().float().permute(1, 2, 0).numpy()
    rgb = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _matcher(model: OpenCVBMModel) -> cv2.StereoBM:
    matcher = cv2.StereoBM_create(
        numDisparities=model.num_disparities,
        blockSize=model.block_size,
    )
    matcher.setUniquenessRatio(model.uniqueness_ratio)
    matcher.setSpeckleWindowSize(model.speckle_window_size)
    matcher.setSpeckleRange(model.speckle_range)
    matcher.setDisp12MaxDiff(model.disp12_max_diff)
    return matcher


def _visibility(disp: torch.Tensor) -> torch.Tensor:
    _, _, width = disp.shape
    x = torch.arange(width, device=disp.device).view(1, 1, width)
    return ((disp >= 0.5) & (x - disp >= 0) & (disp < width - 1)).float()


def run_stereo_matching(
    model: OpenCVBMModel,
    left: torch.Tensor,
    right: torch.Tensor,
    _device: str = "cpu",
    **_unused,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Run CPU StereoBM for each item of a ``(B,3,H,W)`` RGB batch."""
    if left.ndim != 4 or right.ndim != 4 or left.shape != right.shape:
        raise ValueError("StereoBM expects equal left/right tensors shaped (B, 3, H, W)")
    if left.shape[1] != 3:
        raise ValueError("StereoBM expects RGB tensors with three channels")

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
