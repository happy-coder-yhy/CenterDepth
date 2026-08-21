"""RGB 引导滤波（guided filter）：用原图边缘锐化/对齐视差图。

LAS2 等前馈模型的视差在物体边缘处较平滑（soft-argmax 回归的固有特性），
直接用引导滤波把视差边缘对齐到左右原图的颜色边缘，可显著提升深度图锐度，
且不引入时间域伪影（纯空间操作，逐帧独立）。

实现为标准 guided filter（He et al. 2013），用 boxFilter 加速，输入输出
均为 numpy float32。
"""

from __future__ import annotations

import cv2
import numpy as np


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 300.0,
) -> np.ndarray:
    """guided filter：guide 为引导图（0-255 灰度），src 为被滤波图（视差）。"""
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    r = max(2, int(radius))
    mean_I = cv2.boxFilter(I, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    corr_I = cv2.boxFilter(I * I, -1, (r, r))
    corr_Ip = cv2.boxFilter(I * p, -1, (r, r))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    return (mean_a * I + mean_b).astype(np.float32)


def guided_filter_batch(
    bgr_pairs: list,
    dL: np.ndarray,
    dR: np.ndarray | None,
    radius: int = 8,
    eps: float = 300.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """批量引导滤波：dL 用左图灰度引导，dR 用右图灰度引导。

    bgr_pairs: [(left_bgr, right_bgr), ...]（校正后）；dL/dR: (B, H, W) float32。
    """
    out_l = np.empty_like(dL)
    out_r = np.empty_like(dR) if dR is not None else None
    for b, (l_bgr, r_bgr) in enumerate(bgr_pairs):
        gL = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2GRAY)
        out_l[b] = guided_filter(gL, dL[b], radius, eps)
        if dR is not None:
            gR = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2GRAY)
            out_r[b] = guided_filter(gR, dR[b], radius, eps)
    return out_l, out_r
