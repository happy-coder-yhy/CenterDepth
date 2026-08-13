"""结果可视化工具：伪彩图、面板标注与总览图拼接。"""

from __future__ import annotations

import cv2
import numpy as np


def colorize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """深度 -> jet 伪彩图（无效区域为黑色）。

    归一化用 p98 百分位 + gamma 0.6：线性 jet 会把近场（小深度）压到
    很小的色域、边缘看起来模糊；gamma 扩展近场对比，让物体边缘清晰。
    """
    d = depth.copy()
    d[~valid] = np.nan
    if not np.isfinite(d).any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmax = np.nanpercentile(d, 98)
    norm = np.clip(np.nan_to_num(d / max(vmax, 1e-6), nan=0.0), 0, 1)
    norm = norm**0.6  # gamma：扩展近场（小深度）对比
    norm = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored


def colorize_map(x: np.ndarray, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """任意单通道图 -> 归一化伪彩图。"""
    norm = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cmap)


def to_gray_bgr(x: np.ndarray) -> np.ndarray:
    """单通道灰度 -> 三通道 BGR（便于与其他彩色面板拼接）。"""
    g = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """左上角加文字标签。"""
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out


def make_overview(
    rect_left: np.ndarray,
    center_rgb: np.ndarray,
    rect_right: np.ndarray,
    disp: np.ndarray,
    center_depth: np.ndarray,
    center_valid: np.ndarray,
    conf: np.ndarray,
) -> np.ndarray:
    """2x3 总览图：左校正 | 中心RGB | 右校正 / 视差 | 中心深度 | 置信度。"""
    top = np.hstack(
        [
            add_label(rect_left, "rect_left"),
            add_label(center_rgb, "center_rgb"),
            add_label(rect_right, "rect_right"),
        ]
    )
    bot = np.hstack(
        [
            add_label(colorize_map(disp), "disparity"),
            add_label(colorize_depth(center_depth, center_valid), "center_depth"),
            add_label(to_gray_bgr(conf), "confidence"),
        ]
    )
    if top.shape[0] != bot.shape[0]:
        bot = cv2.resize(bot, (top.shape[1], top.shape[0]))
    return np.vstack([top, bot])
