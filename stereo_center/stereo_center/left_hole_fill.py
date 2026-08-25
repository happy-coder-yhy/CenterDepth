"""Conservative RGB-gated depth completion for small left-view holes."""

from __future__ import annotations

import cv2
import numpy as np


def fill_small_left_holes(
    depth: np.ndarray,
    valid: np.ndarray,
    guide_bgr: np.ndarray,
    max_area: int,
    color_tol: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Fill bounded interior invalid components from compatible boundary depths.

    This is deliberately a visualization postprocess, not a replacement for
    stereo matching: components touching an image edge or exceeding ``max_area``
    remain invalid. A component is filled only when at least three valid boundary
    pixels have similar grayscale intensity to the pixels inside the hole.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    guide_bgr = np.asarray(guide_bgr)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("depth and valid must be equally shaped 2D arrays")
    if guide_bgr.shape[:2] != depth.shape or guide_bgr.ndim != 3:
        raise ValueError("guide_bgr must have shape (H, W, C) matching depth")
    if max_area < 0:
        raise ValueError("max_area must be non-negative")
    if color_tol < 0:
        raise ValueError("color_tol must be non-negative")

    out_depth = depth.copy()
    out_valid = valid.copy()
    stats = {"filled_components": 0, "filled_pixels": 0}
    if max_area == 0 or out_valid.all():
        return out_depth, out_valid, stats

    height, width = depth.shape
    gray = cv2.cvtColor(guide_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        (~out_valid).astype(np.uint8), connectivity=8
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    for label in range(1, count):
        x, y, component_width, component_height, area = component_stats[label]
        if area > max_area:
            continue
        if x == 0 or y == 0 or x + component_width == width or y + component_height == height:
            continue
        x0, x1 = x - 1, x + component_width + 1
        y0, y1 = y - 1, y + component_height + 1
        component = labels[y0:y1, x0:x1] == label
        boundary = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary &= ~component
        local_valid = out_valid[y0:y1, x0:x1]
        local_depth = out_depth[y0:y1, x0:x1]
        local_gray = gray[y0:y1, x0:x1]
        boundary &= local_valid
        boundary &= np.isfinite(local_depth) & (local_depth > 0)
        component_gray = float(np.median(local_gray[component]))
        boundary &= np.abs(local_gray - component_gray) <= color_tol
        boundary_depths = local_depth[boundary]
        if boundary_depths.size < 3:
            continue
        local_depth[component] = float(np.median(boundary_depths))
        local_valid[component] = True
        stats["filled_components"] += 1
        stats["filled_pixels"] += int(area)
    return out_depth, out_valid, stats
