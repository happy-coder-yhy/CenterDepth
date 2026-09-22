"""Metric two-view geometry for temporal stereo-depth priors."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RelativePose:
    """Transform from the previous rectified camera to the current camera."""

    rotation: np.ndarray
    translation_m: np.ndarray
    correspondence_count: int
    inlier_count: int
    reprojection_rmse_px: float

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / self.correspondence_count


def estimate_relative_pose_pnp(
    previous_depth_m: np.ndarray,
    previous_to_current_flow: np.ndarray,
    camera_matrix: np.ndarray,
    valid_source: np.ndarray,
    *,
    max_samples: int = 6000,
    ransac_reprojection_px: float = 2.0,
    min_inliers: int = 80,
) -> RelativePose | None:
    """Estimate previous-camera to current-camera pose from metric depth/flow.

    The 3-D source points come from the previous rectified stereo depth. Dense
    optical flow supplies their 2-D current-frame observations. RANSAC rejects
    independently moving points, occlusions, and bad depth/flow correspondences.
    """
    depth = np.asarray(previous_depth_m, dtype=np.float32)
    flow = np.asarray(previous_to_current_flow, dtype=np.float32)
    valid = np.asarray(valid_source, dtype=bool)
    if depth.ndim != 2 or flow.shape != (*depth.shape, 2) or valid.shape != depth.shape:
        raise ValueError("depth/valid must be H,W and flow must be H,W,2")
    K = np.asarray(camera_matrix, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("camera_matrix must be 3x3")

    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    target_x, target_y = xx + flow[..., 0], yy + flow[..., 1]
    usable = valid & np.isfinite(depth) & (depth > 0)
    usable &= np.isfinite(flow).all(axis=-1)
    usable &= (target_x >= 0) & (target_x < width) & (target_y >= 0) & (target_y < height)
    indices = np.flatnonzero(usable)
    if indices.size < min_inliers:
        return None
    if indices.size > max_samples:
        indices = indices[np.linspace(0, indices.size - 1, max_samples, dtype=np.int64)]

    x, y, z = xx.ravel()[indices], yy.ravel()[indices], depth.ravel()[indices]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    object_points = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z)).astype(np.float32)
    image_points = np.column_stack((target_x.ravel()[indices], target_y.ravel()[indices])).astype(np.float32)
    success, rvec, translation, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        K,
        None,
        iterationsCount=200,
        reprojectionError=float(ransac_reprojection_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return None
    inlier_indices = inliers.reshape(-1)
    # LM refinement makes the pose stable without allowing rejected points back in.
    rvec, translation = cv2.solvePnPRefineLM(
        object_points[inlier_indices], image_points[inlier_indices], K, None, rvec, translation
    )
    projected, _ = cv2.projectPoints(object_points[inlier_indices], rvec, translation, K, None)
    residual = projected.reshape(-1, 2) - image_points[inlier_indices]
    rotation, _ = cv2.Rodrigues(rvec)
    return RelativePose(
        rotation=rotation.astype(np.float32),
        translation_m=translation.reshape(3).astype(np.float32),
        correspondence_count=int(indices.size),
        inlier_count=int(inlier_indices.size),
        reprojection_rmse_px=float(np.sqrt(np.mean(np.square(residual), dtype=np.float64))),
    )


def reproject_depth_zbuffer(
    previous_depth_m: np.ndarray,
    pose: RelativePose,
    camera_matrix: np.ndarray,
    valid_source: np.ndarray,
    target_valid: np.ndarray,
    *,
    previous_to_current_flow: np.ndarray | None = None,
    max_flow_disagreement_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject previous metric depth into the current camera with a z-buffer.

    When flow is supplied, source pixels must agree with the estimated rigid
    camera motion. This rejects dynamic objects and unreliable correspondences
    before they can seed FFS in the current image.
    """
    depth = np.asarray(previous_depth_m, dtype=np.float32)
    source_valid = np.asarray(valid_source, dtype=bool)
    target_valid = np.asarray(target_valid, dtype=bool)
    if depth.ndim != 2 or source_valid.shape != depth.shape or target_valid.shape != depth.shape:
        raise ValueError("depth and masks must share H,W shape")
    K = np.asarray(camera_matrix, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("camera_matrix must be 3x3")
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    source = source_valid & np.isfinite(depth) & (depth > 0)
    if previous_to_current_flow is not None:
        flow = np.asarray(previous_to_current_flow, dtype=np.float32)
        if flow.shape != (*depth.shape, 2):
            raise ValueError("flow must have shape H,W,2")
        source &= np.isfinite(flow).all(axis=-1)
    selected = np.flatnonzero(source)
    prior = np.zeros_like(depth)
    target_mask = np.zeros_like(source)
    rigid_error = np.full_like(depth, np.nan)
    if not selected.size:
        return prior, target_mask, rigid_error

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth.ravel()[selected]
    points = np.stack(
        ((xx.ravel()[selected] - cx) * z / fx, (yy.ravel()[selected] - cy) * z / fy, z), axis=1
    )
    transformed = points @ pose.rotation.T + pose.translation_m[None]
    z_current = transformed[:, 2]
    u = fx * transformed[:, 0] / z_current + cx
    v = fy * transformed[:, 1] / z_current + cy
    in_front = z_current > 0
    if previous_to_current_flow is not None:
        flow_flat = flow.reshape(-1, 2)[selected]
        error = np.linalg.norm(np.column_stack((u, v)) - np.column_stack((xx.ravel()[selected], yy.ravel()[selected])) - flow_flat, axis=1)
        rigid_error.ravel()[selected] = error
        in_front &= error <= float(max_flow_disagreement_px)
    ui, vi = np.rint(u).astype(np.int32), np.rint(v).astype(np.int32)
    in_bounds = in_front & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not in_bounds.any():
        return prior, target_mask, rigid_error
    linear = vi[in_bounds] * width + ui[in_bounds]
    z_buffer = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, linear, z_current[in_bounds].astype(np.float32))
    target_mask = np.isfinite(z_buffer.reshape(height, width)) & target_valid
    prior[target_mask] = z_buffer.reshape(height, width)[target_mask]
    return prior, target_mask, rigid_error
