"""VDEgo-C2 双鱼眼标定加载与平行双目校正。

当前为 minimal feasibility 版本：
- 使用 calibration.json 中的 KB4 鱼眼内参（与 OpenCV fisheye 模型一致）；
- 忽略左右相机间的小旋转（约 0.75°），采用“理想平行双目”假设
  （R=I, t=(B,0,0)）。实测极线对齐误差中位数 ~0.9px，
  满足 S²M² 官方建议的垂直视差 <2px 要求。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Kalibr 四元数 (x, y, z, w，标量在后) -> 旋转矩阵。"""
    x, y, z, w = qx, qy, qz, qw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_vdego_calibration(calib_path: str | Path) -> Dict:
    """读取 VDEgo-C2 calibration.json，返回校正所需参数。"""
    data = json.loads(Path(calib_path).read_text())["value0"]

    cams = []
    for it in data["intrinsics"]:
        p = it["intrinsics"]
        K = np.array(
            [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]],
            dtype=np.float64,
        )
        D = np.array([p["k1"], p["k2"], p["k3"], p["k4"]], dtype=np.float64)
        cams.append((K, D))

    poses = []
    for t in data["T_imu_cam"]:
        R = quat_to_rotmat(t["qx"], t["qy"], t["qz"], t["qw"])
        tt = np.array([t["px"], t["py"], t["pz"]], dtype=np.float64)
        poses.append((R, tt))

    resolution = tuple(int(v) for v in data["resolution"][0])  # (w, h)
    t1, t2 = poses[0][1], poses[1][1]
    baseline = float(np.linalg.norm(t1 - t2))  # 米

    return {
        "K1": cams[0][0],
        "D1": cams[0][1],
        "K2": cams[1][0],
        "D2": cams[1][1],
        "resolution": resolution,
        "baseline": baseline,
    }


def load_orbbec_calibration(calib_path: str | Path) -> Dict:
    """Read Orbbec's two-camera YAML calibration.

    The supplied translation is millimetres, and the second camera pose is
    expressed relative to the declared reference camera.  OpenCV's stereo
    rectification expects the same left-to-right transform after conversion to
    metres.
    """
    data = yaml.safe_load(Path(calib_path).read_text())
    cameras = data.get("cameras", [])
    if len(cameras) < 2:
        raise ValueError("Orbbec calibration must contain at least two cameras")
    reference_id = data.get("calibration_info", {}).get("reference_camera", cameras[0]["id"])
    by_id = {camera["id"]: camera for camera in cameras}
    if reference_id not in by_id:
        raise ValueError(f"reference camera {reference_id!r} is missing from calibration")
    left = by_id[reference_id]
    right = next(camera for camera in cameras if camera["id"] != reference_id)

    def camera_params(camera: Dict) -> tuple[np.ndarray, np.ndarray]:
        intr = camera["intrinsics"]
        dist = camera["distortion"]
        K = np.array(
            [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
            dtype=np.float64,
        )
        D = np.array([dist["k1"], dist["k2"], dist["k3"], dist["k4"]], dtype=np.float64)
        return K, D

    K1, D1 = camera_params(left)
    K2, D2 = camera_params(right)
    R_lr = np.asarray(right["extrinsics"]["rotation"], dtype=np.float64)
    t_lr = np.asarray(right["extrinsics"]["translation"], dtype=np.float64) * 1e-3
    return {
        "K1": K1,
        "D1": D1,
        "K2": K2,
        "D2": D2,
        "resolution": (int(left["image_width"]), int(left["image_height"])),
        "baseline": float(np.linalg.norm(t_lr)),
        "R_lr": R_lr,
        "t_lr": t_lr,
    }


def compute_rectification_maps(cal: Dict, output_size: Tuple[int, int] | None = None) -> Dict:
    """Generate fisheye stereo rectification maps.

    Legacy VDEgo calibration has no reliable relative rotation, so its original
    parallel-stereo approximation remains the default.  Calibrations providing
    ``R_lr``/``t_lr`` use their measured left-to-right transform.
    """
    src_size = cal["resolution"]
    out_size = output_size or src_size

    # Legacy VDEgo uses the original parallel-stereo approximation.
    R = np.asarray(cal.get("R_lr", np.eye(3)), dtype=np.float64)
    t = np.asarray(
        cal.get("t_lr", [cal["baseline"], 0.0, 0.0]), dtype=np.float64
    )

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        cal["K1"],
        cal["D1"],
        cal["K2"],
        cal["D2"],
        src_size,
        R,
        t,
        flags=cv2.fisheye.CALIB_ZERO_DISPARITY,
        newImageSize=out_size,
    )
    map1L, map2L = cv2.fisheye.initUndistortRectifyMap(
        cal["K1"], cal["D1"], R1, P1, out_size, cv2.CV_32FC1
    )
    map1R, map2R = cv2.fisheye.initUndistortRectifyMap(
        cal["K2"], cal["D2"], R2, P2, out_size, cv2.CV_32FC1
    )

    return {
        "mapsL": (map1L, map2L),
        "mapsR": (map1R, map2R),
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "output_size": out_size,
        "fx": float(P1[0, 0]),  # 校正后焦距（像素），与 output_size 对应
        "baseline": cal["baseline"],
    }


def rectify_pair(
    left: np.ndarray, right: np.ndarray, rect: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """对左右半图应用校正，返回校正后的 (left, right) BGR 图。"""
    m1L, m2L = rect["mapsL"]
    m1R, m2R = rect["mapsR"]
    rL = cv2.remap(left, m1L, m2L, cv2.INTER_LINEAR)
    rR = cv2.remap(right, m1R, m2R, cv2.INTER_LINEAR)
    return rL, rR
