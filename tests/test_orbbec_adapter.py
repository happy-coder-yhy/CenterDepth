import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "stereo_center"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stereo_center import calib
from stereo_center.orbbec import (
    forward_decode_read_count,
    match_left_to_right_pts,
    pts_metadata_mismatch,
)


class OrbbecCalibrationTests(unittest.TestCase):
    def test_zero_disparity_flag_supports_opencv5_namespace(self):
        fake_cv2 = SimpleNamespace(
            CALIB_ZERO_DISPARITY=1024,
            fisheye=SimpleNamespace(),
        )

        self.assertEqual(calib.zero_disparity_flag(fake_cv2), 1024)

    def test_load_orbbec_calibration_preserves_relative_pose_in_meters(self):
        content = """
calibration_info:
  reference_camera: cam_0
cameras:
  - id: cam_0
    image_width: 1600
    image_height: 1300
    intrinsics: {fx: 500.0, fy: 501.0, cx: 799.0, cy: 649.0}
    distortion: {k1: 0.1, k2: -0.01, k3: 0.001, k4: 0.0}
    extrinsics:
      rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
      translation: [0.0, 0.0, 0.0]
  - id: cam_1
    image_width: 1600
    image_height: 1300
    intrinsics: {fx: 502.0, fy: 503.0, cx: 801.0, cy: 651.0}
    distortion: {k1: 0.2, k2: -0.02, k3: 0.002, k4: 0.0}
    extrinsics:
      rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
      translation: [-120.0, -2.0, 1.0]
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration_camera.yaml"
            path.write_text(content)
            result = calib.load_orbbec_calibration(path)

        self.assertEqual(result["resolution"], (1600, 1300))
        np.testing.assert_allclose(
            result["R_lr"], [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(result["t_lr"], [-0.12, -0.002, 0.001])
        self.assertAlmostEqual(result["baseline"], np.linalg.norm([-0.12, -0.002, 0.001]))


class OrbbecSynchronizationTests(unittest.TestCase):
    def test_pts_matching_uses_nearest_right_frame_and_rejects_large_gaps(self):
        pairs, offsets = match_left_to_right_pts(
            np.array([100, 200, 300, 900]),
            np.array([95, 205, 290, 700]),
            max_delta_us=20,
        )

        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(offsets.tolist(), [5, 5, 10])

    def test_pts_remain_authoritative_when_container_frame_metadata_is_wrong(self):
        self.assertTrue(pts_metadata_mismatch(983, 975, 999, 992))
        self.assertFalse(pts_metadata_mismatch(983, 975, 983, 975))

    def test_forward_decode_uses_reads_for_monotonic_pts_indices(self):
        self.assertIsNone(forward_decode_read_count(None, 12))
        self.assertEqual(forward_decode_read_count(12, 12), 0)
        self.assertEqual(forward_decode_read_count(12, 15), 3)
        self.assertIsNone(forward_decode_read_count(15, 12))


if __name__ == "__main__":
    unittest.main()
