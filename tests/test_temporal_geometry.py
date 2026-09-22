import unittest

import cv2
import numpy as np

from stereo_center.temporal_geometry import estimate_relative_pose_pnp, reproject_depth_zbuffer


class TemporalGeometryTests(unittest.TestCase):
    def test_pnp_pose_and_zbuffer_reprojection(self):
        height, width = 72, 96
        K = np.array([[80.0, 0.0, 47.5], [0.0, 80.0, 35.5], [0.0, 0.0, 1.0]])
        depth = np.full((height, width), 2.0, np.float32)
        yy, xx = np.indices((height, width), dtype=np.float32)
        rvec = np.array([[0.0], [0.015], [0.0]], dtype=np.float64)
        expected_rotation, _ = cv2.Rodrigues(rvec)
        expected_translation = np.array([0.025, -0.01, 0.04])
        points = np.stack(((xx-K[0, 2])*depth/K[0, 0], (yy-K[1, 2])*depth/K[1, 1], depth), axis=-1)
        transformed = points @ expected_rotation.T + expected_translation
        u = K[0, 0]*transformed[..., 0]/transformed[..., 2] + K[0, 2]
        v = K[1, 1]*transformed[..., 1]/transformed[..., 2] + K[1, 2]
        flow = np.stack((u-xx, v-yy), axis=-1).astype(np.float32)
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)

        pose = estimate_relative_pose_pnp(depth, flow, K, valid, max_samples=2000)

        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.rotation, expected_rotation, atol=2e-3)
        np.testing.assert_allclose(pose.translation_m, expected_translation, atol=2e-3)
        prior, target_valid, error = reproject_depth_zbuffer(depth, pose, K, valid, np.ones_like(valid), previous_to_current_flow=flow)
        self.assertGreater(target_valid.mean(), 0.8)
        self.assertLess(np.nanmean(error[valid]), 0.05)
        self.assertTrue(np.all(prior[target_valid] > 0))

    def test_invalid_or_nonrigid_flow_rejected(self):
        depth = np.ones((8, 8), np.float32)
        K = np.array([[10.0, 0.0, 3.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]])
        flow = np.zeros((8, 8, 2), np.float32)
        self.assertIsNone(estimate_relative_pose_pnp(depth, flow, K, np.zeros((8, 8), bool)))


if __name__ == "__main__":
    unittest.main()
