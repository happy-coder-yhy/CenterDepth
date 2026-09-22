import unittest

import numpy as np
import torch

from stereo_center.flow_metrics import flow_metrics, region_flow_metrics
from stereo_center.romav2_flow import warp_to_flow


class FlowTests(unittest.TestCase):
    def test_identity_and_translation_resize(self):
        h, w = 8, 16
        y, x = torch.meshgrid((torch.arange(h)+.5)*2/h-1,
                              (torch.arange(w)+.5)*2/w-1, indexing="ij")
        warp = torch.stack((x, y), -1)[None]
        self.assertLess(float(warp_to_flow(warp, 650, 800).abs().max()), 1e-5)
        result = warp_to_flow(warp + torch.tensor([10*2/800, -5*2/650]), 650, 800)
        self.assertTrue(torch.allclose(result[:, 0], torch.full_like(result[:, 0], 10), atol=1e-4))
        self.assertTrue(torch.allclose(result[:, 1], torch.full_like(result[:, 1], -5), atol=1e-4))

    def test_metrics_and_strict_thresholds(self):
        gt = np.array([[[0., 0.], [100., 0.], [100., 0.]]])
        pred = gt + np.array([[[3., 0.], [4., 0.], [6., 0.]]])
        result = flow_metrics(pred, gt, np.ones((1, 3), bool))
        self.assertAlmostEqual(result["AEPE_px"], 13/3)
        self.assertAlmostEqual(result["BadPix_3px_pct"], 200/3)
        self.assertAlmostEqual(result["KITTI_rule_outlier_pct"], 100/3)

    def test_angle_zero_invalid_and_regions(self):
        gt = np.zeros((2, 2, 2))
        result = flow_metrics(gt, gt, np.ones((2, 2), bool))
        self.assertEqual(result["mean_AE_deg"], 0)
        self.assertIsNone(result["direction_AE_deg"])
        result = region_flow_metrics(gt, gt, np.ones((2, 2), bool), np.eye(2, dtype=bool))
        self.assertEqual(result["OCC"]["count"], 2)
        self.assertEqual(result["s40_plus"]["count"], 0)
        pred = gt.copy(); pred[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            flow_metrics(pred, gt, np.ones((2, 2), bool))


if __name__ == "__main__":
    unittest.main()
