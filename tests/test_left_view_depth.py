import unittest
import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "stereo_center" / "scripts" / "run_depth_video.py"
spec = importlib.util.spec_from_file_location("run_depth_video", SCRIPT_PATH)
run_depth_video = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_depth_video)


class LeftViewDepthTests(unittest.TestCase):
    def test_left_depth_uses_metric_stereo_formula_and_validity(self):
        disp = torch.tensor([[[2.0, 4.0, 0.0]]])
        occ = torch.tensor([[[1.0, 0.0, 1.0]]])

        depth, valid = run_depth_video.left_view_depth_from_disparity(
            disp, occ, fx=100.0, baseline=0.1, device="cpu"
        )

        self.assertEqual(depth.shape, (1, 1, 1, 3))
        self.assertEqual(valid.tolist(), [[[[True, False, False]]]])
        self.assertAlmostEqual(float(depth[0, 0, 0, 0]), 5.0)
        self.assertAlmostEqual(float(depth[0, 0, 0, 1]), 2.5)

    def test_paper_left_vis_ignores_visibility_mask_for_visualization(self):
        disp = torch.tensor([[[2.0, 4.0, 0.0]]])
        occ = torch.tensor([[[1.0, 0.0, 1.0]]])

        _depth, valid = run_depth_video.left_view_depth_from_disparity(
            disp, occ, fx=100.0, baseline=0.1, device="cpu", valid_mode="paper"
        )

        self.assertEqual(valid.tolist(), [[[[True, True, False]]]])


if __name__ == "__main__":
    unittest.main()
