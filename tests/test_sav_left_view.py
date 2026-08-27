import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "stereo_center"
    / "scripts"
    / "run_depth_video_sav.py"
)
spec = importlib.util.spec_from_file_location("run_depth_video_sav", SCRIPT_PATH)
run_depth_video_sav = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_depth_video_sav)


class SavLeftViewTests(unittest.TestCase):
    def test_gpu_peak_memory_reports_reserved_cuda_memory_in_gib(self):
        with (
            patch.object(run_depth_video_sav.torch.cuda, "is_available", return_value=True),
            patch.object(run_depth_video_sav.torch.cuda, "synchronize") as synchronize,
            patch.object(run_depth_video_sav.torch.cuda, "reset_peak_memory_stats") as reset,
            patch.object(
                run_depth_video_sav.torch.cuda,
                "max_memory_reserved",
                return_value=3 * 1024 * 1024 * 1024,
            ),
        ):
            enabled = run_depth_video_sav.reset_gpu_peak_memory("cuda:0")
            peak_gib = run_depth_video_sav.gpu_peak_memory_gib("cuda:0", enabled)

        self.assertTrue(enabled)
        self.assertEqual(peak_gib, 3.0)
        self.assertEqual(synchronize.call_count, 2)
        reset.assert_called_once_with("cuda:0")

    def test_left_view_depth_uses_left_disparity_without_center_fusion(self):
        disparity = torch.tensor([[[5.0, 0.0], [10.0, -1.0]]])

        depth, valid = run_depth_video_sav.left_view_depth_from_disparity(
            disparity, fx=100.0, baseline=0.2, device="cpu"
        )

        torch.testing.assert_close(depth[0, 0, 0, 0], torch.tensor(4.0))
        torch.testing.assert_close(depth[0, 0, 1, 0], torch.tensor(2.0))
        self.assertEqual(valid.tolist(), [[[[True, False], [True, False]]]])

    def test_sav_uses_orbbec_loader_for_yaml_calibration(self):
        loader = run_depth_video_sav.calibration_loader_for_path("calibration_camera.yaml")

        self.assertEqual(loader.__name__, "load_orbbec_calibration")


if __name__ == "__main__":
    unittest.main()
