import unittest
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "stereo_center" / "scripts" / "run_depth_video.py"
spec = importlib.util.spec_from_file_location("run_depth_video", SCRIPT_PATH)
run_depth_video = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_depth_video)


class DepthVideoTimingTests(unittest.TestCase):
    def test_timing_artifact_name_uses_backend(self):
        self.assertEqual(run_depth_video.timing_artifact_name("waft"), "waft_timing.json")
        self.assertEqual(run_depth_video.timing_artifact_name("ffs"), "ffs_timing.json")
        self.assertEqual(run_depth_video.timing_artifact_name("las2"), "las2_timing.json")


if __name__ == "__main__":
    unittest.main()
