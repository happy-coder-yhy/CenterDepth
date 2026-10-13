import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "stereo_center/scripts/evaluate_middlebury_storage.py"
spec = importlib.util.spec_from_file_location("middlebury_eval", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MiddleburyStorageTests(unittest.TestCase):
    def test_disparity_depth_uses_doffs_and_mm(self):
        result = module.metric_depth(np.array([[90, 190, np.inf, -1]], dtype="f4"),
                                     {"fx": 1000, "baseline_mm": 100, "doffs": 10})
        np.testing.assert_array_equal(result, [[1, 0.5, 0, 0]])

    def test_pfm_endianness_and_vertical_flip(self):
        data = np.array([[1, 2, np.inf], [4, 5, 6]], dtype="f4")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pfm"
            for dtype, scale in (("<f4", -1), (">f4", 1), ("<f4", -0.003922)):
                path.write_bytes(f"Pf\n# comment\n3 2\n{scale}\n".encode() + np.flipud(data).astype(dtype).tobytes())
                np.testing.assert_array_equal(module.read_pfm(path), data)
            path.write_bytes(b"Pf\n3 2\n-1\n")
            with self.assertRaises(ValueError):
                module.read_pfm(path)

    def test_masks_and_invalid_predictions(self):
        gt = np.array([[1, 2, np.inf]], dtype="f4")
        pred = np.array([[1.5, np.inf, 99]], dtype="f4")
        result, _ = module.disparity_metrics(pred, gt, np.ones_like(gt, dtype=bool))
        self.assertEqual(result["gt_pixels"], 2)
        self.assertEqual(result["evaluated_pixels"], 1)
        self.assertEqual(result["epe_px"], 0.5)

    def test_calibration_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.txt"
            path.write_text("cam0=[1000 0 400; 0 1000 300; 0 0 1]\nbaseline=100\ndoffs=10\nwidth=800\nheight=600\n")
            self.assertEqual(module.read_calibration(path)["fx"], 1000)


if __name__ == "__main__":
    unittest.main()
