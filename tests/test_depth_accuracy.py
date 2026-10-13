import sys
import unittest
import subprocess
import tempfile
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stereo_center"))
from stereo_center.depth_accuracy import DepthAccuracy


class DepthAccuracyTests(unittest.TestCase):
    def test_known_errors_regions_and_missing_prediction(self):
        metric = DepthAccuracy()
        metric.update([1.001, 2.004, np.nan, 10.02, 99], [1, 2, 5, 10, 0])
        result = metric.result()
        self.assertEqual(result["all"]["reference_pixels"], 4)
        self.assertEqual(result["all"]["coverage"], 0.75)
        self.assertAlmostEqual(result["all"]["mae_m"], 0.025 / 3)
        self.assertAlmostEqual(result["all"]["rmse_m"], ((1e-6 + 16e-6 + 400e-6) / 3)**0.5)
        self.assertEqual(result["3-5m"]["invalid_prediction_pixels"], 1)
        self.assertIsNone(result[">10m"]["mae_m"])

    def test_mask_and_signed_bias(self):
        metric = DepthAccuracy()
        metric.update([0.9, 1.1, 100], [1, 1, 1], [True, True, False])
        self.assertAlmostEqual(metric.result()["all"]["bias_m"], 0)
        self.assertAlmostEqual(metric.result()["all"]["mae_m"], 0.1)
        with self.assertRaises(ValueError):
            metric.update([1], [1, 2])
        with self.assertRaises(ValueError):
            metric.update([1], [1], [True, False])

    def test_cli_uses_common_support_and_refuses_self_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = np.ones((1, 2, 3), dtype="f4")
            raw = gt + 0.01
            restored = raw + 0.001
            restored[0, 0, 0] = np.nan
            for name, array in (("gt", gt), ("raw", raw), ("stored", restored)):
                np.save(root / f"{name}.npy", array)
            command = [sys.executable, str(Path(__file__).resolve().parents[1] /
                       "stereo_center/scripts/evaluate_depth_accuracy.py"),
                       "--prediction", str(root / "raw.npy"), "--reference", str(root / "gt.npy"),
                       "--reconstructed", str(root / "stored.npy"), "--reference-kind", "ground_truth",
                       "--reference-description", "synthetic test only", "--aligned-metric-z-confirmed",
                       "--output", str(root / "report.json")]
            subprocess.run(command, check=True, capture_output=True)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["prediction"]["all"]["coverage"], 1)
            self.assertEqual(report["reconstructed"]["all"]["coverage"], 5 / 6)
            self.assertAlmostEqual(report["common_support_delta"]["all"]["mae_m"], 0.001, places=6)
            frames = root / "frames"
            frames.mkdir()
            np.save(frames / "depth_000000.npy", raw[0])
            frame_command = command.copy()
            frame_command[frame_command.index("--prediction") + 1] = str(frames)
            frame_command[frame_command.index("--output") + 1] = str(root / "frames_report.json")
            subprocess.run(frame_command, check=True, capture_output=True)
            self.assertEqual(json.loads((root / "frames_report.json").read_text())["prediction"], report["prediction"])
            command[command.index("--reference") + 1] = str(root / "raw.npy")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("independent", result.stderr)


if __name__ == "__main__":
    unittest.main()
