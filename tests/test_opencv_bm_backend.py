from pathlib import Path
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stereo_center import stereo_backend


class OpenCVBMBackendTests(unittest.TestCase):
    def test_backend_registry_includes_opencv_bm(self):
        self.assertIn("opencv_bm", stereo_backend.BACKENDS)

    def test_backend_registry_resolves_opencv_bm(self):
        self.assertEqual(
            stereo_backend.get_backend("opencv_bm").__name__,
            "stereo_center.opencv_bm_inference",
        )

    def test_load_uses_classical_bm_defaults(self):
        model = stereo_backend.load("opencv_bm", "ignored", "", "cpu")

        self.assertEqual(model.num_disparities, 128)
        self.assertEqual(model.block_size, 15)
        self.assertEqual(model.uniqueness_ratio, 10)

    def test_load_rejects_invalid_bm_parameters(self):
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            stereo_backend.load(
                "opencv_bm", "ignored", "", "cpu", bm_num_disparities=20
            )
        with self.assertRaisesRegex(ValueError, "odd"):
            stereo_backend.load(
                "opencv_bm", "ignored", "", "cpu", bm_block_size=10
            )

    def test_recovers_known_horizontal_disparity_for_batched_input(self):
        height, width, disparity = 96, 192, 8
        rng = np.random.default_rng(7)
        left_image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        right_image = np.empty_like(left_image)
        right_image[:, :-disparity] = left_image[:, disparity:]
        right_image[:, -disparity:] = left_image[:, -1:]
        left = torch.from_numpy(left_image).permute(2, 0, 1).float().unsqueeze(0)
        right = torch.from_numpy(right_image).permute(2, 0, 1).float().unsqueeze(0)
        model = stereo_backend.load(
            "opencv_bm", "ignored", "", "cpu",
            bm_num_disparities=64, bm_block_size=9,
        )

        disp, occ, conf, elapsed = stereo_backend.run(
            "opencv_bm", model, left, right, "cpu"
        )

        self.assertEqual(tuple(disp.shape), (1, height, width))
        self.assertEqual(tuple(occ.shape), (1, height, width))
        self.assertTrue(torch.equal(conf, torch.ones_like(disp)))
        self.assertGreaterEqual(elapsed, 0.0)
        interior = disp[0, 16:-16, 80:-16]
        interior_valid = occ[0, 16:-16, 80:-16] > 0.5
        self.assertGreater(int(interior_valid.sum()), 100)
        self.assertAlmostEqual(
            float(interior[interior_valid].median()), float(disparity), delta=1.0
        )


if __name__ == "__main__":
    unittest.main()
