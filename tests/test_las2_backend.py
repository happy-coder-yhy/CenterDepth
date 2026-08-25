from unittest import mock
import unittest
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stereo_center import las2_inference


class LAS2BackendTests(unittest.TestCase):
    def test_single_direction_ignores_backend_compatibility_kwargs(self):
        left = torch.zeros(2, 3, 4, 5)
        right = torch.zeros(2, 3, 4, 5)
        disp = torch.ones(2, 4, 5)

        with mock.patch.object(las2_inference, "_run_once", return_value=(disp, 0.25)):
            out_disp, occ, conf, elapsed = las2_inference.run_stereo_matching(
                object(), left, right, "cpu", hiera="direct"
            )

        self.assertEqual(tuple(out_disp.shape), (2, 4, 5))
        self.assertEqual(tuple(occ.shape), (2, 4, 5))
        self.assertEqual(tuple(conf.shape), (2, 4, 5))
        self.assertEqual(elapsed, 0.25)

    def test_bidirectional_batch_ignores_backend_compatibility_kwargs(self):
        left = torch.zeros(2, 3, 4, 5)
        right = torch.zeros(2, 3, 4, 5)
        disp = torch.ones(2, 4, 5)

        with mock.patch.object(
            las2_inference, "_run_once", side_effect=[(disp, 0.25), (disp + 1, 0.5)]
        ):
            dL, dR, occL, occR, confL, confR, elapsed = (
                las2_inference.run_stereo_matching_bi_batch(
                    object(), left, right, "cpu", hiera="direct"
                )
            )

        self.assertEqual(tuple(dL.shape), (2, 4, 5))
        self.assertEqual(tuple(dR.shape), (2, 4, 5))
        self.assertEqual(tuple(occL.shape), (2, 4, 5))
        self.assertEqual(tuple(occR.shape), (2, 4, 5))
        self.assertEqual(tuple(confL.shape), (2, 4, 5))
        self.assertEqual(tuple(confR.shape), (2, 4, 5))
        self.assertEqual(elapsed, 0.75)


if __name__ == "__main__":
    unittest.main()
