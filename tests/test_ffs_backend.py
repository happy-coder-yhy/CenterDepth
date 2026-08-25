from pathlib import Path
from unittest import mock
import tempfile
import unittest
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stereo_center import stereo_backend
from stereo_center import ffs_inference


class FFSBackendTests(unittest.TestCase):
    def test_backend_registry_includes_ffs(self):
        self.assertIn("ffs", stereo_backend.BACKENDS)
        self.assertIs(stereo_backend.get_backend("ffs"), ffs_inference)

    def test_resolve_checkpoint_accepts_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "model_best_bp2_serialize.pth"
            ckpt.write_bytes(b"placeholder")

            self.assertEqual(ffs_inference.resolve_checkpoint(root), ckpt)

    def test_resolve_checkpoint_accepts_direct_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "custom_model.pth"
            ckpt.write_bytes(b"placeholder")

            self.assertEqual(ffs_inference.resolve_checkpoint(ckpt), ckpt)

    def test_lr_confidence_reflects_left_right_agreement(self):
        disp_left = torch.full((1, 3, 6), 2.0)
        disp_right_good = torch.full((1, 3, 6), 2.0)
        disp_right_bad = torch.zeros((1, 3, 6))

        conf_good = ffs_inference._lr_confidence(disp_left, disp_right_good)
        conf_bad = ffs_inference._lr_confidence(disp_left, disp_right_bad)

        valid_region = conf_good[:, :, 2:]
        self.assertGreater(float(valid_region.mean()), 0.99)
        self.assertLess(float(conf_bad[:, :, 2:].mean()), 0.2)

    def test_single_direction_preserves_batch_dimension_for_video_pipeline(self):
        left = torch.zeros(2, 3, 4, 5)
        right = torch.zeros(2, 3, 4, 5)
        disp = torch.ones(2, 4, 5)

        with mock.patch.object(ffs_inference, "_run_once", return_value=(disp, 0.25)):
            out_disp, occ, conf, elapsed = ffs_inference.run_stereo_matching(
                object(), left, right, "cpu", hiera="direct"
            )

        self.assertEqual(tuple(out_disp.shape), (2, 4, 5))
        self.assertEqual(tuple(occ.shape), (2, 4, 5))
        self.assertEqual(tuple(conf.shape), (2, 4, 5))
        self.assertEqual(elapsed, 0.25)


if __name__ == "__main__":
    unittest.main()
