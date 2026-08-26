from pathlib import Path
import sys
import tempfile
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stereo_center import stereo_backend
from stereo_center import stereonet_inference


class StereoNetBackendTests(unittest.TestCase):
    def test_backend_registry_resolves_stereonet_adapter(self):
        self.assertIn("stereonet", stereo_backend.BACKENDS)
        self.assertEqual(
            stereo_backend.get_backend("stereonet").__name__,
            "stereo_center.stereonet_inference",
        )

    def test_resolve_checkpoint_accepts_direct_ckpt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "custom.ckpt"
            checkpoint.write_bytes(b"placeholder")

            self.assertEqual(stereonet_inference.resolve_checkpoint(checkpoint), checkpoint)

    def test_resolve_checkpoint_accepts_directory_with_expected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint = directory / "epoch=20-step=744533.ckpt"
            checkpoint.write_bytes(b"placeholder")

            self.assertEqual(stereonet_inference.resolve_checkpoint(directory), checkpoint)

    def test_resolve_checkpoint_reports_expected_filename_when_missing(self):
        expected_name = "epoch=20-step=744533.ckpt"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, expected_name):
                stereonet_inference.resolve_checkpoint(Path(tmp))

    def test_normalize_state_dict_keys_removes_one_model_prefix_only(self):
        state_dict = {
            "model.encoder.weight": torch.tensor([1.0]),
            "model.model.decoder.bias": torch.tensor([2.0]),
            "feature_extractor.weight": torch.tensor([3.0]),
        }

        normalized = stereonet_inference.normalize_state_dict_keys(state_dict)

        self.assertEqual(
            set(normalized),
            {"encoder.weight", "model.decoder.bias", "feature_extractor.weight"},
        )
        self.assertIs(normalized["feature_extractor.weight"], state_dict["feature_extractor.weight"])

    def test_restore_disparity_resizes_and_undoes_horizontal_scale(self):
        disparity = torch.full((1, 25, 40), 10.0)

        restored = stereonet_inference.restore_disparity(
            disparity, output_hw=(50, 80), scale_x=0.5
        )

        self.assertEqual(tuple(restored.shape), (1, 50, 80))
        self.assertTrue(torch.equal(restored, torch.full((1, 50, 80), 20.0)))

    def test_prepare_inputs_resizes_pair_and_normalizes_rgb(self):
        left = torch.zeros((2, 3, 100, 200), dtype=torch.float32)
        right = torch.full((2, 3, 100, 200), 255.0, dtype=torch.float32)

        prepared_left, prepared_right, scale_x = stereonet_inference.prepare_inputs(
            left, right, max_side=125
        )

        self.assertEqual(scale_x, 0.625)
        self.assertEqual(tuple(prepared_left.shape), (2, 3, 62, 125))
        self.assertEqual(tuple(prepared_right.shape), (2, 3, 62, 125))
        self.assertTrue(torch.isfinite(prepared_left).all())
        self.assertTrue(torch.isfinite(prepared_right).all())
        self.assertTrue(torch.equal(prepared_left, torch.full_like(prepared_left, -1.0)))
        self.assertTrue(torch.equal(prepared_right, torch.full_like(prepared_right, 1.0)))


if __name__ == "__main__":
    unittest.main()
