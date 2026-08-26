from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stereo_center import stereo_backend
from stereo_center import stereonet_inference


def _fake_soft_argmin(cost: torch.Tensor, _: int) -> torch.Tensor:
    return cost


class _FakeFeatureExtractor(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


class _FakeCostVolumizer(nn.Module):
    def forward(self, inputs, side: str = "left") -> torch.Tensor:
        del side
        batch, _, height, width = inputs[0].shape
        return torch.full((batch, 1, height, width), width - 1.0)


class _FakeRefiner(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full_like(inputs[:, -1:], 15.0) - inputs[:, -1:]


class _FakeStereoNet(nn.Module):
    candidate_disparities = 256
    k_refinement_layers = 3

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = _FakeFeatureExtractor()
        self.cost_volumizer = _FakeCostVolumizer()
        self.refiners = nn.ModuleList([_FakeRefiner() for _ in range(3)])


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

    def test_source_revision_validation_requires_the_pinned_revision(self):
        with patch.object(
            stereonet_inference,
            "_source_revision",
            return_value=stereonet_inference.PINNED_SOURCE_REVISION,
        ):
            self.assertEqual(
                stereonet_inference.validate_source_revision(Path("source")),
                stereonet_inference.PINNED_SOURCE_REVISION,
            )
        with patch.object(stereonet_inference, "_source_revision", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "source revision mismatch"):
                stereonet_inference.validate_source_revision(Path("source"))

    def test_checkpoint_sha256_validation_rejects_mismatch(self):
        contents = b"test checkpoint"
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.ckpt"
            checkpoint.write_bytes(contents)
            expected = "f908001a4b96aac17dfcc9519072c6282ad28800926524bc5178523070356e32"
            with patch.object(stereonet_inference, "CHECKPOINT_SHA256", expected):
                self.assertEqual(
                    stereonet_inference.verify_checkpoint_sha256(checkpoint), expected
                )
            with patch.object(stereonet_inference, "CHECKPOINT_SHA256", "0" * 64):
                with self.assertRaisesRegex(RuntimeError, "checkpoint SHA-256 mismatch"):
                    stereonet_inference.verify_checkpoint_sha256(checkpoint)

    def test_run_returns_batched_visibility_confidence_and_timing(self):
        wrapper = stereonet_inference.StereoNetModel(
            model=_FakeStereoNet(),
            checkpoint=Path("checkpoint.ckpt"),
            source_root=Path("source"),
            source_revision=stereonet_inference.PINNED_SOURCE_REVISION,
            max_side=16,
            soft_argmin=_fake_soft_argmin,
        )
        left = torch.full((2, 3, 8, 16), 255.0)
        right = torch.full((2, 3, 8, 16), 255.0)
        timing_out = {"existing": 1.0}

        disparity, visibility, confidence, elapsed = stereonet_inference.run_stereo_matching(
            wrapper, left, right, device="cpu", timing_out=timing_out
        )

        self.assertEqual(tuple(disparity.shape), (2, 8, 16))
        self.assertTrue(torch.allclose(disparity, torch.full_like(disparity, 15.0)))
        self.assertTrue(torch.equal(visibility, torch.zeros_like(visibility)))
        self.assertTrue(torch.equal(confidence, torch.ones_like(confidence)))
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(timing_out["existing"], 1.0)
        self.assertEqual(timing_out, wrapper.timing | {"existing": 1.0})
        self.assertIn("stereo_forward_seconds", timing_out)
        self.assertEqual(
            timing_out["model_forward_seconds"], timing_out["stereo_forward_seconds"]
        )

    def test_stereonet_remains_unsupported_by_bidirectional_dispatchers(self):
        with self.assertRaises(ValueError):
            stereo_backend.run_bi("stereonet", None, None, None, "cpu")
        with self.assertRaises(ValueError):
            stereo_backend.run_bi_batch("stereonet", None, None, None, "cpu")

    def test_load_rejects_unknown_model_type(self):
        with self.assertRaisesRegex(ValueError, "stereonet_sceneflow_rgb"):
            stereonet_inference.load_stereonet("wrong", "weights/stereonet", "cpu")

    def test_load_restores_foreign_modules_after_compatibility_import(self):
        saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "stereonet",
                "pytorch_lightning",
                "pytorch_lightning.callbacks",
                "pytorch_lightning.callbacks.model_checkpoint",
            )
        }
        foreign_stereonet = types.ModuleType("stereonet")
        foreign_stereonet.__file__ = "/tmp/foreign-stereonet/__init__.py"
        sys.modules["stereonet"] = foreign_stereonet
        for name in tuple(saved_modules):
            if name != "stereonet":
                sys.modules.pop(name, None)
        try:
            wrapper = stereonet_inference.load_stereonet(
                "stereonet_sceneflow_rgb", "weights/stereonet", "cpu"
            )
            self.assertIs(sys.modules["stereonet"], foreign_stereonet)
            self.assertNotIn("pytorch_lightning", sys.modules)
            self.assertEqual(
                wrapper.source_revision, stereonet_inference.PINNED_SOURCE_REVISION
            )
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_total_timing_synchronizes_before_start_and_elapsed_read(self):
        wrapper = stereonet_inference.StereoNetModel(
            model=_FakeStereoNet(),
            checkpoint=Path("checkpoint.ckpt"),
            source_root=Path("source"),
            source_revision=stereonet_inference.PINNED_SOURCE_REVISION,
            max_side=16,
            soft_argmin=_fake_soft_argmin,
        )
        events: list[str] = []

        def synchronize(_: str) -> None:
            events.append("sync")

        def clock() -> float:
            events.append("clock")
            return float(len(events))

        with patch.object(stereonet_inference, "_synchronize", synchronize), patch.object(
            stereonet_inference.time, "perf_counter", clock
        ):
            stereonet_inference.run_stereo_matching(
                wrapper,
                torch.full((1, 3, 8, 16), 255.0),
                torch.full((1, 3, 8, 16), 255.0),
                device="cpu",
            )

        self.assertEqual(events[:2], ["sync", "clock"])
        self.assertEqual(events[-2:], ["sync", "clock"])


if __name__ == "__main__":
    unittest.main()
