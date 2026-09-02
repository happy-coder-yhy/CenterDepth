from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_ffs_tensorrt
from stereo_center import ffs_tensorrt_inference


class FFSTensorRTBackendTests(unittest.TestCase):
    def test_resolve_engine_paths_requires_both_static_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "feature_runner.engine").write_bytes(b"feature")
            with self.assertRaises(FileNotFoundError):
                ffs_tensorrt_inference.resolve_engine_paths(root)

            (root / "post_runner.engine").write_bytes(b"post")
            self.assertEqual(
                ffs_tensorrt_inference.resolve_engine_paths(root),
                (root / "feature_runner.engine", root / "post_runner.engine"),
            )

    def test_static_batch_padding_keeps_original_samples(self):
        left = torch.arange(2 * 3 * 2 * 3).reshape(2, 3, 2, 3)
        right = left + 100

        padded_left, padded_right = ffs_tensorrt_inference.pad_static_batch(
            left, right, 4
        )

        self.assertEqual(tuple(padded_left.shape), (4, 3, 2, 3))
        self.assertTrue(torch.equal(padded_left[:2], left))
        self.assertTrue(torch.equal(padded_right[:2], right))
        self.assertTrue(torch.equal(padded_left[2], left[-1]))
        self.assertTrue(torch.equal(padded_right[3], right[-1]))

    def test_static_spatial_padding_is_removed_from_output(self):
        image = torch.ones(1, 3, 650, 800)
        padded = ffs_tensorrt_inference.pad_static_spatial(image, 672, 800)

        self.assertEqual(tuple(padded.shape), (1, 3, 672, 800))
        self.assertTrue(torch.equal(padded[:, :, :650], image))
        self.assertEqual(float(padded[:, :, 650:].sum()), 0.0)

    def test_post_runner_chunks_batch_to_its_engine_limit(self):
        class FakeRunner:
            def __init__(self):
                self.batches = []

            def run_trt(self, _engine, _context, inputs):
                self.batches.append(inputs["features_left_04"].clone())
                return {"disp": inputs["features_left_04"][:, :1]}

        runner = FakeRunner()
        inputs = {
            "features_left_04": torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1),
            "gwc_volume": torch.ones(4, 1, 1, 1),
        }

        disp = ffs_tensorrt_inference.run_post_in_chunks(
            runner, object(), object(), inputs, batch_limit=1
        )

        self.assertEqual([batch.shape[0] for batch in runner.batches], [1, 1, 1, 1])
        self.assertTrue(torch.equal(disp[:, 0, 0, 0], torch.arange(4, dtype=torch.float32)))

    def test_engine_batch_limit_uses_profile_maximum(self):
        class FakeEngine:
            def get_tensor_profile_shape(self, input_name, profile_index):
                self.input_name = input_name
                self.profile_index = profile_index
                return ((1, 224, 168, 200), (6, 224, 168, 200), (12, 224, 168, 200))

        engine = FakeEngine()
        self.assertEqual(
            ffs_tensorrt_inference.engine_batch_limit(engine, "features_left_04"), 12
        )
        self.assertEqual((engine.input_name, engine.profile_index), ("features_left_04", 0))

    def test_feature_only_export_traces_at_target_batch(self):
        self.assertEqual(
            build_ffs_tensorrt.export_trace_batch_size("feature", 12), 12
        )

    def test_post_export_traces_at_single_batch(self):
        self.assertEqual(
            build_ffs_tensorrt.export_trace_batch_size("post", 12), 1
        )

    def test_combined_export_uses_distinct_feature_and_post_trace_batches(self):
        self.assertEqual(
            build_ffs_tensorrt.export_trace_batch_sizes("both", 12), (12, 1)
        )

    def test_fp16_export_converts_model_and_input_dtype(self):
        class FakeModel:
            def __init__(self):
                self.dtype = torch.float32
                self.half_calls = 0

            def half(self):
                self.half_calls += 1
                return self

        model = FakeModel()

        dtype = build_ffs_tensorrt.prepare_model_for_export(model, fp16=True)

        self.assertEqual(dtype, torch.float16)
        self.assertEqual(model.dtype, torch.float16)
        self.assertEqual(model.half_calls, 1)

    def test_fp32_export_preserves_model_dtype(self):
        class FakeModel:
            def __init__(self):
                self.dtype = torch.float32
                self.half_calls = 0

            def half(self):
                self.half_calls += 1
                return self

        model = FakeModel()

        dtype = build_ffs_tensorrt.prepare_model_for_export(model, fp16=False)

        self.assertEqual(dtype, torch.float32)
        self.assertEqual(model.dtype, torch.float32)
        self.assertEqual(model.half_calls, 0)

    def test_engine_dimensions_accept_padded_vdego_scale_half_shape(self):
        self.assertEqual(
            build_ffs_tensorrt.validate_engine_dimensions(608, 960), (608, 960)
        )

    def test_network_flags_support_tensor_rt_versions_with_or_without_explicit_batch(self):
        legacy = SimpleNamespace(
            NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=2)
        )
        modern = SimpleNamespace(NetworkDefinitionCreationFlag=SimpleNamespace())

        self.assertEqual(build_ffs_tensorrt.network_creation_flags(legacy), 1 << 2)
        self.assertEqual(build_ffs_tensorrt.network_creation_flags(modern), 0)

    def test_fp16_flag_is_optional_on_strongly_typed_tensor_rt(self):
        class Config:
            def __init__(self):
                self.flags = []

            def set_flag(self, flag):
                self.flags.append(flag)

        legacy = SimpleNamespace(BuilderFlag=SimpleNamespace(FP16=7))
        modern = SimpleNamespace(BuilderFlag=SimpleNamespace())
        legacy_config = Config()
        modern_config = Config()

        self.assertTrue(build_ffs_tensorrt.enable_fp16(legacy, legacy_config))
        self.assertFalse(build_ffs_tensorrt.enable_fp16(modern, modern_config))
        self.assertEqual(legacy_config.flags, [7])
        self.assertEqual(modern_config.flags, [])

    def test_onnx_dtype_fallback_is_only_used_without_builder_fp16(self):
        legacy = SimpleNamespace(BuilderFlag=SimpleNamespace(FP16=7))
        modern = SimpleNamespace(BuilderFlag=SimpleNamespace())

        self.assertFalse(build_ffs_tensorrt.needs_fp16_onnx_export(legacy))
        self.assertTrue(build_ffs_tensorrt.needs_fp16_onnx_export(modern))


if __name__ == "__main__":
    unittest.main()
