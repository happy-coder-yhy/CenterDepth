import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stereo_center import las2_tensorrt_inference
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import build_las2_tensorrt


class LAS2TensorRTBackendTests(unittest.TestCase):
    def test_export_trace_uses_single_sample_for_memory(self):
        self.assertEqual(build_las2_tensorrt.export_trace_batch_size(16), 1)

    def test_resolve_engine_paths_requires_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                las2_tensorrt_inference.resolve_engine_path(root)
            (root / "las2.engine").write_bytes(b"engine")
            self.assertEqual(
                las2_tensorrt_inference.resolve_engine_path(root), root / "las2.engine"
            )

    def test_batch_limit_reads_dynamic_profile(self):
        class FakeEngine:
            def get_tensor_profile_shape(self, name, profile):
                self.args = name, profile
                return ((1, 3, 672, 800), (8, 3, 672, 800), (16, 3, 672, 800))

        engine = FakeEngine()
        self.assertEqual(las2_tensorrt_inference.engine_batch_limit(engine, "left"), 16)
        self.assertEqual(engine.args, ("left", 0))

    def test_pad_static_spatial_keeps_original_content(self):
        image = torch.ones(2, 3, 650, 800)
        padded = las2_tensorrt_inference.pad_static_spatial(image, 672, 800)
        self.assertEqual(tuple(padded.shape), (2, 3, 672, 800))
        self.assertTrue(torch.equal(padded[:, :, :650], image))
        self.assertEqual(float(padded[:, :, 650:].sum()), 0.0)

    def test_run_engine_in_chunks_concatenates_original_order(self):
        class FakeRunner:
            def __init__(self):
                self.batches = []

            def run_trt(self, inputs):
                self.batches.append(inputs["left"].clone())
                return {"disparity": inputs["left"][:, :1]}

        runner = FakeRunner()
        left = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1)
        output = las2_tensorrt_inference.run_engine_in_chunks(
            runner, {"left": left}, batch_limit=2
        )
        self.assertEqual([batch.shape[0] for batch in runner.batches], [2, 2, 1])
        self.assertTrue(torch.equal(output[:, 0, 0, 0], torch.arange(5, dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
