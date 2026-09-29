import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stereo_center"))
from stereo_center.depth_sz3 import DepthSZ3


class DepthSZ3Tests(unittest.TestCase):
    def test_zarr_reopen_preserves_shape_and_bounds_error_with_large_outlier(self):
        data = np.random.default_rng(17).uniform(0.1, 5, (3, 128, 160)).astype("f4")
        data[1, 8, 10] = 15554015.0
        data[0, 0, 0] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            dst = zarr.open_array(
                tmp, mode="w", shape=data.shape, chunks=(2, 128, 160),
                dtype="f4", compressor=DepthSZ3((2, 128, 160), 0.001),
            )
            dst[:] = data
            restored = zarr.open_array(tmp, mode="r")
            self.assertEqual(restored.shape, data.shape)
            self.assertEqual(restored.dtype, data.dtype)
            self.assertLessEqual(float(np.abs(restored[:].astype("f8") - data).max()), 0.001)
            self.assertEqual(restored[0, 0, 0], 0.0)

    def test_invalid_mask_and_small_positive_depths_remain_distinct(self):
        data = np.random.default_rng(2).uniform(0.0001, 0.01, (1, 128, 160)).astype("f4")
        data[:, ::3, ::2] = 0.0
        codec = DepthSZ3(data.shape, 0.005)
        result = codec.decode(codec.encode(data))
        np.testing.assert_array_equal(result == 0, data == 0)
        self.assertTrue((result >= 0).all())
        self.assertLessEqual(float(np.abs(result.astype("f8") - data).max()), 0.005)

    def test_decode_into_supplied_buffer(self):
        data = np.linspace(0.1, 2, 35, dtype="f4").reshape(1, 5, 7)
        codec = DepthSZ3(data.shape, 0.001)
        out = np.empty_like(data)
        self.assertIs(codec.decode(codec.encode(data), out=out), out)
        self.assertLessEqual(float(np.abs(out.astype("f8") - data).max()), 0.001)

    def test_nonfinite_input_is_rejected(self):
        codec = DepthSZ3((1, 2, 2), 0.001)
        with self.assertRaisesRegex(ValueError, "finite"):
            codec.encode(np.full((1, 2, 2), np.nan, dtype="f4"))

    def test_tiny_chunk_uses_lossless_fallback(self):
        data = np.ones((1, 1, 1), dtype="f4")
        codec = DepthSZ3(data.shape, 0.005)
        np.testing.assert_array_equal(codec.decode(codec.encode(data)), data)


if __name__ == "__main__":
    unittest.main()
