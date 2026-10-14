import sys
import unittest
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numcodecs import get_codec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stereo_center"))
from stereo_center.depth_sz3 import DepthLogSZ3


class DepthLogSZ3Tests(unittest.TestCase):
    def test_bounds_all_tiers_and_boundary_values(self):
        data = np.random.default_rng(12).uniform(0, 10, (1, 128, 160)).astype("<f4")
        data.ravel()[:8] = [1, np.nextafter(np.float32(1), np.float32(0)),
                           np.nextafter(np.float32(1), np.float32(2)), 10,
                           np.nextafter(np.float32(10), np.float32(0)),
                           np.nextafter(np.float32(0), np.float32(1)), 0.00001, 0.1]
        for alpha in (0.0001, 0.001, 0.002, 0.003, 0.1):
            codec = get_codec(DepthLogSZ3(data.shape, alpha).get_config())
            result = codec.decode(codec.encode(data))
            error = np.abs(result.astype("f8") - data)
            self.assertTrue((error <= alpha * np.maximum(1, data.astype("f8"))).all())
            self.assertTrue((result > 0).all())

    def test_exact_exception_bits_and_unchanged_input(self):
        bits = np.array([0, 0x80000000, 0x7fc01234, 0x7f800000, 0xff800000,
                         0xbf800000, 0x7f7fffff, 0x41200001], dtype="<u4")
        data = np.tile(bits, 1024).view("<f4").reshape(1, 64, 128)
        original = data.copy().view("<u4")
        codec = DepthLogSZ3(data.shape)
        result = codec.decode(codec.encode(data))
        np.testing.assert_array_equal(result.view("<u4"), original)
        np.testing.assert_array_equal(data.view("<u4"), original)

    def test_output_buffer_and_corruption(self):
        data = np.ones((1, 2, 3), dtype="<f4")
        codec = DepthLogSZ3(data.shape)
        payload = codec.encode(data)
        out = np.empty_like(data)
        self.assertIs(codec.decode(payload, out), out)
        np.testing.assert_array_equal(out, data)
        for bad in (payload[:8], payload[:-1], payload[:-1] + bytes([payload[-1] ^ 1])):
            with self.assertRaises(ValueError):
                codec.decode(bad)

    def test_invalid_parameters(self):
        for shape in ((), (0, 2), (1.5, 2), (True, 2)):
            with self.assertRaises(ValueError):
                DepthLogSZ3(shape)
        for alpha in (0, -1, np.nan, np.inf, 1, 0.000001):
            with self.assertRaises(ValueError):
                DepthLogSZ3((1, 2, 3), alpha)

    def test_fresh_process_registry_and_random_slices(self):
        import zarr
        from stereo_center.depth_zarr import DepthZarrWriter

        frames = np.random.default_rng(6).uniform(0.1, 10, (5, 32, 40)).astype("f4")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.zarr"
            with DepthZarrWriter(path, 32, 40) as writer:
                for frame in frames:
                    writer.append(frame)
            code = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "import stereo_center.depth_sz3, zarr; "
                "a=zarr.open_group(sys.argv[2],mode='r')['depth']; "
                "assert a.shape==(5,32,40); assert a[4,3:9,2:8].shape==(6,6)"
            )
            subprocess.run([sys.executable, "-c", code,
                            str(Path(__file__).resolve().parents[1] / "stereo_center"), str(path)], check=True)
            result = zarr.open_group(str(path), mode="r")["depth"][1:5:2, 2:7, 3:9]
            source = frames[1:5:2, 2:7, 3:9]
            self.assertTrue((np.abs(result.astype("f8") - source) <= 0.001 * np.maximum(1, source.astype("f8"))).all())


if __name__ == "__main__":
    unittest.main()
