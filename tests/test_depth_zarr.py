import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "stereo_center"
import sys

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stereo_center.depth_zarr import DEPTH_CHUNK_FRAMES, DepthZarrWriter


class DepthZarrWriterTests(unittest.TestCase):
    def test_appends_metric_frames_in_order_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.zarr"
            writer = DepthZarrWriter(
                path,
                height=2,
                width=3,
                metadata={
                    "fps": 30.0,
                    "backend": "ffs",
                    "scale": 0.5,
                    "output_view": "left",
                },
            )
            writer.append(np.full((2, 3), 1.5, dtype=np.float32))
            writer.append(np.arange(6, dtype=np.float32).reshape(2, 3))
            writer.close()

            import zarr

            root = zarr.open_group(str(path), mode="r")
            depth = root["depth"]
            self.assertEqual(depth.shape, (2, 2, 3))
            self.assertEqual(depth.dtype, np.dtype("float32"))
            np.testing.assert_allclose(depth[0], 1.5, atol=0.0005)
            np.testing.assert_allclose(
                depth[1], np.arange(6, dtype=np.float32).reshape(2, 3), atol=0.0005
            )
            self.assertEqual(root.attrs["n_frames"], 2)
            self.assertEqual(root.attrs["height"], 2)
            self.assertEqual(root.attrs["width"], 3)
            self.assertEqual(root.attrs["backend"], "ffs")
            self.assertEqual(root.attrs["storage_encoding"], "sz3")
            self.assertEqual(root.attrs["storage_dtype"], "float32")
            self.assertEqual(root.attrs["absolute_error_tolerance_m"], 0.001)
            self.assertEqual(root.attrs["error_bound_mode"], "ABS")
            self.assertEqual(depth.compressor.codec_id, "centerdepth.sz3.v1")
            self.assertFalse(depth.filters)
            self.assertEqual(depth.chunks, (DEPTH_CHUNK_FRAMES, 2, 3))
            self.assertEqual(root.attrs["temporal_chunk_frames"], DEPTH_CHUNK_FRAMES)

    def test_each_chunk_is_encoded_once_and_tail_is_flushed_on_close(self):
        import zarr
        from stereo_center.depth_sz3 import DepthSZ3

        frames = np.random.default_rng(4).uniform(0.1, 4, (11, 64, 80)).astype("f4")
        frames[:, 0, 0] = 0
        encode = DepthSZ3.encode
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            DepthSZ3, "encode", autospec=True, side_effect=encode
        ) as encoder:
            path = Path(tmp) / "depth.zarr"
            writer = DepthZarrWriter(path, 64, 80, chunk_frames=4)
            for i, frame in enumerate(frames):
                writer.append(frame)
                self.assertEqual(writer.n_frames, i + 1)
            self.assertEqual(encoder.call_count, 2)
            self.assertEqual(zarr.open_group(str(path), mode="r")["depth"].shape[0], 8)
            writer.close()
            writer.close()
            self.assertEqual(encoder.call_count, 3)
            root = zarr.open_group(str(path), mode="r")
            decoded = root["depth"][:]
            self.assertEqual(decoded.shape, frames.shape)
            self.assertEqual(root.attrs["n_frames"], len(frames))
            self.assertLessEqual(float(np.abs(decoded.astype("f8") - frames).max()), 0.001)
            np.testing.assert_array_equal(decoded == 0, frames == 0)
            with self.assertRaises(RuntimeError):
                writer.append(frames[0])

    def test_empty_context_and_invalid_chunk_sizes(self):
        import zarr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.zarr"
            for chunk_frames in (0, -1, 1.5, True):
                with self.assertRaises(ValueError):
                    DepthZarrWriter(path, 2, 3, chunk_frames=chunk_frames)
            with DepthZarrWriter(path, 2, 3) as writer:
                self.assertEqual(writer.n_frames, 0)
            self.assertEqual(zarr.open_group(str(path), mode="r")["depth"].shape, (0, 2, 3))

    def test_append_copies_input_and_context_flushes_tail(self):
        import zarr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.zarr"
            frame = np.full((64, 80), 1.234, dtype="f4")
            with DepthZarrWriter(path, 64, 80, chunk_frames=4) as writer:
                writer.append(frame)
                frame[:] = 9.0
            np.testing.assert_allclose(
                zarr.open_group(str(path), mode="r")["depth"][0], 1.234, atol=0.001, rtol=0
            )

    def test_normalizes_invalid_depth_and_bounds_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.zarr"
            writer = DepthZarrWriter(path, height=2, width=3)
            writer.append(
                np.array(
                    [[1.2344, 1.2346, np.nan], [np.inf, -1.0, 65.4321]],
                    dtype=np.float32,
                )
            )
            writer.close()

            import zarr

            depth = zarr.open_group(str(path), mode="r")["depth"][0]
            np.testing.assert_allclose(
                depth,
                np.array([[1.2344, 1.2346, 0.0], [0.0, 0.0, 65.4321]], dtype=np.float32),
                atol=0.001, rtol=0,
            )
            np.testing.assert_array_equal(depth[[0, 1, 1], [2, 0, 1]], 0.0)

    def test_rejects_wrong_frame_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = DepthZarrWriter(Path(tmp) / "depth.zarr", height=2, width=3)
            with self.assertRaises(ValueError):
                writer.append(np.zeros((3, 2), dtype=np.float32))
            writer.close()


if __name__ == "__main__":
    unittest.main()
