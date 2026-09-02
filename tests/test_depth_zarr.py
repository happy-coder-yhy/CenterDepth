import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "stereo_center"
import sys

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stereo_center.depth_zarr import DepthZarrWriter


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
            np.testing.assert_allclose(depth[0], 1.5)
            np.testing.assert_array_equal(depth[1], np.arange(6, dtype=np.float32).reshape(2, 3))
            self.assertEqual(root.attrs["n_frames"], 2)
            self.assertEqual(root.attrs["height"], 2)
            self.assertEqual(root.attrs["width"], 3)
            self.assertEqual(root.attrs["backend"], "ffs")

    def test_rejects_wrong_frame_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = DepthZarrWriter(Path(tmp) / "depth.zarr", height=2, width=3)
            with self.assertRaises(ValueError):
                writer.append(np.zeros((3, 2), dtype=np.float32))
            writer.close()


if __name__ == "__main__":
    unittest.main()
