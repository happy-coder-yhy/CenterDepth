from pathlib import Path
import sys
import unittest

import numpy as np
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stereo_center.left_hole_fill import fill_small_left_holes


class LeftHoleFillTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((12, 12), 2.0, dtype=np.float32)
        self.valid = np.ones((12, 12), dtype=bool)
        self.guide = np.full((12, 12, 3), 120, dtype=np.uint8)

    def test_fills_small_interior_hole_from_compatible_boundary_depth(self):
        depth = self.depth.copy()
        valid = self.valid.copy()
        valid[4:7, 4:7] = False

        filled_depth, filled_valid, stats = fill_small_left_holes(
            depth, valid, self.guide, max_area=16, color_tol=10
        )

        self.assertTrue(filled_valid[4:7, 4:7].all())
        self.assertTrue(np.allclose(filled_depth[4:7, 4:7], 2.0))
        self.assertEqual(stats["filled_components"], 1)
        self.assertEqual(stats["filled_pixels"], 9)
        self.assertFalse(valid[4:7, 4:7].any())

    def test_keeps_large_and_border_touching_holes_invalid(self):
        depth = self.depth.copy()
        valid = self.valid.copy()
        valid[0:2, 4:6] = False
        valid[4:8, 4:8] = False

        _depth, filled_valid, stats = fill_small_left_holes(
            depth, valid, self.guide, max_area=9, color_tol=10
        )

        self.assertFalse(filled_valid[0:2, 4:6].any())
        self.assertFalse(filled_valid[4:8, 4:8].any())
        self.assertEqual(stats["filled_components"], 0)
        self.assertEqual(stats["filled_pixels"], 0)

    def test_keeps_hole_invalid_when_boundary_color_is_incompatible(self):
        depth = self.depth.copy()
        valid = self.valid.copy()
        guide = self.guide.copy()
        valid[5, 5] = False
        guide[5, 5] = 0

        _depth, filled_valid, stats = fill_small_left_holes(
            depth, valid, guide, max_area=16, color_tol=10
        )

        self.assertFalse(filled_valid[5, 5])
        self.assertEqual(stats["filled_components"], 0)
        self.assertEqual(stats["filled_pixels"], 0)

    def test_computes_boundary_in_the_component_local_bounding_box(self):
        depth = self.depth.copy()
        valid = self.valid.copy()
        valid[4:7, 4:7] = False

        with mock.patch("stereo_center.left_hole_fill.cv2.dilate", wraps=__import__("cv2").dilate) as dilate:
            fill_small_left_holes(depth, valid, self.guide, max_area=16, color_tol=10)

        self.assertEqual(dilate.call_count, 1)
        self.assertEqual(dilate.call_args.args[0].shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
