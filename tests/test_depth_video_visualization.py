import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
sys.path.insert(0, str(PACKAGE_ROOT))

from stereo_center.depth_video_visualization import (
    DepthVisualizationCache,
    render_adaptive_frames,
    sequence_depth_scale,
)
from stereo_center.visualize import colorize_depth_log


class DepthVisualizationTests(unittest.TestCase):
    def test_exact_percentile_matches_numpy_across_float_exponents(self):
        rng = np.random.default_rng(1038)
        values = np.exp(rng.uniform(-30, 60, 10001)).astype(np.float32)
        values = np.r_[values, np.nextafter(np.float32(0), np.float32(1)),
                       np.float32(1), np.float32(1.000001)]
        pairs = [(part, np.ones(part.shape, bool)) for part in np.array_split(values, 11)]
        for percentile in [.1, 50, 99.9, 100]:
            with self.subTest(percentile=percentile):
                result = sequence_depth_scale(lambda: iter(pairs), percentile=percentile)
                np.testing.assert_allclose(result["percentile_depth_m"],
                                           np.percentile(values, percentile), rtol=1e-14)
                self.assertEqual(result["valid_pixels"], values.size)

    def test_masks_nonfinite_zero_and_negative_are_excluded_without_mutation(self):
        depth = np.array([[0, -2, np.nan, np.inf], [.5, 2.91, 10000, 1]], np.float32)
        original = depth.copy()
        valid = np.ones(depth.shape, bool)
        valid[1, 2] = False
        metadata = sequence_depth_scale(lambda: iter([(depth, valid)]))
        self.assertEqual(metadata["valid_pixels"], 3)
        self.assertAlmostEqual(metadata["percentile_depth_m"], np.percentile([.5, float(depth[1, 1]), 1], 99.9))
        self.assertEqual(metadata["dmax_m"], 3)
        np.testing.assert_array_equal(depth, original)

    def test_pixel_weighted_percentile_not_average_of_frame_percentiles(self):
        pairs = [(np.full((1, 10000), 2, np.float32), np.ones((1, 10000), bool)),
                 (np.array([[50]], np.float32), np.array([[True]]))]
        self.assertEqual(sequence_depth_scale(lambda: iter(pairs))["percentile_depth_m"], 2)

    def test_sparse_extreme_outliers_do_not_dominate_range(self):
        depth = np.full((100, 100), 2.91, np.float32)
        depth[0, 0] = 15554015
        metadata = sequence_depth_scale(lambda: iter([(depth, np.ones(depth.shape, bool))]))
        self.assertEqual(metadata["dmax_m"], 3)

    def test_far_scene_and_rounding_boundary(self):
        for depth_m, expected in [(3., 3.), (3.001, 3.5), (9.7, 10.), (12.2, 12.5), (.1, 1.)]:
            depth = np.full((2, 2), depth_m, np.float32)
            scale = sequence_depth_scale(lambda: iter([(depth, np.ones(depth.shape, bool))]))
            self.assertEqual(scale["dmax_m"], expected)

    def test_twenty_meter_ceiling_preserves_actual_percentile(self):
        depth = np.full((20, 20), 451.04, np.float32)
        original = depth.copy()
        rendered = []
        metadata = render_adaptive_frames(
            lambda: iter([(depth, np.ones_like(depth, bool))]),
            lambda i, image: rendered.append(image),
        )
        self.assertEqual(metadata["dmax_m"], 20)
        self.assertEqual(metadata["uncapped_dmax_m"], 451.5)
        self.assertEqual(metadata["maximum_range_m"], 20)
        self.assertTrue(metadata["range_capped"])
        self.assertEqual(metadata["percentile_depth_m"], float(depth[0, 0]))
        self.assertEqual(metadata["saturated_pixels"], depth.size)
        np.testing.assert_array_equal(depth, original)
        np.testing.assert_array_equal(rendered[0], colorize_depth_log(depth, np.ones_like(depth, bool), d_max=20))

    def test_cap_is_configurable_and_applied_after_rounding(self):
        depth = np.full((2, 2), 20.01, np.float32)
        source = lambda: iter([(depth, np.ones_like(depth, bool))])
        result = sequence_depth_scale(source, maximum_m=20.1)
        self.assertEqual(result["dmax_m"], 20.1)
        self.assertEqual(result["uncapped_dmax_m"], 20.5)
        self.assertTrue(result["range_capped"])
        depth[:] = 3
        result = sequence_depth_scale(source)
        self.assertEqual(result["dmax_m"], 3)
        self.assertFalse(result["range_capped"])

    def test_all_invalid_uses_documented_fallback_and_renders_black(self):
        pair = (np.zeros((4, 4), np.float32), np.zeros((4, 4), bool))
        frames = []
        metadata = render_adaptive_frames(lambda: iter([pair]), lambda i, img: frames.append(img))
        self.assertIsNone(metadata["percentile_depth_m"])
        self.assertEqual(metadata["dmax_m"], 1)
        self.assertEqual(metadata["fallback"], "no_valid_positive_depth")
        self.assertFalse(frames[0].any())

    def test_sequence_uses_same_color_for_same_depth_and_counts_saturation(self):
        a = np.ones((20, 20), np.float32)
        b = np.full_like(a, 8)
        b[0, 0] = 1
        b[0, 1] = 1000
        # More than 1000 pixels are needed to exclude the one outlier at P99.9.
        pairs = [(a, np.ones_like(a, bool))] * 3 + [(b, np.ones_like(b, bool))]
        rendered = []
        result = render_adaptive_frames(lambda: iter(pairs), lambda i, img: rendered.append(img))
        self.assertEqual(result["dmax_m"], 8)
        self.assertEqual(result["saturated_pixels"], 1)
        np.testing.assert_array_equal(rendered[0][0, 0], rendered[-1][0, 0])
        for (d, v), img in zip(pairs, rendered):
            np.testing.assert_array_equal(img, colorize_depth_log(d, v, d_max=8))

    def test_log_mapping_matches_legacy_and_keeps_subminimum_depth_valid(self):
        depth = np.array([[.1, .3, 1, 20, 100, 0, np.nan]], np.float32)
        original = depth.copy()
        mask = np.isfinite(depth) & (depth > 0)
        rendered = []
        result = render_adaptive_frames(
            lambda: iter([(depth, mask)]), lambda i, img: rendered.append(img)
        )
        self.assertEqual(result["mapping"], "log")
        self.assertEqual(result["dmin_m"], .3)
        self.assertEqual(result["dmax_m"], 20)
        self.assertEqual(result["valid_pixels"], 5)
        self.assertEqual(result["below_minimum_pixels"], 1)
        self.assertEqual(result["below_minimum_fraction_of_valid"], .2)
        self.assertEqual(result["saturated_pixels"], 1)
        np.testing.assert_array_equal(rendered[0], colorize_depth_log(depth, mask))
        np.testing.assert_array_equal(rendered[0][0, 0], rendered[0][0, 1])
        np.testing.assert_array_equal(rendered[0][0, 3], rendered[0][0, 4])
        self.assertFalse(rendered[0][0, 5:].any())
        np.testing.assert_array_equal(depth, original)

    def test_empty_source_invalid_settings_and_changed_source_fail(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            sequence_depth_scale(lambda: iter([]))
        for kwargs in [{"percentile": 0}, {"percentile": float("nan")}, {"step_m": 0},
                       {"minimum_m": -1}, {"maximum_m": float("inf")},
                       {"maximum_m": .5}, {"maximum_m": float("nan")}]:
            with self.assertRaises(ValueError):
                sequence_depth_scale(lambda: iter([]), **kwargs)
        pair = (np.ones((2, 2), np.float32), np.ones((2, 2), bool))
        calls = iter([[pair], []])
        with self.assertRaisesRegex(ValueError, "changed"):
            sequence_depth_scale(lambda: iter(next(calls)))

    def test_cache_preserves_raw_float32_mask_indices_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            depth = np.array([[1.00003, 2], [0, np.nan]], np.float32)
            original = depth.copy()
            mask = np.array([[True, False], [False, False]])
            with DepthVisualizationCache(directory) as cache:
                path = cache.path
                cache.append(depth, mask, 123)
                depth[:] = 999
                for _ in range(3):
                    d, v = list(cache.frames())[0]
                    np.testing.assert_array_equal(d, original)
                    np.testing.assert_array_equal(v, mask)
                self.assertEqual(cache.frame_indices, [123])
            self.assertFalse(path.exists())

    def test_cache_cleans_up_on_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                with DepthVisualizationCache(directory) as cache:
                    path = cache.path
                    raise RuntimeError("test")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
