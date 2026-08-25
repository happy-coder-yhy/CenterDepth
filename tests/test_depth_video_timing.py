import unittest
import argparse
import importlib.util
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "stereo_center" / "scripts" / "run_depth_video.py"
spec = importlib.util.spec_from_file_location("run_depth_video", SCRIPT_PATH)
run_depth_video = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_depth_video)


class DepthVideoTimingTests(unittest.TestCase):
    def test_timing_artifact_name_uses_backend(self):
        self.assertEqual(run_depth_video.timing_artifact_name("waft"), "waft_timing.json")
        self.assertEqual(run_depth_video.timing_artifact_name("ffs"), "ffs_timing.json")
        self.assertEqual(run_depth_video.timing_artifact_name("las2"), "las2_timing.json")
        self.assertEqual(
            run_depth_video.timing_artifact_name("opencv_bm"), "opencv_bm_timing.json"
        )
        self.assertEqual(
            run_depth_video.timing_artifact_name("opencv_sgbm"), "opencv_sgbm_timing.json"
        )

    def test_opencv_bm_needs_no_weights_and_exposes_classical_parameters(self):
        self.assertEqual(run_depth_video.resolve_weights_dir(None, "opencv_bm"), Path("."))
        parser = argparse.ArgumentParser()
        run_depth_video.add_opencv_bm_arguments(parser)

        args = parser.parse_args([])

        self.assertEqual(args.bm_num_disparities, 128)
        self.assertEqual(args.bm_block_size, 15)
        self.assertEqual(args.bm_uniqueness_ratio, 10)
        self.assertEqual(args.bm_speckle_window_size, 100)
        self.assertEqual(args.bm_speckle_range, 2)
        self.assertEqual(args.bm_disp12_max_diff, 1)

    def test_opencv_bm_is_limited_to_left_single_direction_output(self):
        valid = SimpleNamespace(
            stereo_backend="opencv_bm", output_view="left", bi=0
        )
        run_depth_video.validate_backend_mode(valid)

        with self.assertRaisesRegex(ValueError, "left.*bi=0"):
            run_depth_video.validate_backend_mode(
                SimpleNamespace(stereo_backend="opencv_bm", output_view="center", bi=0)
            )
        with self.assertRaisesRegex(ValueError, "left.*bi=0"):
            run_depth_video.validate_backend_mode(
                SimpleNamespace(stereo_backend="opencv_bm", output_view="left", bi=1)
            )

    def test_opencv_sgbm_needs_no_weights_and_exposes_native_parameters(self):
        self.assertEqual(run_depth_video.resolve_weights_dir(None, "opencv_sgbm"), Path("."))
        parser = argparse.ArgumentParser()
        run_depth_video.add_opencv_sgbm_arguments(parser)

        args = parser.parse_args([])

        self.assertEqual(args.sgbm_min_disparity, 0)
        self.assertEqual(args.sgbm_num_disparities, 128)
        self.assertEqual(args.sgbm_block_size, 5)
        self.assertIsNone(args.sgbm_p1)
        self.assertIsNone(args.sgbm_p2)
        self.assertEqual(args.sgbm_mode, "3way")

    def test_opencv_sgbm_is_limited_to_left_single_direction_output(self):
        valid = SimpleNamespace(
            stereo_backend="opencv_sgbm", output_view="left", bi=0
        )
        run_depth_video.validate_backend_mode(valid)

        with self.assertRaisesRegex(ValueError, "left.*bi=0"):
            run_depth_video.validate_backend_mode(
                SimpleNamespace(stereo_backend="opencv_sgbm", output_view="center", bi=0)
            )
        with self.assertRaisesRegex(ValueError, "left.*bi=0"):
            run_depth_video.validate_backend_mode(
                SimpleNamespace(stereo_backend="opencv_sgbm", output_view="left", bi=1)
            )

    def test_opencv_bm_timing_metadata_records_reproducible_parameters(self):
        args = SimpleNamespace(
            bm_num_disparities=128,
            bm_block_size=15,
            bm_uniqueness_ratio=10,
            bm_speckle_window_size=100,
            bm_speckle_range=2,
            bm_disp12_max_diff=1,
        )

        self.assertEqual(
            run_depth_video.opencv_bm_parameters(args),
            {
                "num_disparities": 128,
                "block_size": 15,
                "uniqueness_ratio": 10,
                "speckle_window_size": 100,
                "speckle_range": 2,
                "disp12_max_diff": 1,
            },
        )

    def test_opencv_sgbm_timing_metadata_records_reproducible_parameters(self):
        args = SimpleNamespace(
            sgbm_min_disparity=0,
            sgbm_num_disparities=128,
            sgbm_block_size=5,
            sgbm_p1=None,
            sgbm_p2=None,
            sgbm_disp12_max_diff=1,
            sgbm_uniqueness_ratio=10,
            sgbm_speckle_window_size=100,
            sgbm_speckle_range=2,
            sgbm_mode="3way",
        )

        self.assertEqual(
            run_depth_video.opencv_sgbm_parameters(args),
            {
                "min_disparity": 0,
                "num_disparities": 128,
                "block_size": 5,
                "p1": 600,
                "p2": 2400,
                "disp12_max_diff": 1,
                "uniqueness_ratio": 10,
                "speckle_window_size": 100,
                "speckle_range": 2,
                "mode": "3way",
            },
        )

    def test_left_hole_fill_cli_defaults_and_metadata(self):
        parser = argparse.ArgumentParser()
        run_depth_video.add_left_hole_fill_arguments(parser)

        args = parser.parse_args([])

        self.assertEqual(args.left_hole_fill, 0)
        self.assertEqual(args.left_hole_fill_max_area, 256)
        self.assertEqual(args.left_hole_fill_color_tol, 20.0)
        self.assertEqual(
            run_depth_video.left_hole_fill_parameters(args),
            {
                "enabled": False,
                "max_area": 256,
                "color_tol": 20.0,
            },
        )

    def test_left_hole_fill_is_limited_to_left_view(self):
        run_depth_video.validate_left_hole_fill_mode(
            SimpleNamespace(left_hole_fill=1, output_view="left")
        )
        with self.assertRaisesRegex(ValueError, "left-view"):
            run_depth_video.validate_left_hole_fill_mode(
                SimpleNamespace(left_hole_fill=1, output_view="center")
            )

    def test_temporal_initialization_accepts_only_bi_waft_direct_mode(self):
        valid = SimpleNamespace(
            waft_temporal_init=1,
            stereo_backend="waft",
            bi=1,
            temporal_raft=0,
        )
        run_depth_video.validate_waft_temporal_mode(valid, "direct")

        invalid_cases = [
            SimpleNamespace(**{**vars(valid), "stereo_backend": "ffs"}),
            SimpleNamespace(**{**vars(valid), "bi": 0}),
            SimpleNamespace(**{**vars(valid), "temporal_raft": 1}),
        ]
        for invalid in invalid_cases:
            with self.subTest(invalid=vars(invalid)):
                with self.assertRaises(ValueError):
                    run_depth_video.validate_waft_temporal_mode(invalid, "direct")
        with self.assertRaises(ValueError):
            run_depth_video.validate_waft_temporal_mode(valid, "hiera")

    def test_waft_temporal_cli_arguments_have_conservative_defaults(self):
        parser = argparse.ArgumentParser()
        run_depth_video.add_waft_temporal_arguments(parser)

        args = parser.parse_args([])

        self.assertEqual(args.waft_temporal_init, 0)
        self.assertEqual(args.waft_temporal_flow_iters, 12)
        self.assertEqual(args.waft_temporal_blend, 0.75)
        self.assertEqual(args.waft_temporal_photo_tol, 40.0)
        self.assertEqual(args.waft_temporal_flow_abs_tol, 0.5)
        self.assertEqual(args.waft_temporal_flow_rel_tol, 0.01)
        self.assertEqual(args.waft_temporal_disp_abs_tol, 3.0)
        self.assertEqual(args.waft_temporal_disp_rel_tol, 0.15)

    def test_model_iteration_cli_uses_unified_iters(self):
        parser = argparse.ArgumentParser()
        run_depth_video.add_model_iteration_arguments(parser)

        self.assertIsNone(parser.parse_args([]).iters)
        self.assertEqual(parser.parse_args(["--iters", "5"]).iters, 5)
        self.assertEqual(parser.parse_args(["--waft-iters", "4"]).iters, 4)
        self.assertEqual(parser.parse_args(["--ffs-valid-iters", "8"]).iters, 8)

        help_text = parser.format_help()
        self.assertIn("--iters", help_text)
        self.assertNotIn("--waft-iters", help_text)
        self.assertNotIn("--ffs-valid-iters", help_text)

    def test_model_iteration_defaults_preserve_backend_behavior(self):
        self.assertIsNone(run_depth_video.resolve_model_iters("waft", None))
        self.assertEqual(run_depth_video.resolve_model_iters("ffs", None), 8)
        self.assertEqual(run_depth_video.resolve_model_iters("waft", 5), 5)
        self.assertEqual(run_depth_video.resolve_model_iters("ffs", 6), 6)

    def test_processing_end_is_clamped_to_available_pts_pairs(self):
        self.assertEqual(run_depth_video.resolve_processing_end(973, 0, -1, 0), 973)
        self.assertEqual(run_depth_video.resolve_processing_end(973, 968, 983, 0), 973)
        self.assertEqual(run_depth_video.resolve_processing_end(973, 968, -1, 3), 971)

    def test_waft_temporal_kwargs_are_disabled_without_mode(self):
        args = SimpleNamespace(
            waft_temporal_init=0,
            waft_temporal_flow_iters=12,
            waft_temporal_blend=0.75,
            waft_temporal_photo_tol=40.0,
            waft_temporal_flow_abs_tol=0.5,
            waft_temporal_flow_rel_tol=0.01,
            waft_temporal_disp_abs_tol=3.0,
            waft_temporal_disp_rel_tol=0.15,
        )

        self.assertEqual(run_depth_video.waft_temporal_kwargs(args, object(), {}), {})

        args.waft_temporal_init = 1
        flow_model = object()
        state = {}
        kwargs = run_depth_video.waft_temporal_kwargs(args, flow_model, state)
        self.assertIs(kwargs["temporal_flow_model"], flow_model)
        self.assertIs(kwargs["temporal_state"], state)
        self.assertEqual(kwargs["temporal_flow_iters"], 12)
        self.assertEqual(kwargs["temporal_disp_rel_tol"], 0.15)

    def test_temporal_valid_ratio_is_weighted_by_batch_frames(self):
        records = [
            {"batch_size": 8, "temporal_valid_ratio": 0.5},
            {"batch_size": 2, "temporal_valid_ratio": 1.0},
        ]

        ratio = run_depth_video.weighted_temporal_valid_ratio(records)

        self.assertAlmostEqual(ratio, 0.6)


if __name__ == "__main__":
    unittest.main()
