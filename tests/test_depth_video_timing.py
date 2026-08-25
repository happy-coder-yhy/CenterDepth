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
