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
        self.assertEqual(run_depth_video.timing_artifact_name("foundation"), "foundation_timing.json")

    def test_resolve_weights_dir_accepts_foundation_env_var(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("FOUNDATION_WEIGHTS_DIR")
            os.environ["FOUNDATION_WEIGHTS_DIR"] = tmp
            try:
                self.assertEqual(run_depth_video.resolve_weights_dir(None, "foundation"), Path(tmp))
            finally:
                if old is None:
                    os.environ.pop("FOUNDATION_WEIGHTS_DIR", None)
                else:
                    os.environ["FOUNDATION_WEIGHTS_DIR"] = old

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
