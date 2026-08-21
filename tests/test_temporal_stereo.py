import sys
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAFT_ROOT = PROJECT_ROOT / "stereo_center" / "third_party" / "waft"
if str(WAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(WAFT_ROOT))

from stereo_center.temporal_stereo import (  # noqa: E402
    backward_warp,
    forward_backward_consistency_mask,
    temporal_alignment_mask,
)
from algorithms.waft import WAFT, temporal_warmstart_disparity  # noqa: E402
from stereo_center.waft_inference import (  # noqa: E402
    _pack_temporal_reference_pairs,
    run_stereo_matching_bi_batch,
)


class TemporalGeometryTests(unittest.TestCase):
    def test_backward_warp_uses_current_to_previous_flow(self):
        previous = torch.tensor([[[[10.0, 20.0, 30.0, 40.0]]]])
        current_to_previous = torch.zeros(1, 2, 1, 4)
        current_to_previous[:, 0] = 1.0

        warped, valid = backward_warp(previous, current_to_previous)

        torch.testing.assert_close(
            warped,
            torch.tensor([[[[20.0, 30.0, 40.0, 0.0]]]]),
        )
        self.assertEqual(valid.tolist(), [[[[True, True, True, False]]]])

    def test_forward_backward_consistency_rejects_mismatch_and_bounds(self):
        # Current -> previous is +1 everywhere. Previous -> current is -1 for
        # the corresponding pixels except at the third valid current pixel.
        backward = torch.zeros(1, 2, 1, 4)
        backward[:, 0] = 1.0
        forward = torch.zeros(1, 2, 1, 4)
        forward[:, 0] = torch.tensor([[-1.0, -1.0, -1.0, 2.0]])

        mask = forward_backward_consistency_mask(
            forward, backward, abs_tol=0.1, rel_tol=0.0
        )

        self.assertEqual(mask.tolist(), [[[[True, True, False, False]]]])

    def test_temporal_alignment_mask_rejects_photometric_changes(self):
        current = torch.tensor([[[[10.0, 100.0]], [[10.0, 100.0]], [[10.0, 100.0]]]])
        previous = torch.tensor([[[[10.0, 10.0]], [[10.0, 10.0]], [[10.0, 10.0]]]])
        forward = torch.zeros(1, 2, 1, 2)
        backward = torch.zeros(1, 2, 1, 2)

        mask = temporal_alignment_mask(
            current,
            previous,
            forward,
            backward,
            photo_tol=20.0,
            flow_abs_tol=0.1,
            flow_rel_tol=0.0,
        )

        torch.testing.assert_close(mask, torch.tensor([[[[1.0, 0.0]]]]))


class TemporalWarmstartTests(unittest.TestCase):
    def test_waft_forward_accepts_temporal_initialization(self):
        parameters = inspect.signature(WAFT.forward).parameters
        self.assertIn("temporal_init", parameters)

    def test_temporal_warmstart_keeps_groups_separate_and_uses_carry(self):
        current = torch.tensor([10.0, 20.0, 30.0, 40.0]).view(4, 1, 1, 1)
        flow = torch.zeros(4, 2, 1, 1)
        mask = torch.ones(4, 1, 1, 1)
        carry = torch.tensor([5.0, 25.0]).view(2, 1, 1)

        warm, effective = temporal_warmstart_disparity(
            current,
            flow,
            mask,
            group_size=2,
            carry_disparity=carry,
            blend=1.0,
            disparity_abs_tol=100.0,
            disparity_rel_tol=0.0,
        )

        torch.testing.assert_close(
            warm[:, 0, 0, 0], torch.tensor([5.0, 10.0, 25.0, 30.0])
        )
        torch.testing.assert_close(effective, torch.ones_like(effective))

    def test_temporal_warmstart_falls_back_on_disparity_disagreement(self):
        current = torch.tensor([10.0, 12.0, 40.0]).view(3, 1, 1, 1)
        flow = torch.zeros(3, 2, 1, 1)
        mask = torch.ones(3, 1, 1, 1)

        warm, effective = temporal_warmstart_disparity(
            current,
            flow,
            mask,
            group_size=3,
            carry_disparity=None,
            blend=0.75,
            disparity_abs_tol=3.0,
            disparity_rel_tol=0.0,
        )

        torch.testing.assert_close(
            warm[:, 0, 0, 0], torch.tensor([10.0, 10.5, 40.0])
        )
        torch.testing.assert_close(
            effective[:, 0, 0, 0], torch.tensor([0.0, 1.0, 0.0])
        )


class TemporalBatchPackingTests(unittest.TestCase):
    def _images(self):
        left = torch.tensor([1.0, 2.0]).view(2, 1, 1, 1).expand(-1, 3, 1, 2)
        right = torch.tensor([10.0, 20.0]).view(2, 1, 1, 1).expand(-1, 3, 1, 2)
        return left, right

    def test_first_batch_marks_each_temporal_group_boundary_invalid(self):
        current, previous, boundary = _pack_temporal_reference_pairs(
            *self._images(), temporal_state={}
        )

        torch.testing.assert_close(current[:, 0, 0, 0], torch.tensor([1.0, 2.0, 10.0, 20.0]))
        torch.testing.assert_close(previous[:, 0, 0, 0], torch.tensor([1.0, 1.0, 10.0, 10.0]))
        torch.testing.assert_close(boundary[:, 0, 0, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]))

    def test_later_batch_uses_left_and_right_carry_images(self):
        state = {
            "previous_reference_images": torch.tensor([0.0, 9.0])
            .view(2, 1, 1, 1)
            .expand(-1, 3, 1, 2)
        }

        _current, previous, boundary = _pack_temporal_reference_pairs(
            *self._images(), temporal_state=state
        )

        torch.testing.assert_close(previous[:, 0, 0, 0], torch.tensor([0.0, 1.0, 9.0, 10.0]))
        torch.testing.assert_close(boundary, torch.ones_like(boundary))

    def test_bidirectional_batch_api_accepts_temporal_flow_model(self):
        parameters = inspect.signature(run_stereo_matching_bi_batch).parameters
        self.assertIn("temporal_flow_model", parameters)
        self.assertIn("temporal_state", parameters)

    def test_batch_integration_updates_carry_in_one_bidirectional_forward(self):
        class FakeWAFT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.last_temporal_init = None

            def forward(self, sample, temporal_init=None, timing=None):
                self.calls += 1
                self.last_temporal_init = temporal_init
                n, _c, h, w = sample["img1"].shape
                disparity = torch.arange(1, n + 1, dtype=torch.float32).view(n, 1, 1)
                disparity = disparity.expand(n, h, w)
                return {
                    "disp_pred": disparity,
                    "delta_info_preds": [torch.zeros(n, 2, h, w)],
                    "temporal_valid_mask": temporal_init["mask"],
                }

        model = FakeWAFT()
        left = torch.zeros(2, 3, 2, 4)
        right = torch.zeros_like(left)
        state = {}
        timing = {}

        def zero_flow(_model, img0, img1, iters):
            self.assertEqual(iters, 7)
            return torch.zeros(img0.shape[0], 2, *img0.shape[-2:])

        with patch("stereo_center.raft_flow.flow_between", side_effect=zero_flow):
            result = run_stereo_matching_bi_batch(
                model,
                left,
                right,
                device="cpu",
                use_amp=False,
                hiera="direct",
                conf_mode="ones",
                occ_mode="visibility",
                timing_out=timing,
                temporal_flow_model=torch.nn.Identity(),
                temporal_state=state,
                temporal_flow_iters=7,
                temporal_disp_abs_tol=100.0,
            )

        self.assertEqual(model.calls, 1)
        self.assertEqual(result[0].shape, (2, 2, 4))
        self.assertEqual(model.last_temporal_init["group_size"], 2)
        self.assertEqual(state["previous_reference_images"].shape, (2, 3, 2, 4))
        torch.testing.assert_close(
            state["previous_disparities"][:, 0, 0], torch.tensor([2.0, 4.0])
        )
        self.assertIn("temporal_flow", timing)
        self.assertAlmostEqual(timing["temporal_valid_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
