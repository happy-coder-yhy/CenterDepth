import unittest

import torch

from stereo_center.temporal_depth import fuse_previous_depth


class TemporalDepthFusionTests(unittest.TestCase):
    def _rgb(self, values):
        return torch.tensor(values, dtype=torch.float32).view(1, 1, 1, -1).expand(-1, 3, -1, -1)

    def test_zero_flow_fuses_inverse_depth_without_recursion(self):
        current = torch.tensor([[[[4.0, 4.0]]]])
        previous = torch.tensor([[[[2.0, 2.0]]]])
        flow = torch.zeros(1, 2, 1, 2)

        result = fuse_previous_depth(
            current, previous, self._rgb([10.0, 10.0]), self._rgb([10.0, 10.0]), flow, flow,
            max_prior_weight=0.2, depth_abs_tol=10.0, depth_rel_tol=0.0,
        )

        torch.testing.assert_close(result.depth, torch.full_like(current, 10.0 / 3.0))
        torch.testing.assert_close(result.weight, torch.full_like(current, 0.2))
        self.assertTrue(result.accepted.all())

    def test_current_to_previous_direction_is_used_for_depth_sampling(self):
        current = torch.tensor([[[[20.0, 30.0, 40.0, 50.0]]]])
        previous = torch.tensor([[[[10.0, 20.0, 30.0, 40.0]]]])
        current_to_previous = torch.zeros(1, 2, 1, 4)
        current_to_previous[:, 0] = 1.0
        previous_to_current = torch.zeros_like(current_to_previous)
        previous_to_current[:, 0] = -1.0

        result = fuse_previous_depth(
            current, previous, self._rgb([20, 30, 40, 50]), self._rgb([10, 20, 30, 40]),
            previous_to_current, current_to_previous,
            max_prior_weight=1.0, depth_abs_tol=100.0, depth_rel_tol=0.0,
            photo_tol=1.0,
        )

        torch.testing.assert_close(result.depth[..., :3], torch.tensor([[[[20.0, 30.0, 40.0]]]]))
        self.assertFalse(result.accepted[..., 3].item())

    def test_invalid_history_does_not_contribute_to_bilinear_sample(self):
        current = torch.tensor([[[[4.0, 4.0]]]])
        previous = torch.tensor([[[[2.0, 0.0]]]])
        flow = torch.zeros(1, 2, 1, 2)

        result = fuse_previous_depth(
            current, previous, self._rgb([10.0, 10.0]), self._rgb([10.0, 10.0]), flow, flow,
            max_prior_weight=0.2, depth_abs_tol=10.0, depth_rel_tol=0.0,
        )

        self.assertTrue(result.accepted[..., 0].item())
        self.assertFalse(result.accepted[..., 1].item())
        self.assertEqual(result.depth[..., 1].item(), 4.0)

    def test_depth_disagreement_rejects_history(self):
        current = torch.tensor([[[[10.0]]]])
        previous = torch.tensor([[[[2.0]]]])
        flow = torch.zeros(1, 2, 1, 1)

        result = fuse_previous_depth(
            current, previous, self._rgb([10.0]), self._rgb([10.0]), flow, flow,
            max_prior_weight=0.2, depth_abs_tol=0.1, depth_rel_tol=0.0,
        )

        self.assertFalse(result.accepted.item())
        self.assertEqual(result.weight.item(), 0.0)
        self.assertEqual(result.depth.item(), 10.0)

    def test_depth_above_experiment_range_does_not_contribute(self):
        current = torch.tensor([[[[25.0]]]])
        previous = torch.tensor([[[[25.0]]]])
        flow = torch.zeros(1, 2, 1, 1)

        result = fuse_previous_depth(
            current, previous, self._rgb([10.0]), self._rgb([10.0]), flow, flow,
            max_prior_weight=0.2, max_depth=20.0,
        )

        self.assertFalse(result.accepted.item())
        self.assertEqual(result.depth.item(), 25.0)


if __name__ == "__main__":
    unittest.main()
