import unittest

import numpy as np
import torch

from test_spcra_k4_runtime import load_contract_module


class TestK4AdapterContext(unittest.TestCase):
    def setUp(self):
        self.context = load_contract_module('spcra_k4_context')
        self.density = self.context.DensityMapSpec(
            point_cloud_range=(0., 0., -1., 4., 4., 1.), grid_size=4,
        )

    def test_per_view_density_uses_the_installed_points_without_cross_frame_leakage(self):
        # Given two frames whose retained view points occupy different cells.
        proposals = self.context.ReferenceProposalContext.empty(batch_size=2)
        first_points = torch.tensor([
            [0., .25, .25, 0.], [0., .30, .30, 0.], [1., 3.25, 3.25, 0.],
        ])
        second_points = torch.tensor([
            [0., 2.25, .25, 0.], [1., 1.25, 3.25, 0.], [1., 1.30, 3.30, 0.],
        ])
        deltas = torch.eye(4).repeat(2, 1, 1)

        # When each view rebuilds formal adapter inputs from its actual points.
        first = proposals.for_view(first_points, deltas, self.density)
        second = proposals.for_view(second_points, deltas, self.density)

        # Then both the view and frame dimensions remain independent.
        self.assertFalse(torch.equal(first.density_map, second.density_map))
        self.assertEqual(torch.count_nonzero(first.density_map[0]).item(), 1)
        self.assertEqual(torch.count_nonzero(first.density_map[1]).item(), 1)
        self.assertEqual(torch.count_nonzero(second.density_map[0]).item(), 1)
        self.assertEqual(torch.count_nonzero(second.density_map[1]).item(), 1)
        self.assertGreater(first.density_map[0, 0, 0, 0].item(), 0.)
        self.assertEqual(first.density_map[1, 0, 0, 0].item(), 0.)

    def test_view_delta_transforms_geometry_without_reordering_proposal_fields(self):
        # Given two proposal rows with stable class, score, row, and mask identity.
        boxes = torch.tensor([[[1., 0., 0., 2., 1., 1., 0., 3., .8],
                               [2., 1., 0., 1., 1., 1., .5, 7., .4]]])
        mask = torch.tensor([[True, True]])
        proposals = self.context.ReferenceProposalContext.from_inputs(boxes, mask, batch_size=1)
        delta = torch.tensor([[[0., -1., 0., 4.], [1., 0., 0., -2.],
                               [0., 0., 1., 1.], [0., 0., 0., 1.]]])
        points = torch.tensor([[0., 1., 1., 0.]])

        # When the immutable reference context is projected into the view.
        actual = proposals.for_view(points, delta, self.density)

        # Then only geometry changes and every non-geometric field stays row-aligned.
        expected_geometry = torch.tensor([[4., -1., 1., 2., 1., 1., np.pi / 2],
                                          [3., 0., 1., 1., 1., 1., .5 + np.pi / 2]])
        torch.testing.assert_close(actual.proposal_boxes[0, :, :7], expected_geometry)
        torch.testing.assert_close(actual.proposal_boxes[0, :, 7:], boxes[0, :, 7:])
        torch.testing.assert_close(actual.proposal_mask, mask)
        np.testing.assert_array_equal(proposals.boxes[0][:, 0], [1., 2.])

    def test_reference_predictions_are_snapshotted_as_immutable_adapter_rows(self):
        # Given current reference predictions with detector geometry, class, and score.
        prediction = {
            'pred_boxes': torch.tensor([[1., 2., 0., 1., 1., 1., .25],
                                        [3., 4., 0., 2., 1., 1., -.5]]),
            'pred_labels': torch.tensor([2, 9]),
            'pred_scores': torch.tensor([.8, .3]),
        }

        # When the formal proposal context is constructed once from the reference.
        proposals = self.context.ReferenceProposalContext.from_predictions([prediction])
        prediction['pred_boxes'][0, 0] = 99.

        # Then rows are detached, normalized to the adapter contract, and read-only.
        self.assertEqual(proposals.boxes[0].shape, (2, 9))
        np.testing.assert_array_equal(proposals.boxes[0][:, 7], [2., 9.])
        np.testing.assert_allclose(proposals.boxes[0][:, 8], [.8, .3])
        self.assertEqual(proposals.boxes[0][0, 0], 1.)
        with self.assertRaises(ValueError):
            proposals.boxes[0][0, 0] = -1.

    def test_missing_proposals_produce_explicit_empty_batched_inputs(self):
        # Given no proposal source for either frame.
        proposals = self.context.ReferenceProposalContext.from_inputs(
            proposal_boxes=None, proposal_mask=None, batch_size=2,
        )

        # When the adapter inputs are materialized.
        actual = proposals.for_view(
            torch.empty((0, 4)), torch.eye(4).repeat(2, 1, 1), self.density,
        )

        # Then absence is represented explicitly rather than by missing keys or None.
        self.assertEqual(actual.proposal_boxes.shape, (2, 0, 9))
        self.assertEqual(actual.proposal_mask.shape, (2, 0))
        self.assertEqual(actual.proposal_mask.dtype, torch.bool)

    def test_ragged_batch_padding_is_zero_and_inactive(self):
        # Given unequal per-frame proposal counts.
        proposals = self.context.ReferenceProposalContext.from_inputs(
            proposal_boxes=(
                np.array([[1., 0., 0., 1., 1., 1., 0., 1., .9],
                          [2., 0., 0., 1., 1., 1., 0., 2., .8]], dtype=np.float32),
                np.array([[10., 0., 0., 1., 1., 1., 0., 9., .7]], dtype=np.float32),
            ),
            proposal_mask=(np.array([True, True]), np.array([True])),
            batch_size=2,
        )
        deltas = torch.eye(4).repeat(2, 1, 1)
        deltas[0, 0, 3] = 5.
        deltas[1, 0, 3] = -3.

        # When both frames are materialized into one padded adapter batch.
        actual = proposals.for_view(torch.empty((0, 4)), deltas, self.density)

        # Then each frame uses only its own delta and padding cannot become active.
        torch.testing.assert_close(actual.proposal_boxes[0, :, 0], torch.tensor([6., 7.]))
        torch.testing.assert_close(actual.proposal_boxes[1, :, 0], torch.tensor([7., 0.]))
        torch.testing.assert_close(actual.proposal_mask, torch.tensor([[True, True], [True, False]]))
        torch.testing.assert_close(actual.proposal_boxes[1, 1], torch.zeros(9))


if __name__ == '__main__':
    unittest.main()
