import unittest
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from test_spcra_k4_runtime import load_contract_module


class BatchDict(TypedDict):
    batch_size: int
    gt_boxes: NDArray[np.float64]
    tta_pseudo_weights: NDArray[np.float64]
    tta_pseudo_reg_weights: NDArray[np.float64]


class TestK4Velocity(unittest.TestCase):
    def setUp(self):
        self.bridge = load_contract_module('spcra_k4_training')
        self.thresholds = {'NEG_THRESH': [.1] * 10, 'SCORE_THRESH': [.5] * 10}

    def prediction(self, boxes, *, scores=None, accepted=None):
        size = len(boxes)
        return {
            'pred_boxes': np.asarray(boxes, dtype=np.float64),
            'pred_labels': np.arange(1, size + 1),
            'pred_scores': np.full(size, .9) if scores is None else np.asarray(scores),
            'spcra_reliability': np.linspace(.2, .8, size),
            'spcra_k4_accepted': np.ones(size, dtype=bool) if accepted is None else np.asarray(accepted),
        }

    def batch(self, batch_size: int) -> BatchDict:
        return {
            'batch_size': batch_size,
            'gt_boxes': np.empty((0, 0, 10)),
            'tta_pseudo_weights': np.empty((0, 0)),
            'tta_pseudo_reg_weights': np.empty((0, 0)),
        }

    def test_nonzero_asymmetric_velocity_stays_aligned_through_filter_and_injection(self):
        # Given interleaved keep, ignore, removal, and invalid-velocity rows.
        prediction = self.prediction(
            [[10., 0., 0., 2., 2., 2., 0., 1.5, -2.5],
             [20., 0., 0., 2., 2., 2., 0., -3.25, 4.75],
             [30., 0., 0., 2., 2., 2., 0., 8., 9.],
             [40., 0., 0., 2., 2., 2., 0., -6., -7.]],
            scores=[.9, .4, .05, .9], accepted=[True, False, False, True],
        )
        prediction['pred_boxes'][3, 7] = np.nan

        # When the formal pseudo bridge filters and injects the frame.
        info = self.bridge.build_k4_pseudo_info(prediction, self.thresholds)
        batch: BatchDict = self.batch(1)
        self.bridge.inject_k4_pseudo_labels(batch, [info])

        # Then geometry, reliability, and velocity share both row masks.
        self.assertEqual(info['gt_boxes'].shape, (2, 9))
        np.testing.assert_array_equal(info['gt_boxes'][:, 0], [10., 20.])
        np.testing.assert_array_equal(info['velocity_xy'], [[1.5, -2.5], [-3.25, 4.75]])
        np.testing.assert_array_equal(info['reliability_weights'], [.2, .4])
        np.testing.assert_array_equal(batch['gt_boxes'][0, 0],
                                      [10., 0., 0., 2., 2., 2., 0., 1.5, -2.5, 1.])
        np.testing.assert_array_equal(batch['tta_pseudo_weights'], [[.2]])
        np.testing.assert_array_equal(batch['tta_pseudo_reg_weights'], [[.2]])

    def test_each_nonfinite_detector_velocity_invalidates_its_row(self):
        # Given accepted detector rows with one non-finite velocity component each.
        prediction = self.prediction(
            [[10., 0., 0., 2., 2., 2., 0., np.nan, 1.],
             [20., 0., 0., 2., 2., 2., 0., 1., np.inf]],
        )

        # When detector output crosses the pseudo boundary.
        info = self.bridge.build_k4_pseudo_info(prediction, self.thresholds)

        # Then invalid rows are removed from every persisted array.
        self.assertEqual(info['gt_boxes'].shape, (0, 9))
        self.assertEqual(info['velocity_xy'].shape, (0, 2))
        self.assertEqual(info['reliability_weights'].shape, (0,))

    def test_seven_column_detector_boxes_get_explicit_zero_velocity(self):
        # Given a valid detector row without velocity columns.
        prediction = self.prediction([[10., 0., 0., 2., 2., 2., 0.]])

        # When the row is persisted and injected.
        info = self.bridge.build_k4_pseudo_info(prediction, self.thresholds)
        batch: BatchDict = self.batch(1)
        self.bridge.inject_k4_pseudo_labels(batch, [info])

        # Then the explicit sidecar and training target both carry zero velocity.
        np.testing.assert_array_equal(info['velocity_xy'], [[0., 0.]])
        np.testing.assert_array_equal(batch['gt_boxes'][0, 0, 7:9], [0., 0.])

    def test_empty_frame_retains_formal_sidecar_shapes(self):
        # Given a detector frame with no rows.
        prediction = self.prediction(np.empty((0, 9)))

        # When the empty pseudo record is persisted and injected.
        info = self.bridge.build_k4_pseudo_info(prediction, self.thresholds)
        batch: BatchDict = self.batch(1)
        self.bridge.inject_k4_pseudo_labels(batch, [info])

        # Then all formal schemas retain their empty dimensions.
        self.assertEqual(info['gt_boxes'].shape, (0, 9))
        self.assertEqual(info['velocity_xy'].shape, (0, 2))
        self.assertEqual(batch['gt_boxes'].shape, (1, 0, 10))

    def test_unequal_batch_rows_preserve_per_frame_velocity_and_padding(self):
        # Given frames with two, zero, and one positive pseudo rows.
        first = self.bridge.build_k4_pseudo_info(self.prediction(
            [[10., 0., 0., 2., 2., 2., 0., 1., -1.],
             [20., 0., 0., 2., 2., 2., 0., 2., -2.]]), self.thresholds)
        empty = self.bridge.build_k4_pseudo_info(self.prediction(np.empty((0, 7))), self.thresholds)
        last = self.bridge.build_k4_pseudo_info(self.prediction(
            [[30., 0., 0., 2., 2., 2., 0., 3., -3.]]), self.thresholds)
        batch: BatchDict = self.batch(3)

        # When the ragged records are injected as one padded batch.
        self.bridge.inject_k4_pseudo_labels(batch, [first, empty, last])

        # Then real velocities stay frame-aligned and padding remains zero.
        self.assertEqual(batch['gt_boxes'].shape, (3, 2, 10))
        np.testing.assert_array_equal(batch['gt_boxes'][0, :, 7:9], [[1., -1.], [2., -2.]])
        np.testing.assert_array_equal(batch['gt_boxes'][1], np.zeros((2, 10)))
        np.testing.assert_array_equal(batch['gt_boxes'][2, :, 7:9], [[3., -3.], [0., 0.]])
        np.testing.assert_array_equal(batch['tta_pseudo_weights'][1], [0., 0.])

    def test_missing_or_malformed_persisted_velocity_sidecar_fails(self):
        # Given a persisted pseudo record and malformed velocity variants.
        info = self.bridge.build_k4_pseudo_info(self.prediction(
            [[10., 0., 0., 2., 2., 2., 0., 1., -1.]]), self.thresholds)
        info['velocity_xy'] = np.array([[1., -1.]])
        malformed = (
            None,
            np.array([1., -1.]),
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 3)),
            np.array([[np.nan, 0.]]),
            np.array([[0., np.inf]]),
            np.array([[1. + 0.j, 2. + 0.j]]),
        )
        for velocity in malformed:
            with self.subTest(velocity=velocity):
                corrupted = dict(info)
                if velocity is None:
                    del corrupted['velocity_xy']
                else:
                    corrupted['velocity_xy'] = velocity
                # When injection parses the sidecar, then corruption is rejected.
                with self.assertRaises(ValueError):
                    self.bridge.inject_k4_pseudo_labels({'batch_size': 1}, [corrupted])


if __name__ == '__main__':
    unittest.main()
