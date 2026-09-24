"""Formal training bridge contract, independent of detector imports.

build_k4_pseudo_info(prediction, thresholds) preserves original-order sidecars
through NEG removal and SCORE ignores, treating spcra_k4_accepted as authoritative
rescue eligibility. inject_k4_pseudo_labels(batch, infos) replaces annotations and
exports exact cls/reg weights. positive_query_weights(assigned_gt_inds, r,
code_size=10) maps one-based Hungarian assignments (0 background, -1 ignore).
These are composition helpers, not an optimizer or a new training loop.
"""

import unittest

import numpy as np

from test_spcra_k4_runtime import load_contract_module


class TestK4Training(unittest.TestCase):
    def setUp(self):
        self.bridge = load_contract_module('spcra_k4_training')
        boxes = np.tile([0., 0., 0., 2., 3., 4., .1, 91., 92.], (5, 1))
        boxes[:, 0] = [10., 20., 30., 40., 50.]
        self.prediction = {
            'pred_boxes': boxes, 'pred_labels': np.array([1, 10, 2, 7, 4]),
            'pred_scores': np.array([.01, .8, .2, .3, .9]),
            'spcra_reliability': np.array([.93, 0., .375, .62, 1.]),
            'spcra_k4_accepted': np.array([False, True, True, False, True]),
        }
        self.thresholds = {'NEG_THRESH': [.1]*10, 'SCORE_THRESH': [.5]*10}

    def pseudo_info(self):
        return self.bridge.build_k4_pseudo_info(self.prediction, self.thresholds)

    def test_identity_survives_when_neg_removes_and_score_ignores(self):
        """Given interleaved decisions; when filtered; then reliability follows identity."""
        info = self.pseudo_info()
        np.testing.assert_array_equal(info['gt_boxes'][:, 0], [20., 30., 40., 50.])
        np.testing.assert_array_equal(info['gt_boxes'][:, 7], [10, 2, -7, 4])
        np.testing.assert_array_equal(info['reliability_weights'], [0., .375, .62, 1.])
        np.testing.assert_array_equal(self.prediction['pred_labels'], [1, 10, 2, 7, 4])

    def test_explicit_box7_class_score_when_predictions_have_velocity(self):
        """Given nine-column detector boxes; when serialized; then use explicit schema."""
        info = self.pseudo_info()
        self.assertEqual(info['gt_boxes'].shape, (4, 9))
        np.testing.assert_array_equal(info['gt_boxes'][:, :7], self.prediction['pred_boxes'][1:, :7])
        np.testing.assert_array_equal(info['gt_boxes'][:, 8], [.8, .2, .3, .9])

    def test_rescued_proposal_remains_positive_when_injected(self):
        """Given accepted score .2 below .5; when injected; then retain it with exact r."""
        info = self.pseudo_info()
        batch = {'batch_size': 1, 'frame_id': ['sample'], 'gt_boxes': np.full((1, 1, 10), 999.)}
        self.bridge.inject_k4_pseudo_labels(batch, [info])
        np.testing.assert_array_equal(batch['gt_boxes'][0, :, 0], [20., 30., 50.])
        np.testing.assert_array_equal(batch['gt_boxes'][0, :, -1], [10, 2, 4])
        np.testing.assert_array_equal(batch['tta_pseudo_weights'], [[0., .375, 1.]])
        np.testing.assert_array_equal(batch['tta_pseudo_reg_weights'], [[0., .375, 1.]])

    def test_query_cls_and_reg_weights_when_assignment_reorders_pseudo(self):
        """Given reordered positives including zero r; when assigned; then weights equal r."""
        assignment = np.array([3, 0, 1, -1, 2, 1])
        cls, reg = self.bridge.positive_query_weights(assignment, np.array([0., .375, 1.]), code_size=10)
        np.testing.assert_array_equal(cls, [1., 1., 0., 0., .375, 0.])
        np.testing.assert_array_equal(reg, np.repeat(np.array([1., 0., 0., 0., .375, 0.])[:, None], 10, axis=1))

    def test_full_bridge_when_prediction_identity_reaches_positive_queries(self):
        """Given raw sidecars; when saved/injected/assigned; then exact r reaches losses."""
        batch = {'batch_size': 1, 'frame_id': ['sample']}
        info = self.pseudo_info()
        self.bridge.inject_k4_pseudo_labels(batch, [info])
        cls, reg = self.bridge.positive_query_weights(
            np.array([2, 3, 1]), batch['tta_pseudo_weights'][0], code_size=10,
        )
        np.testing.assert_array_equal(cls, [.375, 1., 0.])
        np.testing.assert_array_equal(reg[:, 0], [.375, 1., 0.])

    def test_empty_pseudo_replaces_annotations_when_no_proposals_survive(self):
        """Given stale annotations/weights; when pseudo is empty; then replace all three."""
        for key in self.prediction:
            self.prediction[key] = self.prediction[key][:0]
        batch = {'batch_size': 1, 'frame_id': ['sample'],
                 'gt_boxes': np.full((1, 2, 10), 999.),
                 'tta_pseudo_weights': np.ones((1, 2)),
                 'tta_pseudo_reg_weights': np.ones((1, 2))}
        self.bridge.inject_k4_pseudo_labels(batch, [self.pseudo_info()])
        self.assertEqual(batch['gt_boxes'].shape, (1, 0, 10))
        self.assertEqual(batch['tta_pseudo_weights'].shape, (1, 0))
        self.assertEqual(batch['tta_pseudo_reg_weights'].shape, (1, 0))

    def test_one_based_classes_when_all_ten_classes_are_injected(self):
        """Given labels 1..10; when bridging; then preserve labels, never shift twice."""
        self.prediction = {
            'pred_boxes': np.tile([0., 0., 0., 2., 2., 2., 0.], (10, 1)),
            'pred_labels': np.arange(1, 11), 'pred_scores': np.full(10, .9),
            'spcra_reliability': np.linspace(0., 1., 10),
            'spcra_k4_accepted': np.ones(10, dtype=bool),
        }
        info = self.pseudo_info()
        batch = {'batch_size': 1, 'frame_id': ['sample']}
        self.bridge.inject_k4_pseudo_labels(batch, [info])
        np.testing.assert_array_equal(info['gt_boxes'][:, 7], np.arange(1, 11))
        np.testing.assert_array_equal(batch['gt_boxes'][0, :, -1], np.arange(1, 11))
        np.testing.assert_array_equal(batch['tta_pseudo_weights'][0], np.linspace(0., 1., 10))

    def test_malformed_sidecars_fail_when_crossing_prediction_boundary(self):
        """Given missing/wrong/nonfinite sidecars; when parsed; then fail, never use ones."""
        for key, value in (
            ('spcra_reliability', None), ('spcra_reliability', np.ones(4)),
            ('spcra_reliability', np.ones((5, 1))),
            ('spcra_reliability', np.array([0., .2, np.nan, .4, 1.])),
            ('spcra_reliability', np.array([0., .2, np.inf, .4, 1.])),
            ('spcra_reliability', np.array([-.1, .2, .3, .4, 1.])),
            ('spcra_reliability', np.array([0., .2, .3, .4, 1.1])),
            ('spcra_k4_accepted', None), ('spcra_k4_accepted', np.ones(4, dtype=bool)),
            ('spcra_k4_accepted', np.ones(5)),
        ):
            with self.subTest(key=key, value=value):
                prediction = dict(self.prediction)
                if value is None:
                    del prediction[key]
                else:
                    prediction[key] = value
                with self.assertRaises(ValueError):
                    self.bridge.build_k4_pseudo_info(prediction, self.thresholds)

    def test_malformed_pseudo_weights_fail_when_injected(self):
        """Given misaligned persisted sidecars; when injected; then reject corruption."""
        info = self.pseudo_info()
        info['reliability_weights'] = np.ones(3)
        with self.assertRaises(ValueError):
            self.bridge.inject_k4_pseudo_labels({'batch_size': 1}, [info])


if __name__ == '__main__':
    unittest.main()
