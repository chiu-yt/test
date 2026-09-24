from dataclasses import replace

import numpy as np
import unittest

from figure7_fixtures import observation, with_reliability
from pcdet.utils.figure7_selection import reference_rows, candidates, retain, select_pair
from pcdet.utils.figure7_schema import Figure7Error, PoolLimits


class TestFigure7Selection(unittest.TestCase):
    def test_thresholds_when_on_boundaries(self):
        """Given boundary evidence, when ranked, then retain approved thresholds only."""
        cases = (
            (True, .65, .80, True), (True, .6499, .80, False),
            (True, .9, .7999, False), (False, .85, .30, True),
            (False, .8499, .30, False), (False, .9, .3001, False),
            (True, .85, .30, True), (True, .8499, .30, False),
            (True, .85, .3001, False),
        )
        for stable, score, reliability, eligible in cases:
            with self.subTest(stable=stable, score=score, reliability=reliability):
                frame = with_reliability(observation('token', stable, score), reliability)
                rows = reference_rows(frame)
                self.assertEqual(bool(candidates(frame, rows)), eligible)


    def test_missing_match_when_stable_is_ineligible(self):
        """Given high r but a missing view, when ranked, then reject stable eligibility."""
        frame = with_reliability(observation('token', False), .9)
        self.assertFalse(candidates(frame, reference_rows(frame)))


    def test_rejected_when_high_confidence_is_ineligible(self):
        """Given a rejected reference, when ranked, then never promote it."""
        frame = observation('token')
        assert frame.evidence is not None
        frame = replace(frame, evidence=replace(frame.evidence, reference_mask=np.array([False])))
        self.assertFalse(candidates(frame, reference_rows(frame)))


    def test_pair_when_same_token_fails(self):
        """Given two cases on one token, when paired, then fail instead of duplicating."""
        frames = (observation('same'), observation('same', False))
        ranked = tuple(item for frame in frames for item in candidates(frame, reference_rows(frame)))
        with self.assertRaises(Figure7Error):
            select_pair(ranked)

    def test_class_priority_when_other_class_has_higher_confidence(self):
        """Given a confident car and a pedestrian, when bounded, then retain priority class."""
        pedestrian = observation('pedestrian', score=.85)
        assert pedestrian.evidence is not None
        pedestrian = replace(pedestrian, evidence=replace(
            pedestrian.evidence, reference_prediction=replace(
                pedestrian.evidence.reference_prediction, labels=np.array([9]),
            ),
        ))
        car = observation('car', score=.99)
        assert car.evidence is not None
        car = replace(car, evidence=replace(car.evidence, reference_prediction=replace(
            car.evidence.reference_prediction, labels=np.array([1]),
        )))
        ranked = tuple(item for frame in (car, pedestrian)
                       for item in candidates(frame, reference_rows(frame)))
        self.assertEqual(retain(ranked, PoolLimits(1, 1))[0].identity.token, 'pedestrian')

    def test_pair_when_matching_confidence_beats_pool_leader(self):
        """Given same-class candidates, when paired, then prefer closest confidence."""
        frames = (observation('stable-close', score=.9), observation('stable-top', score=.99),
                  observation('variable', False, .9))
        ranked = tuple(item for frame in frames for item in candidates(frame, reference_rows(frame)))
        self.assertEqual(select_pair(ranked)[1].identity.token, 'stable-close')

    def test_variable_when_all_views_match_remains_eligible(self):
        """Given low r and four matches, when ranked, then admit Case A without a missing view."""
        frame = with_reliability(observation('token'), .2)
        assert frame.evidence is not None
        frame = replace(frame, evidence=replace(frame.evidence, view_quality=np.full((1, 4), .2)))
        rows = reference_rows(frame)
        self.assertEqual(rows[0].matches, (0, 0, 0, 0))
        self.assertTrue(rows[0].variable_eligible)
        ranked = candidates(frame, rows)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].pool, 'variable')

    def test_class_names_when_labels_follow_root_nuscenes_order(self):
        """Given nuScenes labels, when summarized, then label 8 is bicycle and 9 is pedestrian."""
        frame = observation('token')
        assert frame.evidence is not None
        expected = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
        for label, name in enumerate(expected, start=1):
            with self.subTest(label=label):
                prediction = replace(frame.evidence.reference_prediction, labels=np.array([label]))
                labeled = replace(frame, evidence=replace(frame.evidence, reference_prediction=prediction))
                self.assertEqual(reference_rows(labeled)[0].class_name, name)
