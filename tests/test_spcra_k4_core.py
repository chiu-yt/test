"""Formal K4 contract; run with python3 -m unittest discover -s tests -p 'test_spcra_k4_*.py'."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


class TestK4Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'pcdet/tta_methods/spcra_k4_core.py'
        spec = importlib.util.spec_from_file_location('spcra_k4_core', path)
        assert spec is not None and spec.loader is not None
        cls.core = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.core
        spec.loader.exec_module(cls.core)

    def predictions(self, centers=(0.0,), scores=None):
        boxes = np.tile([0., 0., 0., 2., 2., 2., 0.], (len(centers), 1))
        boxes[:, 0] = centers
        return self.core.Predictions(
            boxes=boxes, labels=np.ones(len(centers), dtype=np.int64),
            scores=np.full(len(centers), 0.9) if scores is None else np.array(scores),
        )

    def evaluate(self, reference, views, **options):
        return self.core.compute_k4_reliability(
            reference, tuple(views), self.core.K4Policy(**options),
        )

    def test_perfect_and_missing_views_use_fixed_denominator(self):
        # Given one eligible singleton and zero through four perfect views.
        reference, empty = self.predictions(), self.predictions(())
        for matched_count in (4, 3, 0):
            with self.subTest(matched_count=matched_count):
                views = [reference] * matched_count + [empty] * (4 - matched_count)
                # When four independent views are aggregated.
                result = self.evaluate(reference, views)
                # Then a singleton has no floor, support penalty, or legacy cap.
                np.testing.assert_array_equal(result.reliability, [matched_count / 4])

    def test_exact_mean_of_four_exponential_costs(self):
        # Given distance-only costs 0, 0.25, 0.5, 1 at a one-meter gate.
        views = [self.predictions((offset,)) for offset in (0., .25, .5, 1.)]
        # When matching includes the boundary distance.
        result = self.evaluate(self.predictions(), views, max_center_distance=1.)
        # Then all four exp(-cost) terms, not a frame-level surrogate, contribute.
        expected = np.exp(-np.array([0., .25, .5, 1.]))
        np.testing.assert_allclose(result.view_quality, expected[None, :], rtol=1e-7)
        np.testing.assert_allclose(result.reliability, [expected.mean()], rtol=1e-7)

    def test_diagnostics_never_rescale_reliability(self):
        # Given a perfect singleton among unmatched eligible proposals.
        reference, view = self.predictions((0., 10., 20.)), self.predictions((0., 30.))
        # When the diagnostic support scale changes by six orders of magnitude.
        results = [self.evaluate(reference, [view] * 4, support_tau=tau)
                   for tau in (.001, 1000.)]
        # Then coverage/support may change but the formal r remains unchanged.
        for result in results:
            np.testing.assert_array_equal(result.reliability, [1., 0., 0.])
        self.assertFalse(np.array_equal(results[0].support, results[1].support))
        self.assertTrue(np.all(np.asarray(results[0].coverage) < 1.))

    def test_reference_priority_is_descending_score_then_original_index(self):
        # Given three competing references, with a score tie at original indices 1/2.
        reference = self.predictions((0., 0., 0.), (.6, .9, .9))
        view = self.predictions((0., 0.))
        # When greedy one-to-one matching runs independently in every view.
        result = self.evaluate(reference, [view] * 4)
        # Then the lower-score reference loses and the tied earlier index wins first.
        np.testing.assert_array_equal(result.match_indices, [[-1]*4, [0]*4, [1]*4])
        np.testing.assert_array_equal(result.reliability, [0., 1., 1.])

    def test_view_priority_is_distance_then_original_index_not_score(self):
        # Given a farther high-score candidate and two equally near candidates.
        view = self.predictions((.8, .2, -.2), (.99, .6, .95))
        # When the reference selects its nearest available candidate.
        result = self.evaluate(self.predictions(), [view] * 4)
        # Then distance beats score and original index breaks the distance tie.
        np.testing.assert_array_equal(result.match_indices, [[1]*4])

    def test_all_ten_classes_match_only_their_own_class(self):
        # Given ten colocated references and views in reversed class order.
        reference, view = self.predictions((0.,)*10), self.predictions((0.,)*10)
        reference.labels[:] = np.arange(1, 11)
        view.labels[:] = np.arange(10, 0, -1)
        # When class-constrained matching runs.
        result = self.evaluate(reference, [view]*4)
        # Then all ten classes survive without cross-class consumption.
        np.testing.assert_array_equal(result.match_indices, np.repeat(np.arange(9, -1, -1)[:, None], 4, axis=1))
        np.testing.assert_array_equal(result.reliability, np.ones(10))

    def test_wrong_class_and_outside_boundary_are_unmatched(self):
        # Given a colocated wrong class and same-class point just outside the gate.
        view = self.predictions((0., np.nextafter(1., np.inf)))
        view.labels[0] = 2
        # When the inclusive one-meter gate is applied.
        result = self.evaluate(self.predictions(), [view]*4, max_center_distance=1.)
        # Then neither candidate can create a match.
        np.testing.assert_array_equal(result.match_indices, [[-1]*4])
        np.testing.assert_array_equal(result.reliability, [0.])

    def test_invalid_geometry_and_nonfinite_scores_keep_original_masks(self):
        # Given invalid centers/dimensions/yaw/scores interleaved with valid boxes.
        for field, value in ((0, np.nan), (1, np.inf), (3, 0.), (4, -1.), (6, np.nan)):
            with self.subTest(field=field, value=value):
                reference, view = self.predictions((0.,)*4), self.predictions((0.,)*4)
                reference.boxes[0, field] = view.boxes[2, field] = value
                reference.scores[2] = view.scores[0] = np.nan
                # When each view filters locally without compacting output indices.
                result = self.evaluate(reference, [view]*4)
                # Then original-length masks and original view indices stay aligned.
                np.testing.assert_array_equal(result.reference_mask, [False, True, False, True])
                np.testing.assert_array_equal(result.view_masks, [[False, True, False, True]]*4)
                np.testing.assert_array_equal(result.match_indices, [[-1]*4, [1]*4, [-1]*4, [3]*4])
                np.testing.assert_array_equal(result.reliability, [0., 1., 0., 1.])

    def test_inverse_alignment_precedes_matching(self):
        # Given a view translated ten meters from the reference frame.
        view = self.predictions((10.,))
        matrix = np.eye(4)
        matrix[0, 3] = 10.
        aligned_view = self.core.Predictions(
            boxes=view.boxes, labels=view.labels, scores=view.scores,
            lidar_aug_matrix=matrix,
        )
        # When the view's forward augmentation is inverted for comparison.
        result = self.evaluate(self.predictions(), [aligned_view]*4)
        # Then aligned geometry yields a perfect match rather than an unmatched box.
        np.testing.assert_array_equal(result.reliability, [1.])

    def test_camera_rescue_changes_only_local_eligibility(self):
        # Given a low-score reference and views with differing camera support/geometry.
        reference = self.predictions(scores=(.2,))
        reference = self.core.Predictions(boxes=reference.boxes, labels=reference.labels,
                                          scores=reference.scores, camera_support=np.array([.9]))
        views = []
        for center, support in ((0., .9), (0., .1), (10., .9), (0., .9)):
            pred = self.predictions((center,), (.2,))
            views.append(self.core.Predictions(boxes=pred.boxes, labels=pred.labels,
                         scores=pred.scores, camera_support=np.array([support])))
        views[3].labels[0] = 2
        # When rescue admits only locally supported low-score proposals.
        result = self.evaluate(reference, views, min_proposal_score=.5,
                               camera_rescue_enabled=True, camera_low_score=.1, camera_thresh=.6)
        # Then rescued unmatched/wrong-class boxes contribute zero, never fabricated quality.
        np.testing.assert_array_equal(result.reference_mask, [True])
        np.testing.assert_array_equal(result.view_masks, [[True], [False], [True], [True]])
        np.testing.assert_array_equal(result.view_quality, [[1., 0., 0., 0.]])
        np.testing.assert_array_equal(result.reliability, [.25])

    def test_camera_support_without_rescue_cannot_admit_low_score_boxes(self):
        # Given high camera support on otherwise ineligible predictions.
        pred = self.predictions(scores=(.2,))
        supported = self.core.Predictions(boxes=pred.boxes, labels=pred.labels,
                    scores=pred.scores, camera_support=np.array([1.]))
        # When camera rescue is disabled for reference and all four views.
        result = self.evaluate(supported, [supported]*4, min_proposal_score=.5,
                               camera_rescue_enabled=False)
        # Then support alone does not admit a proposal or create reliability.
        np.testing.assert_array_equal(result.reference_mask, [False])
        np.testing.assert_array_equal(result.view_masks, [[False]]*4)
        np.testing.assert_array_equal(result.reliability, [0.])

    def test_formal_method_rejects_other_view_counts(self):
        # Given a valid reference but a non-K4 view count.
        reference = self.predictions()
        for count in (0, 1, 3, 5):
            with self.subTest(count=count):
                # When formal K4 aggregation is requested.
                with self.assertRaises(ValueError):
                    self.evaluate(reference, [reference]*count)
                # Then no variable-denominator result is returned.


if __name__ == '__main__':
    unittest.main()
