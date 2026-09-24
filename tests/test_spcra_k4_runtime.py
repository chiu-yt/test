"""CPU composition contract: four extra forwards, not a replacement MOS optimizer.

run_k4_views(model, batch, predictions, *, view_samples, build_view,
forward_view, policy) returns enriched prediction dictionaries. build_view receives
an isolated reference batch, retained indices and forward transform; forward_view
receives (model, view_batch) and returns prediction dictionaries. ViewSamples is
the existing seeding API, one entry per frame. Only compact predictions may live
across forwards, never view batches/features. No dataset stream is prescribed.
"""

import importlib.util
from pathlib import Path
import pickle
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
import weakref

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_contract_module(name):
    path = ROOT / 'pcdet/tta_methods' / (name + '.py')
    if not path.is_file():
        raise AssertionError(f'Missing formal K4 composition module: {path.name}')
    package_name = '_k4_contract_modules'
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(path.parent)]
        sys.modules[package_name] = package
    qualified = package_name + '.' + name
    if qualified not in sys.modules:
        spec = importlib.util.spec_from_file_location(qualified, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return sys.modules[qualified]


class FakeModel:
    """Mutable mode tree without a detector, Torch, or device allocation."""

    def __init__(self):
        self.training = True
        self.child = SimpleNamespace(training=False)

    def modules(self):
        return (self, self.child)

    def train(self, mode=True):
        for module in self.modules():
            module.training = mode
        return self

    def eval(self):
        return self.train(False)


class ForwardFailure(RuntimeError):
    pass


class SinkFailure(RuntimeError):
    pass


class TestK4Runtime(unittest.TestCase):
    def setUp(self):
        self.runtime = load_contract_module('spcra_k4_runtime')
        self.core = load_contract_module('spcra_k4_core')
        self.seeding = load_contract_module('spcra_k4_seeding')
        self.model = FakeModel()
        self.batch = {'points': np.arange(80.).reshape(20, 4),
                      'images': np.ones((2, 3)), 'frame_id': ['sample-a']}
        self.prediction = {'pred_boxes': np.array([[0., 0., 0., 2., 2., 2., 0.]]),
                           'pred_labels': np.array([10]), 'pred_scores': np.array([.9])}
        self.policy = self.core.K4Policy()
        self.calls = []
        self.live_views = []
        self.fail_at = None
        self.samples = self.seeding.sample_unique_views(
            'sample-a', base_seed=42, schema_version='k4_v1', reference_digest='a'*64,
            law_identifier='spcra_legacy_bernoulli_world_v1', build_view=self.sample_spec,
        )

    @staticmethod
    def sample_spec(drop_rng, geometry_rng):
        transform = np.eye(4)
        transform[0, 3] = geometry_rng.uniform(-.2, .2)
        return np.sort(drop_rng.choice(20, 14, replace=False)), transform

    def build_view(self, pristine, indices, transform):
        self.assertTrue(all(reference() is None for reference in self.live_views),
                        'Previous view state survived into the next build')
        np.testing.assert_array_equal(pristine['points'], self.batch['points'])
        np.testing.assert_array_equal(pristine['images'], self.batch['images'])
        self.assertIsNot(pristine['points'], self.batch['points'])
        pristine['points'][:] = -999
        pristine['images'][:] = -999
        pristine['points'] = np.zeros((len(indices), 4))
        pristine['lidar_aug_matrix'] = np.array(transform, copy=True)
        self.live_views.extend(weakref.ref(pristine[key])
                               for key in ('points', 'images', 'lidar_aug_matrix'))
        return pristine

    def forward_view(self, model, view):
        self.assertTrue(all(not module.training for module in model.modules()))
        index = len(self.calls)
        self.calls.append(index)
        view['spatial_features_2d'] = np.ones((4, 4))
        self.live_views.append(weakref.ref(view['spatial_features_2d']))
        random.random()
        np.random.random()
        if index == self.fail_at:
            raise ForwardFailure()
        boxes = self.prediction['pred_boxes'].copy()
        boxes[:, 0] += view['lidar_aug_matrix'][0, 3] + index * .25
        return [dict(self.prediction, pred_boxes=boxes)]

    def execute(self):
        return self.runtime.run_k4_views(
            self.model, self.batch, [self.prediction], view_samples=(self.samples,),
            build_view=self.build_view, forward_view=self.forward_view, policy=self.policy,
        )

    def execute_with_sink(self, evidence_sink):
        return self.runtime.run_k4_views(
            self.model, self.batch, [self.prediction], view_samples=(self.samples,),
            build_view=self.build_view, forward_view=self.forward_view, policy=self.policy,
            evidence_sink=evidence_sink,
        )

    def test_four_sequential_forwards_when_reference_is_mutable(self):
        """Given mutable input; when running views; then isolate and release all four."""
        original = self.batch['points'].copy()
        output = self.execute()
        self.assertEqual(self.calls, [0, 1, 2, 3])
        self.assertTrue(all(reference() is None for reference in self.live_views))
        np.testing.assert_array_equal(self.batch['points'], original)
        np.testing.assert_array_equal(self.batch['images'], np.ones((2, 3)))
        self.assertNotIn('spcra_reliability', self.prediction)
        np.testing.assert_array_equal(output[0]['pred_scores'], [.9])

    def test_attached_sidecar_when_four_views_have_distinct_costs(self):
        """Given costs 0/.25/.5/.75; when composing; then attach the real core result."""
        reference = self.core.Predictions(self.prediction['pred_boxes'], [10], [.9])
        views = []
        for index, matrix in enumerate(self.samples.transforms):
            boxes = self.prediction['pred_boxes'].copy()
            boxes[:, 0] += matrix[0, 3] + index * .25
            views.append(self.core.Predictions(boxes, [10], [.9], lidar_aug_matrix=matrix))
        expected = self.core.compute_k4_reliability(reference, tuple(views), self.policy)
        output = self.execute()
        np.testing.assert_allclose(output[0]['spcra_reliability'], expected.reliability)
        np.testing.assert_array_equal(output[0]['spcra_k4_accepted'], expected.reference_mask)

    def test_distinct_deterministic_specs_when_same_token_is_replayed(self):
        """Given identical identity; when replayed; then use identical distinct specs."""
        recorded = []
        original_builder = self.build_view

        def recording_builder(pristine, indices, transform):
            recorded.append((indices.copy(), transform.copy()))
            return original_builder(pristine, indices, transform)

        self.build_view = recording_builder
        self.execute()
        self.execute()
        self.assertEqual(len(recorded), 8)
        self.assertEqual(len(set(self.samples.fingerprints)), 4)
        for index in range(4):
            for actual in (recorded[index], recorded[index + 4]):
                np.testing.assert_array_equal(actual[0], self.samples.indices[index])
                np.testing.assert_array_equal(actual[1], self.samples.transforms[index])

    def test_modes_and_rng_restored_when_success_or_forward_failure(self):
        """Given mixed modes/global RNG; when success or failure; then restore both."""
        for failure in (None, 0, 2, 3):
            with self.subTest(failure=failure):
                self.calls.clear()
                self.fail_at = failure
                python_state = random.getstate()
                numpy_state = pickle.dumps(np.random.get_state())
                if failure is None:
                    self.execute()
                else:
                    with self.assertRaises(ForwardFailure):
                        self.execute_with_sink(lambda evidence: None)
                self.assertEqual([module.training for module in self.model.modules()], [True, False])
                self.assertEqual(random.getstate(), python_state)
                self.assertEqual(pickle.dumps(np.random.get_state()), numpy_state)
                self.assertTrue(all(reference() is None for reference in self.live_views))

    def test_modes_and_rng_restored_when_view_builder_fails(self):
        """Given a failing builder; when it consumes RNG; then restore caller state."""
        python_state = random.getstate()
        numpy_state = pickle.dumps(np.random.get_state())

        def failing_builder(pristine, indices, transform):
            random.random()
            np.random.random()
            raise ForwardFailure()

        self.build_view = failing_builder
        with self.assertRaises(ForwardFailure):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertEqual([module.training for module in self.model.modules()], [True, False])
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(pickle.dumps(np.random.get_state()), numpy_state)

    def test_k4_input_error_when_forward_changes_batch_size(self):
        """Given a missing view output; when running views; then retain the K4 error and restore state."""
        python_state = random.getstate()
        numpy_state = pickle.dumps(np.random.get_state())
        self.forward_view = lambda model, view: []

        with self.assertRaisesRegex(self.core.K4InputError, 'View predictions must retain the reference batch order'):
            self.execute()

        self.assertEqual([module.training for module in self.model.modules()], [True, False])
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(pickle.dumps(np.random.get_state()), numpy_state)
        self.assertTrue(all(reference() is None for reference in self.live_views))

    def test_optional_sink_observes_owned_evidence_without_changing_enrichment(self):
        baseline = self.execute()
        self.calls.clear()
        captured = []

        output = self.execute_with_sink(captured.append)

        self.assertEqual(set(output[0]), set(baseline[0]))
        for key in output[0]:
            np.testing.assert_array_equal(output[0][key], baseline[0][key])
        self.assertEqual(len(captured), 1)
        observed = captured[0]
        np.testing.assert_array_equal(observed.reliability, output[0]['spcra_reliability'])
        np.testing.assert_array_equal(observed.reference_mask, output[0]['spcra_k4_accepted'])
        self.assertEqual(observed.fingerprints, self.samples.fingerprints)
        self.assertTrue(all(reference() is None for reference in self.live_views))

    def test_sink_failure_restores_modes_rng_and_releases_view_features(self):
        python_state = random.getstate()
        numpy_state = pickle.dumps(np.random.get_state())

        def failing_sink(evidence):
            random.random()
            np.random.random()
            raise SinkFailure()

        with self.assertRaises(SinkFailure):
            self.execute_with_sink(failing_sink)
        self.assertEqual([module.training for module in self.model.modules()], [True, False])
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(pickle.dumps(np.random.get_state()), numpy_state)
        self.assertTrue(all(reference() is None for reference in self.live_views))

    def test_sink_observes_empty_prediction_rows(self):
        self.prediction = {
            'pred_boxes': np.empty((0, 7)),
            'pred_labels': np.empty(0, dtype=np.int64),
            'pred_scores': np.empty(0),
        }
        captured = []

        output = self.execute_with_sink(captured.append)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].match_indices.shape, (0, 4))
        self.assertEqual(output[0]['spcra_reliability'].shape, (0,))


if __name__ == '__main__':
    unittest.main()
