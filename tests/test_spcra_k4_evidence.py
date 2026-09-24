import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_contract_module(name):
    path = ROOT / 'pcdet/tta_methods' / (name + '.py')
    if not path.is_file():
        raise AssertionError(f'Missing formal K4 composition module: {path.name}')
    package_name = '_k4_evidence_contract_modules'
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


class TestK4Evidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_contract_module('spcra_k4_core')
        cls.seeding = load_contract_module('spcra_k4_seeding')
        cls.evidence = load_contract_module('spcra_k4_evidence')

    def predictions(self, centers, scores, support=None, transform=None):
        boxes = np.tile([0., 0., 0., 2., 2., 2., 0.], (len(centers), 1))
        boxes[:, 0] = centers
        return self.core.Predictions(
            boxes, np.ones(len(centers), dtype=np.int64), np.array(scores),
            camera_support=support, lidar_aug_matrix=transform,
        )

    def samples(self):
        transforms = tuple(np.eye(4, dtype=np.float32) for _ in range(4))
        return self.seeding.ViewSamples(
            indices=tuple(np.arange(index + 1) for index in range(4)),
            transforms=transforms,
            fingerprints=tuple(f'fingerprint-{index}' for index in range(4)),
            attempts=(0, 1, 2, 3),
            law_identifier='spcra_legacy_bernoulli_world_v1',
        )

    def test_evidence_owns_original_order_predictions_and_formal_results(self):
        reference = self.predictions((0., 5.), (.2, .9), support=np.array([.9, .1]))
        views = []
        for transform in self.samples().transforms:
            view = self.predictions((99., 0., 5.), (.9, .2, .9),
                                    support=np.array([.1, .9, .1]), transform=transform)
            view.boxes[0, 3] = 0.
            views.append(view)
        policy = self.core.K4Policy(
            min_proposal_score=.5, camera_rescue_enabled=True,
            camera_low_score=.1, camera_thresh=.6,
        )
        result = self.core.compute_k4_reliability(reference, tuple(views), policy)

        source = self.evidence.K4EvidenceInput(reference, tuple(views), self.samples())
        observed = self.evidence.build_k4_evidence(source, result)

        np.testing.assert_array_equal(observed.reference_prediction.boxes[:, 0], [0., 5.])
        np.testing.assert_array_equal(observed.view_predictions[0].boxes[:, 0], [99., 0., 5.])
        np.testing.assert_array_equal(observed.reference_mask, [True, True])
        np.testing.assert_array_equal(observed.reference_rescue_mask, [True, False])
        np.testing.assert_array_equal(observed.view_masks[0], [False, True, True])
        np.testing.assert_array_equal(observed.view_rescue_masks[0], [False, True, False])
        np.testing.assert_array_equal(observed.match_indices, [[1] * 4, [2] * 4])
        np.testing.assert_array_equal(observed.view_quality, result.view_quality)
        np.testing.assert_array_equal(observed.reliability, result.reliability)
        np.testing.assert_array_equal(observed.coverage, result.coverage)
        np.testing.assert_array_equal(observed.support, result.support)
        self.assertEqual(observed.fingerprints, self.samples().fingerprints)
        self.assertEqual(observed.attempts, self.samples().attempts)
        self.assertEqual(observed.law_identifier, self.samples().law_identifier)

    def test_evidence_arrays_are_read_only_owned_copies_with_retained_indices(self):
        samples = self.samples()
        reference = self.predictions((0.,), (.9,))
        views = tuple(self.predictions((0.,), (.9,), transform=matrix)
                      for matrix in samples.transforms)
        result = self.core.compute_k4_reliability(reference, views, self.core.K4Policy())

        source = self.evidence.K4EvidenceInput(reference, views, samples)
        observed = self.evidence.build_k4_evidence(source, result)
        arrays = (
            observed.reference_prediction.boxes,
            observed.reference_prediction.labels,
            observed.reference_prediction.scores,
            observed.match_indices,
            observed.view_quality,
            observed.reliability,
            observed.coverage,
            observed.support,
            *observed.transforms,
            *observed.retained_indices,
            observed.reference_mask,
        )

        expected_indices = tuple(np.arange(index + 1) for index in range(4))
        self.assertEqual(len(observed.retained_indices), 4)
        for index, expected in enumerate(expected_indices):
            np.testing.assert_array_equal(observed.retained_indices[index], expected)
            self.assertEqual(observed.retained_indices[index].dtype, np.dtype(np.int64))
            self.assertEqual(observed.fingerprints[index], f'fingerprint-{index}')
            self.assertEqual(observed.attempts[index], index)
            self.assertFalse(np.shares_memory(observed.retained_indices[index], samples.indices[index]))
        reference.boxes[0, 0] = 42.
        samples.transforms[0][0, 3] = 42.
        samples.indices[3][0] = 42
        self.assertEqual(observed.reference_prediction.boxes[0, 0], 0.)
        self.assertEqual(observed.transforms[0][0, 3], 0.)
        np.testing.assert_array_equal(observed.retained_indices[3], expected_indices[3])
        self.assertTrue(all(array.flags.owndata and not array.flags.writeable for array in arrays))

    def test_empty_predictions_are_valid_observed_evidence(self):
        samples = self.samples()
        reference = self.predictions((), ())
        views = tuple(self.predictions((), (), transform=matrix) for matrix in samples.transforms)
        result = self.core.compute_k4_reliability(reference, views, self.core.K4Policy())

        source = self.evidence.K4EvidenceInput(reference, views, samples)
        observed = self.evidence.build_k4_evidence(source, result)

        self.assertEqual(observed.reference_prediction.boxes.shape, (0, 7))
        self.assertEqual(observed.match_indices.shape, (0, 4))
        self.assertEqual(len(observed.view_predictions), 4)


if __name__ == '__main__':
    unittest.main()
