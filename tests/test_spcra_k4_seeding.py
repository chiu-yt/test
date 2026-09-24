"""Canonical K4 identities and realized-view fingerprints, independent of CUDA."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import unittest

import numpy as np


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def expected_seed(identity, stream):
    payload = canonical_json((f'spcra-k4/{stream}', *identity))
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], 'little')


LAW_ID = 'spcra_legacy_bernoulli_world_v1'


def expected_fingerprint(reference_digest, indices, transform, schema_version, law_identifier=LAW_ID):
    indices = np.asarray(indices, dtype='<i8')
    transform = np.asarray(transform, dtype='<f4')
    header = canonical_json(('spcra-k4/view', law_identifier, reference_digest, schema_version,
                             ('<i8', indices.shape), ('<f4', transform.shape)))
    return hashlib.sha256(header + b'\x00' + indices.tobytes(order='C') +
                          transform.tobytes(order='C')).hexdigest()


def singleton_view(drop_rng, geometry_rng):
    return np.array([drop_rng.integers(4)], dtype='<i8'), np.eye(4, dtype='<f4')


class TestK4Seeding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'pcdet/tta_methods/spcra_k4_seeding.py'
        spec = importlib.util.spec_from_file_location('spcra_k4_seeding', path)
        assert spec is not None and spec.loader is not None
        cls.seeding = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.seeding
        spec.loader.exec_module(cls.seeding)

    def test_identity_fields_and_streams_determine_sha256_pcg64(self):
        # Given identities differing in each field, including a non-ASCII token.
        identities = ((17, 'frame-\u03bb', 0, 'v1', 0), (18, 'frame-\u03bb', 0, 'v1', 0),
                      (17, 'frame-b', 0, 'v1', 0), (17, 'frame-\u03bb', 1, 'v1', 0),
                      (17, 'frame-\u03bb', 0, 'v2', 0), (17, 'frame-\u03bb', 0, 'v1', 1))
        seeds = []
        for identity in identities:
            base_seed, token, view, version, attempt = identity
            for stream in ('drop', 'geometry'):
                wanted = expected_seed(identity, stream)
                expected = np.random.Generator(np.random.PCG64(wanted)).random(16)
                # When the API hashes compact UTF-8 JSON and seeds a local PCG64.
                options = dict(base_seed=base_seed, schema_version=version, attempt=attempt, stream=stream)
                seed = self.seeding.seed_for_view(token, view, **options)
                actual = self.seeding.rng_for_view(token, view, **options).random(16)
                # Then the first 16 SHA256 bytes are interpreted little-endian exactly.
                self.assertEqual(seed, wanted)
                np.testing.assert_array_equal(actual, expected)
                seeds.append(seed)
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_rank_order_global_rng_and_drop_consumption_do_not_affect_geometry(self):
        # Given different ranks, token orders, global RNG states and drop draw counts.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        saved_rank = os.environ.get('RANK')
        try:
            for rank, tokens in ((0, ('a', 'b')), (7, ('b', 'a'))):
                os.environ['RANK'] = str(rank)
                np.random.seed(10 + rank)
                random.seed(20 + rank)
                before_numpy = pickle.dumps(np.random.get_state())
                before_python = random.getstate()
                for token in tokens:
                    for view in range(4):
                        options = dict(base_seed=17, schema_version='v1', attempt=0)
                        # When independent streams are consumed in unequal amounts.
                        drop = self.seeding.rng_for_view(token, view, stream='drop', **options)
                        geometry = self.seeding.rng_for_view(token, view, stream='geometry', **options)
                        actual_drop = drop.random(3)
                        drop.random(100 + rank)
                        actual_geometry = geometry.random(8)
                        # Then neither rank, scheduling nor drop consumption changes either stream.
                        identity = (17, token, view, 'v1', 0)
                        for stream, actual in (('drop', actual_drop), ('geometry', actual_geometry)):
                            expected = np.random.Generator(np.random.PCG64(expected_seed(identity, stream)))
                            np.testing.assert_array_equal(actual, expected.random(len(actual)))
                self.assertEqual(before_numpy, pickle.dumps(np.random.get_state()))
                self.assertEqual(before_python, random.getstate())
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            if saved_rank is None:
                os.environ.pop('RANK', None)
            else:
                os.environ['RANK'] = saved_rank

    def test_fingerprint_hashes_canonical_realized_content(self):
        # Given equivalent endian/layout variants and distinct reference/index/transform/schema values.
        indices, transform = np.array([1, 7], dtype='<i8'), np.eye(4, dtype='<f4')
        translated = transform.copy()
        translated[0, 3] = .25
        reference = hashlib.sha256(b'reference').hexdigest()
        variants = ((reference, indices, transform, 'v1', LAW_ID),
                    (reference, indices.astype('>i8'), np.asfortranarray(transform.astype('>f8')), 'v1', LAW_ID),
                    (reference, indices, translated, 'v1', LAW_ID),
                    (reference, np.array([1, 8]), transform, 'v1', LAW_ID),
                    (hashlib.sha256(b'other').hexdigest(), indices, transform, 'v1', LAW_ID),
                    (reference, indices, transform, 'v2', LAW_ID),
                    (reference, indices, transform, 'v1', 'different-law'))
        # When fingerprints are computed from canonical dtype, shape and C-order content.
        actual = [self.seeding.fingerprint_view(*variant) for variant in variants]
        # Then equivalent realized views collide, but every semantic change is represented.
        self.assertEqual(actual, [expected_fingerprint(*variant) for variant in variants])
        self.assertEqual(actual[0], actual[1])
        self.assertEqual(len(set(actual)), 6)

    def test_realized_collisions_retry_attempt_specific_streams(self):
        # Given an identity transform and only four possible singleton retained subsets.
        reference = hashlib.sha256(b'reference').hexdigest()
        fingerprints, expected_views, attempts = [], [], []
        for view in range(4):
            for attempt in range(100):
                identity = (17, 'collision-frame', view, 'v1', attempt)
                generators = [np.random.Generator(np.random.PCG64(expected_seed(identity, stream)))
                              for stream in ('drop', 'geometry')]
                indices, transform = singleton_view(*generators)
                fingerprint = expected_fingerprint(reference, indices, transform, 'v1', LAW_ID)
                if fingerprint not in fingerprints:
                    fingerprints.append(fingerprint)
                    expected_views.append((indices, transform))
                    attempts.append(attempt)
                    break
        self.assertEqual(len(fingerprints), 4)
        self.assertGreater(max(attempts), 0)
        # When real view construction retries duplicate realized fingerprints.
        results = [self.seeding.sample_unique_views('collision-frame', base_seed=17, schema_version='v1',
                   reference_digest=reference, law_identifier=LAW_ID,
                   build_view=singleton_view, max_attempts=100) for _ in range(2)]
        # Then four distinct repeatable views carry matching transforms, fingerprints and retry counts.
        for result in results:
            self.assertEqual(tuple(result.fingerprints), tuple(fingerprints))
            self.assertEqual(result.law_identifier, LAW_ID)
            np.testing.assert_array_equal(result.attempts, attempts)
            np.testing.assert_array_equal(result.indices, [item[0] for item in expected_views])
            np.testing.assert_array_equal(result.transforms, [item[1] for item in expected_views])

    def test_same_indices_with_distinct_realized_transforms_are_distinct_views(self):
        # Given a synthetic builder with fixed indices and a geometry-stream translation.
        def build_view(drop_rng, geometry_rng):
            transform = np.eye(4, dtype='<f4')
            transform[0, 3] = geometry_rng.random()
            return np.array([0], dtype='<i8'), transform
        reference = hashlib.sha256(b'reference').hexdigest()
        expected = []
        for view in range(4):
            identity = (17, 'geometry-frame', view, 'v1', 0)
            generators = [np.random.Generator(np.random.PCG64(expected_seed(identity, stream)))
                          for stream in ('drop', 'geometry')]
            expected.append(build_view(*generators))
        # When uniqueness considers realized geometry as well as retained points.
        result = self.seeding.sample_unique_views('geometry-frame', base_seed=17, schema_version='v1',
            reference_digest=reference, law_identifier=LAW_ID, build_view=build_view, max_attempts=1)
        # Then all four transforms survive without retry; no geometry distribution is prescribed.
        np.testing.assert_array_equal(result.indices, [[0]]*4)
        np.testing.assert_array_equal(result.transforms, [item[1] for item in expected])
        self.assertEqual(tuple(result.fingerprints), tuple(
            expected_fingerprint(reference, indices, transform, 'v1', LAW_ID)
            for indices, transform in expected))
        self.assertEqual(len(set(result.fingerprints)), 4)
        np.testing.assert_array_equal(result.attempts, [0]*4)

    def test_duplicate_realizations_exhaust_exact_attempt_budget(self):
        # Given a real builder that always realizes identical indices and transform.
        calls = []
        def build_view(drop_rng, geometry_rng):
            calls.append((drop_rng.random(), geometry_rng.random()))
            return np.array([0], dtype='<i8'), np.eye(4, dtype='<f4')
        # When one accepted view is followed by three failed attempts for the next view.
        with self.assertRaises(self.seeding.ViewCollisionError):
            self.seeding.sample_unique_views('exhausted', base_seed=17, schema_version='v1',
                reference_digest=hashlib.sha256(b'reference').hexdigest(), law_identifier=LAW_ID,
                build_view=build_view, max_attempts=3)
        # Then different seeds never disguise duplicate realized content as distinct views.
        self.assertEqual(len(calls), 4)


if __name__ == '__main__':
    unittest.main()
