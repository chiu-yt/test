from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import weakref

import numpy as np

from figure7_fixtures import observation, provenance
from pcdet.utils.figure7_runtime import Figure7Collector
from pcdet.utils.figure7_schema import Figure7Error, PoolLimits
from pcdet.utils.figure7_artifacts import verify_record


class TestFigure7Runtime(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_bounded_when_many_occurrences_are_observed(self):
        """Given a stream, when collected, then retain bounded pools and every ledger row."""
        collector = Figure7Collector(self.root / 'run', provenance(), PoolLimits(2, 2))
        for index in range(30):
            collector.observe(observation(str(index), index % 2 == 0, .86 + index / 1000))
            self.assertLessEqual(len(tuple((collector.output / 'records').iterdir())), 4)
            self.assertFalse((collector.output / '_staging').exists())
        pair = collector.finalize()
        self.assertNotEqual(pair[0].identity.token, pair[1].identity.token)
        self.assertEqual(len(collector.retained), 4)
        ledger = [json.loads(line) for line in (collector.output / 'ledger.jsonl').read_text().splitlines()]
        self.assertEqual(len(ledger), 30)
        self.assertEqual(ledger[-1]['references'][0]['matches'], [0, -1, -1, -1])
        self.assertEqual(ledger[-1]['references'][0]['reliability'], .25)
        for path in (collector.output / 'records').iterdir():
            verify_record(path)
            with np.load(path / 'arrays.npz', allow_pickle=False) as archive:
                self.assertEqual(len([name for name in archive.files if name.endswith('_points')]), 5)
                self.assertTrue(all(archive[name].dtype.kind in 'biuf' for name in archive.files))

    def test_release_when_observation_returns(self):
        """Given current-frame arrays, when capture returns, then collector retains no arrays."""
        collector = Figure7Collector(self.root / 'run', provenance())
        frame = observation('token')
        assert frame.points is not None and frame.evidence is not None
        references = (weakref.ref(frame.points.reference), weakref.ref(frame.evidence.reliability))
        collector.observe(frame)
        del frame
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual((collector.limits.stable, collector.limits.variable, collector.limits.capacity), (10, 10, 20))

    def test_ledger_when_incomplete_and_failed(self):
        """Given non-candidates, when observed, then append once each without archives."""
        collector = Figure7Collector(self.root / 'run', provenance())
        frame = observation('token')
        collector.observe(replace(frame, evidence=None, points=None))
        collector.observe(replace(frame, failure='forward failed'))
        collector.observe(replace(frame, points=None))
        collector.observe(observation('low', score=.1))
        self.assertFalse(collector.retained)
        rows = [json.loads(line) for line in (collector.output / 'ledger.jsonl').read_text().splitlines()]
        self.assertEqual([row['status'] for row in rows], ['incomplete', 'failed', 'incomplete', 'ineligible'])
        with self.assertRaises(Figure7Error):
            collector.finalize()
        self.assertEqual(json.loads((collector.output / 'selection.json').read_text())['status'], 'incomplete')

    def test_fresh_directory_when_existing_is_rejected(self):
        """Given an existing directory, when initialized, then never reuse it."""
        with self.assertRaises(FileExistsError):
            Figure7Collector(self.root, provenance())

    def test_determinism_when_stream_is_reversed(self):
        """Given tied candidates, when stream order changes, then selection stays identical."""
        frames = [observation('s' + str(index)) for index in range(4)]
        frames += [observation('v' + str(index), False) for index in range(4)]
        pairs = []
        for index, stream in enumerate((frames, list(reversed(frames)))):
            collector = Figure7Collector(self.root / str(index), provenance(), PoolLimits(2, 2))
            for frame in stream:
                collector.observe(frame)
            pairs.append(tuple(item.identity.token for item in collector.finalize()))
        self.assertEqual(pairs[0], pairs[1])
        self.assertEqual(pairs[0], ('v0', 's0'))

    def test_shared_record_when_both_pools_admit_one_occurrence(self):
        """Given two reference cases, when both pools admit, then store one complete record."""
        frame = observation('both')
        assert frame.evidence is not None
        evidence = frame.evidence
        prediction = replace(evidence.reference_prediction,
                             boxes=np.repeat(evidence.reference_prediction.boxes, 2, axis=0),
                             scores=np.array([.9, .9]), labels=np.array([8, 8]))
        combined = replace(evidence, reference_prediction=prediction,
                           reference_mask=np.array([True, True]),
                           reference_rescue_mask=np.array([False, True]),
                           match_indices=np.array([[0, 0, 0, 0], [-1, -1, -1, -1]]),
                           view_quality=np.array([[1., 1., 1., 1.], [0., 0., 0., 0.]]),
                           reliability=np.array([1., 0.]))
        collector = Figure7Collector(self.root / 'run', provenance(), PoolLimits(1, 1))
        ledger = collector.observe(replace(frame, evidence=combined))
        self.assertEqual(len(collector.retained), 2)
        self.assertEqual(len(tuple((collector.output / 'records').iterdir())), 1)
        self.assertEqual(ledger.admitted_pools, ('stable', 'variable'))
        with self.assertRaises(Figure7Error):
            collector.finalize()

    def test_ledger_when_complete_candidate_loses_ranking(self):
        """Given a full stronger pool, when eligible evidence loses, then eligibility is explicit."""
        collector = Figure7Collector(self.root / 'run', provenance(), PoolLimits(1, 1))
        collector.observe(observation('strong', score=.99))
        ledger = collector.observe(observation('weak', score=.9))
        self.assertEqual(ledger.status, 'complete')
        self.assertTrue(ledger.references[0].stable_eligible)
        self.assertFalse(ledger.references[0].variable_eligible)
        self.assertEqual(ledger.admitted_pools, ())
        self.assertEqual(len(tuple((collector.output / 'records').iterdir())), 1)
