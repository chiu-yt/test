from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from figure7_fixtures import observation, provenance
from pcdet.utils.figure7_artifacts import stage_record, verify_record
from pcdet.utils.figure7_runtime import Figure7Collector
from pcdet.utils.figure7_schema import CurrentPoints, Figure7Error, PoolLimits


class TestFigure7Artifacts(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_ownership_when_sources_mutate(self):
        """Given CPU inputs, when the source mutates, then immutable evidence is unchanged."""
        values = np.ones((3, 4), dtype=np.float32)
        points = CurrentPoints(values, (values,) * 4, np.eye(4))
        values[:] = 7
        np.testing.assert_array_equal(points.reference, np.ones((3, 4)))
        self.assertTrue(all(not value.flags.writeable for value in (points.reference, *points.views)))
        with self.assertRaises(FrozenInstanceError):
            setattr(points, 'reference', values)

    def test_object_points_when_supplied_are_rejected(self):
        """Given object dtype points, when captured, then reject before persistence."""
        values = np.full((1, 3), 'bad', dtype=object)
        with self.assertRaises(Figure7Error):
            CurrentPoints(values, (values,) * 4, np.eye(4))

    def test_checksums_when_archive_corrupted(self):
        """Given a published record, when bytes change, then integrity verification fails."""
        collector = Figure7Collector(self.root / 'run', provenance())
        ledger = collector.observe(observation('stable'))
        path = collector.output / 'records' / ledger.record_id
        with (path / 'arrays.npz').open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaises(Figure7Error):
            verify_record(path)

    def test_failed_publication_when_rename_fails(self):
        """Given a rename failure, when observed, then one failed ledger and no committed record."""
        collector = Figure7Collector(self.root / 'run', provenance(), PoolLimits(1, 1))
        original = Path.replace

        def fail_publish(path: Path, destination: Path) -> Path:
            if path.name == '_staging':
                raise OSError('injected publication failure')
            return original(path, destination)

        with patch.object(Path, 'replace', fail_publish), self.assertRaises(OSError):
            collector.observe(observation('token'))
        rows = (collector.output / 'ledger.jsonl').read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])['status'], 'failed')
        self.assertFalse(tuple((collector.output / 'records').iterdir()))
        self.assertTrue((collector.output / '_staging').is_dir())
        self.assertEqual(json.loads((collector.output / 'selection.json').read_text())['status'], 'failed')
        with self.assertRaises(Figure7Error):
            collector.observe(observation('next'))

    def test_staging_bound_when_replacing_at_capacity(self):
        """Given full pools, when staging replacements, then storage never exceeds M+one."""
        collector = Figure7Collector(self.root / 'run', provenance(), PoolLimits(1, 1))
        collector.observe(observation('stable', score=.85))
        collector.observe(observation('variable', False, .85))
        counts = []

        def measured_stage(output, frame, ledger):
            staging = stage_record(output, frame, ledger)
            counts.append(len(tuple((output / 'records').iterdir())) + int(staging.exists()))
            return staging

        with patch('pcdet.utils.figure7_runtime.stage_record', measured_stage):
            collector.observe(observation('better', score=.95))
        self.assertEqual(counts, [3])
        self.assertEqual(len(tuple((collector.output / 'records').iterdir())), 2)

    def test_metadata_when_saved_contains_all_evidence(self):
        """Given an eligible frame, when stored, then exact original-row arrays round-trip."""
        collector = Figure7Collector(self.root / 'run', provenance())
        frame = observation('token', False)
        ledger = collector.observe(frame)
        path = collector.output / 'records' / ledger.record_id
        metadata = json.loads((path / 'metadata.json').read_text())
        self.assertEqual(metadata['observation']['model_step'], 0)
        self.assertEqual(metadata['observation']['fingerprints'], ['a', 'b', 'c', 'd'])
        self.assertEqual(len(metadata['observation']['transforms']), 5)
        assert frame.evidence is not None and frame.points is not None
        with np.load(path / 'arrays.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['reference_points'], frame.points.reference)
            np.testing.assert_array_equal(archive['match_indices'], frame.evidence.match_indices)
            np.testing.assert_array_equal(archive['reliability'], frame.evidence.reliability)
            for index in range(4):
                np.testing.assert_array_equal(archive[f'view_{index}_points'], frame.points.views[index])
                np.testing.assert_array_equal(archive[f'view_{index}_boxes'], frame.evidence.view_predictions[index].boxes)

    def test_ledger_when_reference_is_empty(self):
        """Given an observed empty reference, when collected, then distinguish empty from missing."""
        collector = Figure7Collector(self.root / 'run', provenance())
        frame = observation('token')
        assert frame.evidence is not None
        prediction = replace(frame.evidence.reference_prediction,
                             boxes=np.empty((0, 7)), labels=np.empty(0), scores=np.empty(0))
        evidence = replace(frame.evidence, reference_prediction=prediction,
                           reference_mask=np.empty(0, dtype=bool),
                           reference_rescue_mask=np.empty(0, dtype=bool),
                           match_indices=np.empty((0, 4), dtype=np.int64),
                           view_quality=np.empty((0, 4)), reliability=np.empty(0))
        ledger = collector.observe(replace(frame, evidence=evidence))
        self.assertEqual(ledger.status, 'empty')
        self.assertEqual(ledger.prediction_counts[0], 0)
        self.assertFalse(collector.retained)

    def test_ineligible_when_repeated_identity_changes_evidence(self):
        """Given a committed identity, when different data reuse it, then fail honestly."""
        collector = Figure7Collector(self.root / 'run', provenance())
        collector.observe(observation('same'))
        with self.assertRaises(Figure7Error):
            collector.observe(observation('same', False))
