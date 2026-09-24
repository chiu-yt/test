import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from figure7_fixtures import observation, provenance
from test_spcra_k4_runtime import ForwardFailure, load_contract_module
from pcdet.utils.figure7_runtime import Figure7Collector
from pcdet.utils.figure7_schema import Figure7Error


def stream(tmp_path):
    module = importlib.import_module('pcdet.utils.figure7_stream')
    return module.Figure7StreamCapture(Figure7Collector(tmp_path / 'capture', provenance()))


def check_occurrences_when_one_forward_fails_are_all_recorded(tmp_path):
    capture = stream(tmp_path)
    batch = {'metadata': [{'token': 'a'}, {'token': 'b'}], 'frame_id': ['a', 'b']}
    capture.begin(batch, (2, 7, 14, 2))
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'forward failed'):
        with capture.transaction():
            raise ForwardFailure('forward failed')
    rows = [json.loads(line) for line in (tmp_path / 'capture/ledger.jsonl').read_text().splitlines()]
    assert [row['status'] for row in rows] == ['failed', 'failed']
    assert [row['model_step'] for row in rows] == [7, 7]
    assert capture.identities == ()


def check_actual_points_when_view_inputs_differ_are_owned_and_released(tmp_path):
    capture = stream(tmp_path)
    frame = observation('stable')
    capture.begin({'metadata': [{'token': 'stable'}], 'frame_id': ['stable']}, (0, 3, 6, 1))
    points = np.array([[0., 10., 0., 0., 1.]])
    with capture.transaction():
        capture.reference(points, np.eye(4)[None])
        for index in range(4):
            actual = points.copy()
            actual[:, 1] += index
            capture.view(actual)
            actual[:] = 99
        capture.evidence(frame.evidence)
    rows = [json.loads(line) for line in (tmp_path / 'capture/ledger.jsonl').read_text().splitlines()]
    assert rows[0]['point_counts'] == [1] * 5
    assert len(set(rows[0]['point_checksums'])) == 4
    assert rows[0]['model_step'] == 3
    assert capture.identities == ()


def check_empty_occurrence_when_predictions_empty_is_not_dropped(tmp_path):
    capture = stream(tmp_path)
    capture.begin({'metadata': [{'token': 'empty'}], 'frame_id': ['empty']}, (0, 0, 0, 1))
    with capture.transaction():
        pass
    rows = [json.loads(line) for line in (tmp_path / 'capture/ledger.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['status'] == 'incomplete'


class TestFigure7Stream(unittest.TestCase):
    def test_occurrences_when_points_malformed_preserve_all_observations(self):
        for malformed in range(3):
            for body_failure in (False, True):
                with self.subTest(index=malformed, body_failure=body_failure), TemporaryDirectory() as directory:
                    capture = stream(Path(directory))
                    tokens = ['first', 'middle', 'last']
                    capture.begin({'metadata': [{'token': token} for token in tokens],
                                   'frame_id': tokens}, (0, 7, 0, 3))
                    points = np.array([[index, 10., 0., 0.] for index in range(3)])
                    transforms = np.repeat(np.eye(4)[None], 3, axis=0)
                    transforms[malformed, 0, 0] = np.nan
                    failure = Figure7Error('body failed')
                    with self.assertRaises(Figure7Error) if body_failure else nullcontext() as raised:
                        with capture.transaction():
                            capture.reference(points, transforms)
                            for _ in range(4):
                                capture.view(points)
                            for token in tokens:
                                capture.evidence(observation(token).evidence)
                            if body_failure:
                                raise failure
                    if body_failure:
                        assert raised is not None
                        self.assertIs(raised.exception, failure)
                    rows = [json.loads(line) for line in
                            (capture.collector.output / 'ledger.jsonl').read_text().splitlines()]
                    self.assertEqual([row['identity']['token'] for row in rows], tokens)
                    self.assertEqual(rows[malformed]['status'], 'failed')
                    self.assertIn('Figure7Error', rows[malformed]['detail'])
                    self.assertEqual(rows[malformed]['point_counts'], [])
                    self.assertTrue(rows[malformed]['completion'][0])
                    for index, row in enumerate(rows):
                        self.assertEqual(bool(row['point_counts']), index != malformed)
                        self.assertEqual('body failed' in row['detail'], body_failure)
                    self.assertEqual((capture.identities, capture._reference, capture._transforms), ((), (), ()))
                    self.assertEqual((capture._views, capture._evidence), ([], []))

    def test_occurrences_when_transform_missing_keep_available_evidence(self):
        with TemporaryDirectory() as directory:
            capture = stream(Path(directory))
            capture.begin({'metadata': [{'token': 'a'}, {'token': 'b'}],
                           'frame_id': ['a', 'b']}, (0, 0, 0, 2))
            with capture.transaction():
                points = np.array([[0., 10., 0., 0.], [1., 10., 0., 0.]])
                capture.reference(points, np.eye(4)[None])
                for _ in range(4):
                    capture.view(points)
                capture.evidence(observation('a').evidence)
            rows = [json.loads(line) for line in
                    (capture.collector.output / 'ledger.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(rows[0]['completion']))
            self.assertEqual(rows[1]['status'], 'failed')
            self.assertIn('IndexError', rows[1]['detail'])
            self.assertEqual(rows[1]['prediction_counts'], [])
            self.assertEqual(capture.identities, ())

    def test_storage_failure_when_finishing_propagates_and_clears_state(self):
        with TemporaryDirectory() as directory:
            capture = stream(Path(directory))
            capture.begin({'metadata': [{'token': 'a'}], 'frame_id': ['a']}, (0, 0, 0, 1))
            (capture.collector.output / 'ledger.jsonl').mkdir()
            with self.assertRaises(OSError):
                with capture.transaction():
                    pass
            self.assertEqual(capture.identities, ())

    def test_malformed_token_when_beginning_is_rejected(self):
        with TemporaryDirectory() as directory:
            capture = stream(Path(directory))
            with self.assertRaises(ValueError):
                capture.begin({'metadata': [{'token': None}], 'frame_id': ['frame']}, (0, 0, 0, 1))

    def test_complete_empty_evidence_when_observed_is_recorded_as_empty(self):
        core = load_contract_module('spcra_k4_core')
        codec = load_contract_module('spcra_k4_evidence')
        seed = load_contract_module('spcra_k4_seeding')
        prediction = core.Predictions(np.empty((0, 7)), np.empty(0, dtype=int), np.empty(0))
        views = (prediction,) * 4
        samples = seed.ViewSamples(tuple(np.empty(0, dtype=int) for _ in range(4)),
                                   tuple(np.eye(4) for _ in range(4)),
                                   ('a', 'b', 'c', 'd'), (0, 0, 0, 0), 'test-law')
        evidence = codec.build_k4_evidence(codec.K4EvidenceInput(prediction, views, samples),
                                          core.compute_k4_reliability(prediction, views, core.K4Policy()))
        with TemporaryDirectory() as directory:
            capture = stream(Path(directory))
            capture.begin({'metadata': [{'token': 'empty'}], 'frame_id': ['frame']}, (0, 0, 0, 1))
            with capture.transaction():
                capture.reference(np.empty((0, 4)), np.eye(4)[None])
                for _ in range(4):
                    capture.view(np.empty((0, 4)))
                capture.evidence(evidence)
            ledger = json.loads((Path(directory) / 'capture/ledger.jsonl').read_text())
            self.assertEqual(ledger['status'], 'empty')
            self.assertEqual(ledger['prediction_counts'], [0] * 5)

    def test_stream_transactions(self):
        for case in (check_occurrences_when_one_forward_fails_are_all_recorded,
                     check_actual_points_when_view_inputs_differ_are_owned_and_released,
                     check_empty_occurrence_when_predictions_empty_is_not_dropped):
            with self.subTest(case=case.__name__), TemporaryDirectory() as directory:
                case(Path(directory))
