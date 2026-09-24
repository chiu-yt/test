from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from figure7_fixtures import observation, provenance, with_reliability
from pcdet.utils.figure7_runtime import Figure7Collector
from tools.figure7_utils.loading import Figure7LoadError, _candidate, _validate_arrays, load_capture


def completed_capture(root: Path) -> Path:
    capture = root / 'capture'
    collector = Figure7Collector(capture, provenance())
    case_b = with_reliability(observation('case-b'), .9)
    case_b = replace(case_b, evidence=replace(
        case_b.evidence, view_quality=np.full((1, 4), .9),
    ))
    case_a = with_reliability(observation('case-a', False), .2)
    case_a = replace(case_a, evidence=replace(
        case_a.evidence, view_quality=np.array([[.8, 0., 0., 0.]]),
    ))
    collector.observe(case_b)
    collector.observe(case_a)
    collector.finalize()
    return capture


class TestFigure7RendererLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_record_ids_are_exact_hex_digests(self) -> None:
        """Given unsafe IDs, when candidates parse, then paths cannot be requested."""
        capture = completed_capture(self.root)
        raw = json.loads((capture / 'selection.json').read_text())['case_a']
        for record_id in ('../escape', '/tmp/escape', 'a' * 63, 'z' * 64, 'a' * 65):
            with self.subTest(record_id=record_id), self.assertRaises(Figure7LoadError):
                _candidate(dict(raw, record_id=record_id))

    def test_external_symlinks_are_rejected(self) -> None:
        """Given external record components, when loaded, then containment fails."""
        for component in ('records', 'directory', 'metadata.json', 'arrays.npz', 'checksums.json'):
            with self.subTest(component=component), TemporaryDirectory() as temporary:
                root = Path(temporary)
                capture = completed_capture(root)
                selected = json.loads((capture / 'selection.json').read_text())['case_a']
                directory = capture / 'records' / selected['record_id']
                path = capture / 'records' if component == 'records' else directory
                if component not in ('records', 'directory'):
                    path = directory / component
                outside = root / 'external'
                path.rename(outside)
                path.symlink_to(outside, target_is_directory=outside.is_dir())
                with self.assertRaises(Figure7LoadError):
                    load_capture(capture)

    def test_unchecksummed_ledger_cannot_override_observation(self) -> None:
        """Given modified ledger evidence, when loaded, then checksummed truth wins."""
        changes = (('accepted', 'false'), ('rescued', 0), ('matches', [-1] * 4),
                   ('qualities', [.123] * 4), ('class_name', 'car'), ('score', .8),
                   ('matches', [0]), ('qualities', [1]), ('index', True))
        for field, value in changes:
            with self.subTest(field=field, value=value), TemporaryDirectory() as temporary:
                capture = completed_capture(Path(temporary))
                path = capture / 'ledger.jsonl'
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]['references'][0][field] = value
                path.write_text('\n'.join(json.dumps(row) for row in rows))
                with self.assertRaises(Figure7LoadError):
                    load_capture(capture)

    def test_full_identity_and_case_values_are_cross_checked(self) -> None:
        """Given changed occurrence or case values, when loaded, then mismatch fails."""
        for target in ('ledger', 'retained', 'case'):
            with self.subTest(target=target), TemporaryDirectory() as temporary:
                capture = completed_capture(Path(temporary))
                if target == 'ledger':
                    path = capture / 'ledger.jsonl'
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]['identity']['samples_seen'] += 1
                    path.write_text('\n'.join(json.dumps(row) for row in rows))
                else:
                    path = capture / 'selection.json'
                    selection = json.loads(path.read_text())
                    candidate = selection['retained'][0] if target == 'retained' else selection['case_a']
                    candidate['model_step'] += 1
                    path.write_text(json.dumps(selection))
                with self.assertRaises(Figure7LoadError):
                    load_capture(capture)

    def test_array_semantics_reject_invalid_selected_evidence(self) -> None:
        """Given inconsistent numeric arrays, when validated, then selection fails."""
        record = load_capture(completed_capture(self.root)).case_b
        changes = (('reference_scores', np.array([np.nan])),
                   ('reliability', np.array([np.nan])),
                   ('reference_mask', np.array([False])),
                   ('reference_mask', np.array([1])),
                   ('view_0_mask', np.array([False])),
                   ('view_0_labels', np.array([1])),
                   ('reference_labels', np.array([1])),
                   ('reference_labels', np.array([8.5])),
                   ('reference_labels', np.array([True])),
                   ('reference_points', np.array([[np.inf, 0, 0]])),
                   ('reference_transform', np.zeros((4, 4))),
                   ('match_indices', np.array([[-2, 0, 0, 0]])),
                   ('match_indices', np.array([[.5, 0, 0, 0]])),
                   ('view_quality', np.array([[np.nan, 1, 1, 1]])))
        for name, value in changes:
            with self.subTest(name=name), self.assertRaises(Figure7LoadError):
                _validate_arrays(dict(record.arrays, **{name: value}), record.candidate)
        with self.assertRaises(Figure7LoadError):
            _validate_arrays(record.arrays, replace(record.candidate, reference_index=-1))

    def test_checksummed_rejected_context_loads_and_renders(self) -> None:
        capture = self.root / 'context'
        collector = Figure7Collector(capture, provenance())
        for token, stable, reliability in (('case-b', True, .9), ('case-a', False, .2)):
            frame = with_reliability(observation(token, stable), reliability)
            evidence = frame.evidence
            assert evidence is not None
            predictions = tuple(replace(
                prediction, boxes=np.concatenate((np.full((1, 7), np.nan), prediction.boxes)),
                labels=np.concatenate(([np.nan], prediction.labels)),
                scores=np.concatenate(([np.nan], prediction.scores)),
            ) for prediction in (evidence.reference_prediction, *evidence.view_predictions))
            matches = np.where(evidence.match_indices >= 0, evidence.match_indices + 1, -1)
            frame = replace(frame, evidence=replace(
                evidence, reference_prediction=predictions[0], view_predictions=predictions[1:],
                reference_mask=np.array([False, True]), reference_rescue_mask=np.array([False, False]),
                view_masks=tuple(np.concatenate(([False], mask)) for mask in evidence.view_masks),
                view_rescue_masks=tuple(np.concatenate(([False], mask)) for mask in evidence.view_rescue_masks),
                match_indices=np.concatenate((np.full((1, 4), -1), matches)),
                view_quality=np.concatenate((np.zeros((1, 4)), evidence.view_quality)),
                reliability=np.array([0., reliability]),
                coverage=np.concatenate(([0.], evidence.coverage)),
                support=np.concatenate(([0.], evidence.support)),
            ))
            collector.observe(frame)
        collector.finalize()
        bundle = load_capture(capture)
        self.assertEqual(bundle.case_b.candidate.reference_index, 1)
        self.assertTrue(np.isnan(bundle.case_b.arrays['reference_boxes'][0]).all())
        from tools.figure7_utils.rendering import render_plate
        import matplotlib.pyplot as plt
        figure = render_plate(bundle, (-20., -20., 20., 20.))
        try:
            figure.canvas.draw()
            self.assertEqual(len(figure.axes[7].patches), 2)
        finally:
            plt.close(figure)

    def test_schema_boolean_is_not_an_integer_version(self) -> None:
        capture = completed_capture(self.root)
        path = capture / 'run.json'
        run = json.loads(path.read_text())
        run['schema_version'] = True
        path.write_text(json.dumps(run))
        with self.assertRaises(Figure7LoadError):
            load_capture(capture)

    def test_rejected_nonfinite_context_preserves_original_rows(self) -> None:
        """Given rejected NaN context, when validated, then only accepted geometry is strict."""
        record = load_capture(completed_capture(self.root)).case_b
        arrays = dict(record.arrays)
        for prefix in ('reference', 'view_0', 'view_1', 'view_2', 'view_3'):
            for suffix in ('boxes', 'labels', 'scores', 'mask', 'rescue_mask'):
                value = arrays[prefix + '_' + suffix]
                context = np.array(value, copy=True)
                if suffix == 'boxes':
                    context[:] = np.nan
                if suffix in ('mask', 'rescue_mask'):
                    context[:] = False
                arrays[prefix + '_' + suffix] = np.concatenate((context, value))
        arrays['match_indices'] = np.array([[-1] * 4, [1] * 4])
        arrays['view_quality'] = np.array([[0.] * 4, [1.] * 4])
        arrays['reliability'] = np.array([0., .9])
        _validate_arrays(arrays, replace(record.candidate, reference_index=1))
        self.assertTrue(np.isnan(arrays['reference_boxes'][0]).all())
        for prefix in ('reference', 'view_0'):
            damaged = dict(arrays)
            damaged[prefix + '_boxes'] = arrays[prefix + '_boxes'].copy()
            damaged[prefix + '_boxes'][1, 0] = np.nan
            with self.subTest(prefix=prefix), self.assertRaises(Figure7LoadError):
                _validate_arrays(damaged, replace(record.candidate, reference_index=1))

    def test_loader_uses_only_selected_checksummed_complete_records(self) -> None:
        """Given a complete capture, when loaded, then selected records and ledger are exact."""
        bundle = load_capture(completed_capture(self.root))
        self.assertEqual((bundle.case_a.candidate.identity.token,
                          bundle.case_b.candidate.identity.token), ('case-a', 'case-b'))
        self.assertEqual(len(bundle.ledger), 2)
        self.assertEqual(len(bundle.retained), 2)
        self.assertEqual(bundle.case_a.arrays['reference_points'].shape, (3, 4))

    def test_loader_rejects_invalid_capture(self) -> None:
        """Given invalid evidence, when loaded, then publication cannot start."""
        for damage in ('selection', 'checksum', 'schema'):
            with self.subTest(damage=damage), TemporaryDirectory() as directory:
                capture = completed_capture(Path(directory))
                if damage == 'selection':
                    selection = json.loads((capture / 'selection.json').read_text())
                    selection['status'] = 'incomplete'
                    (capture / 'selection.json').write_text(json.dumps(selection))
                elif damage == 'checksum':
                    selection = json.loads((capture / 'selection.json').read_text())
                    record_id = selection['case_a']['record_id']
                    with (capture / 'records' / record_id / 'arrays.npz').open('ab') as stream:
                        stream.write(b'corrupt')
                else:
                    run = json.loads((capture / 'run.json').read_text())
                    run['schema_version'] = 999
                    (capture / 'run.json').write_text(json.dumps(run))
                with self.assertRaises(Figure7LoadError):
                    load_capture(capture)

    def test_inverse_alignment_matches_k4_transform_convention(self) -> None:
        """Given a translated view, when loaded, then boxes and points return to reference."""
        selected = load_capture(completed_capture(self.root)).case_a
        matrix = np.eye(4)
        matrix[0, 3] = 4.0
        points = np.array([[14., 2., 0., 1.]])
        boxes = np.array([[14., 2., 0., 2., 4., 2., 0.]])
        np.testing.assert_allclose(selected.inverse_points(points, matrix)[0, :3], [10., 2., 0.])
        np.testing.assert_allclose(selected.inverse_boxes(boxes, matrix)[0, :3], [10., 2., 0.])
