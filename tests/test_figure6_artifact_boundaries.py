import json
from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pytest

from pcdet.utils.figure6_artifacts import load_record, scan_records, select_occurrences, write_record
from pcdet.utils.figure6_schema import (
    ArtifactError, CaptureRecord, FIXED_TOKENS, Occurrence, StageCapture, StageState, StageStatus,
)


def capture():
    identity = Occurrence(FIXED_TOKENS[0], 'frame', 0, 0, 0, 0, 2, 0, 2)
    return CaptureRecord(identity, {'protocol': 'test'}, {
        'stage': StageCapture(StageStatus(StageState.COMPLETE, 'teacher'), {
            'values': np.array([1, 2], dtype=np.int32),
        }),
    })


def test_npy_when_disguised_as_npz_is_reported(tmp_path):
    """Given the wrong file format, when scanned, then report a corrupt occurrence."""
    path = write_record(tmp_path, capture())
    with (path / 'tensors.npz').open('wb') as stream:
        np.save(stream, np.arange(3), allow_pickle=False)
    catalog = scan_records(tmp_path)
    assert not catalog.records
    assert len(catalog.issues) == 1


def test_archive_when_member_is_not_an_array_is_reported(tmp_path):
    """Given non-array ZIP content, when scanned, then report a corrupt occurrence."""
    path = write_record(tmp_path, capture())
    with ZipFile(path / 'tensors.npz', 'w') as archive:
        archive.writestr('stage/values.npy', b'not an array')
    catalog = scan_records(tmp_path)
    assert not catalog.records
    assert len(catalog.issues) == 1


@pytest.mark.parametrize('state,arrays', [
    (StageState.MISSING, {'values': np.empty(0)}),
    (StageState.FAILED, {'values': np.empty(0)}),
    (StageState.OBSERVED_EMPTY, {}),
    (StageState.OBSERVED_EMPTY, {'values': np.ones(1)}),
    (StageState.COMPLETE, {'values': np.empty(0)}),
])
def test_stage_when_state_contradicts_arrays_is_rejected(state, arrays):
    """Given contradictory state/payload, when constructed, then reject ambiguity."""
    with pytest.raises(ArtifactError):
        StageCapture(StageStatus(state, 'teacher', 'reason'), arrays)


@pytest.mark.parametrize('field,value', [
    ('global_rank', 2), ('world_size', 0), ('epoch', -1), ('epoch', True),
    ('batch_index', 2), ('batch_size', 0), ('samples_seen', 0.5),
])
def test_identity_when_invalid_is_rejected(field, value):
    """Given invalid rank/index/counters, when constructed, then reject identity."""
    with pytest.raises(ArtifactError):
        replace(capture().identity, **{field: value})


@pytest.mark.parametrize('field,value', [
    ('identity', None), ('stages', []), ('arrays', None), ('schema_version', True),
    ('protocol', {'key': 5}),
])
def test_metadata_when_wrongly_typed_is_rejected(tmp_path, field, value):
    """Given malformed JSON types, when loaded, then fail at the parsing boundary."""
    path = write_record(tmp_path, capture())
    metadata = json.loads((path / 'metadata.json').read_text())
    metadata[field] = value
    (path / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ArtifactError):
        load_record(path)


@pytest.mark.parametrize('arrays', [{}, {'extra': np.zeros(1)}])
def test_archive_when_inventory_differs_is_rejected(tmp_path, arrays):
    """Given missing/extra NPZ entries, when loaded, then reject inventory mismatch."""
    path = write_record(tmp_path, capture())
    np.savez_compressed(path / 'tensors.npz', **arrays)
    with pytest.raises(ArtifactError, match='inventory'):
        load_record(path)


def test_selection_when_multiple_tokens_repeat_retains_each_token(tmp_path):
    """Given repeated mixed tokens, when selected, then no token overwrites another."""
    base = capture()
    for token in reversed(FIXED_TOKENS):
        for rank in (1, 0):
            write_record(tmp_path, replace(base, identity=replace(
                base.identity, token=token, global_rank=rank,
            )))
    selected = select_occurrences(scan_records(tmp_path).records)
    assert tuple(selected) == tuple(sorted(FIXED_TOKENS))
    assert all(choice.selected.record.identity.global_rank == 0 for choice in selected.values())
    assert all(len(choice.alternatives) == 1 for choice in selected.values())


def test_snapshot_when_caller_changes_mapping_is_independent(tmp_path):
    """Given caller-owned maps, when cleared, then the captured snapshot survives."""
    arrays = {'values': np.array([3])}
    stage = StageCapture(StageStatus(StageState.COMPLETE, 'teacher'), arrays)
    stages = {'stage': stage}
    protocol = {'protocol': 'test'}
    snapshot = CaptureRecord(capture().identity, protocol, stages)
    arrays.clear()
    stages.clear()
    protocol.clear()
    loaded = load_record(write_record(tmp_path, snapshot))
    assert loaded.stages['stage'].arrays['values'].tolist() == [3]
    assert loaded.protocol == {'protocol': 'test'}


@pytest.mark.parametrize('values', [np.array(True), np.array(1 + 2j), np.array([b'car'])])
def test_safe_scalar_and_string_arrays_when_persisted_round_trip(tmp_path, values):
    """Given safe scalar/byte/complex data, when persisted, then retain shape and dtype."""
    stage = StageCapture(StageStatus(StageState.COMPLETE, 'teacher'), {'values': values})
    path = write_record(tmp_path, replace(capture(), stages={'stage': stage}))
    restored = load_record(path).stages['stage'].arrays['values']
    np.testing.assert_array_equal(restored, values)
    assert restored.dtype == values.dtype
