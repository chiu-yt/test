import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from pcdet.utils.figure6_schema import (
    ArtifactError, CaptureRecord, FIXED_TOKENS, Occurrence, StageCapture,
    StageState, StageStatus, selected_batch_indices,
)
from pcdet.utils.figure6_artifacts import (
    load_record, scan_records, select_occurrences, write_record,
)


def occurrence():
    return Occurrence(FIXED_TOKENS[0], 'frame-7', 0, 7, 28, 0, 4, 1, 3)


def record(identity=None, values=None):
    return CaptureRecord(identity or occurrence(), {'sparsity': 'density_dec_global'}, {
        'reliability': StageCapture(
            StageStatus(StageState.COMPLETE, 'teacher'),
            {'scores': np.array([0.25, 0.5]) if values is None else values},
        ),
        'rgplm': StageCapture(StageStatus(StageState.OBSERVED_EMPTY, 'teacher'), {
            'boxes': np.empty((0, 7), dtype=np.float32),
        }),
        'sgdfa': StageCapture(StageStatus(StageState.MISSING, 'student'), {}),
        'finaldet': StageCapture(StageStatus(StageState.FAILED, 'student', 'nonfinite'), {}),
    })


def test_selection_when_batch_has_mixed_and_repeated_tokens():
    """Given mixed tokens, when selected, then preserve every matching index."""
    tokens = ('other', FIXED_TOKENS[1], FIXED_TOKENS[0], FIXED_TOKENS[1])
    assert selected_batch_indices(tokens) == (1, 2, 3)
    assert selected_batch_indices(tokens, (FIXED_TOKENS[0],)) == (2,)
    assert selected_batch_indices(('other',)) == ()


def test_input_copies_when_source_arrays_are_mutated(tmp_path):
    """Given source views, when mutated after capture, then saved values stay original."""
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    captured = record(values=source[:, ::2])
    source[:] = -1
    loaded = load_record(write_record(tmp_path, captured))
    np.testing.assert_array_equal(loaded.stages['reliability'].arrays['scores'],
                                  [[0, 2], [4, 6], [8, 10]])
    assert not np.shares_memory(captured.stages['reliability'].arrays['scores'], source)


def test_statuses_when_round_tripped_remain_distinct(tmp_path):
    """Given four statuses, when persisted, then empty is not missing or failed."""
    loaded = load_record(write_record(tmp_path, record()))
    assert tuple(stage.status.state for stage in loaded.stages.values()) == (
        StageState.FAILED, StageState.COMPLETE, StageState.OBSERVED_EMPTY, StageState.MISSING,
    )
    assert loaded.stages['rgplm'].arrays['boxes'].shape == (0, 7)
    assert not loaded.stages['sgdfa'].arrays
    assert loaded.stages['finaldet'].status.detail == 'nonfinite'
    assert loaded.identity == occurrence()
    assert loaded.protocol == {'sparsity': 'density_dec_global'}


def test_metadata_when_written_has_array_integrity_fields(tmp_path):
    """Given a record, when published, then every array is described by shape/dtype/hash."""
    path = write_record(tmp_path, record())
    metadata = json.loads((path / 'metadata.json').read_text())
    assert metadata['schema_version'] == 1
    assert path.parent.parent.name == 'rank_00000'
    assert len(metadata['arrays']) == 2
    assert all(set(item) == {'shape', 'dtype', 'sha256'}
               for item in metadata['arrays'].values())
    assert all(len(item['sha256']) == 64 for item in metadata['arrays'].values())


def test_duplicates_when_scanned_preserve_alternatives_and_order(tmp_path):
    """Given repeated rank/iteration occurrences, when selected, then earliest wins."""
    identities = (
        replace(occurrence(), epoch=1, accumulated_iter_before=0),
        replace(occurrence(), accumulated_iter_before=9),
        replace(occurrence(), global_rank=2),
        replace(occurrence(), batch_index=2),
        occurrence(),
    )
    for identity in identities:
        write_record(tmp_path, record(identity))
    catalog = scan_records(tmp_path)
    selected = select_occurrences(tuple(reversed(catalog.records)))
    choice = selected[FIXED_TOKENS[0]]
    assert choice.selected.record.identity == identities[-1]
    assert tuple(item.record.identity for item in choice.alternatives) == identities[-2::-1]
    assert not catalog.issues
    assert len(catalog.records) == 5
    with pytest.raises(FileExistsError):
        write_record(tmp_path, record())


@pytest.mark.parametrize('damage', ['marker', 'archive', 'json', 'zip', 'version'])
def test_invalid_records_when_scanned_are_reported_not_selected(tmp_path, damage):
    """Given incomplete/corrupt storage, when scanned, then report the bad record."""
    path = write_record(tmp_path, record())
    if damage == 'marker':
        (path / 'metadata.json').unlink()
    elif damage == 'archive':
        (path / 'tensors.npz').unlink()
    elif damage == 'json':
        (path / 'metadata.json').write_text('{')
    elif damage == 'zip':
        (path / 'tensors.npz').write_bytes(b'broken')
    else:
        metadata = json.loads((path / 'metadata.json').read_text())
        metadata['schema_version'] = 999
        (path / 'metadata.json').write_text(json.dumps(metadata))
    catalog = scan_records(tmp_path)
    assert catalog.records == ()
    assert len(catalog.issues) == 1
    assert catalog.issues[0].path == path
    with pytest.raises(ArtifactError):
        load_record(path)


@pytest.mark.parametrize('field,value', [('sha256', '0' * 64), ('shape', [99]), ('dtype', '<i8')])
def test_integrity_when_metadata_tampered_rejects_record(tmp_path, field, value):
    """Given altered metadata, when loaded, then verify every integrity field."""
    path = write_record(tmp_path, record())
    metadata = json.loads((path / 'metadata.json').read_text())
    next(iter(metadata['arrays'].values()))[field] = value
    (path / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ArtifactError):
        load_record(path)


def test_checksum_when_array_bytes_change_rejects_record(tmp_path):
    """Given valid NPZ with changed values, when loaded, then checksum fails."""
    path = write_record(tmp_path, record())
    with np.load(path / 'tensors.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays['reliability/scores'][0] = 42
    np.savez_compressed(path / 'tensors.npz', **arrays)
    with pytest.raises(ArtifactError, match='checksum'):
        load_record(path)


@pytest.mark.parametrize('dtype', ['O', [('field', 'i4')], 'datetime64[ns]'])
def test_unsafe_arrays_when_captured_are_rejected(dtype):
    """Given unsupported dtypes, when captured, then reject before writing."""
    with pytest.raises(ArtifactError):
        record(values=np.empty((0,), dtype=dtype))


def test_loader_when_called_disables_pickle(tmp_path, monkeypatch):
    """Given a safe archive, when loading, then explicitly disable pickle."""
    path = write_record(tmp_path, record(values=np.array(['car', 'bus'])))
    original = np.load
    calls = []

    def checked_load(file, allow_pickle):
        calls.append(allow_pickle)
        return original(file, allow_pickle=allow_pickle)

    monkeypatch.setattr(np, 'load', checked_load)
    loaded = load_record(path)
    assert calls == [False]
    assert loaded.stages['reliability'].arrays['scores'].tolist() == ['car', 'bus']


def test_atomic_publication_when_metadata_replace_fails(tmp_path, monkeypatch):
    """Given interrupted publication, when metadata fails, then no complete record appears."""
    original = Path.replace
    published = []

    def interrupted_replace(source, target):
        assert source.parent == target.parent
        published.append(target.name)
        if target.name == 'metadata.json':
            assert (target.parent / 'tensors.npz').is_file()
            raise OSError('simulated interruption')
        return original(source, target)

    monkeypatch.setattr(Path, 'replace', interrupted_replace)
    with pytest.raises(OSError, match='interruption'):
        write_record(tmp_path, record())
    assert published == ['tensors.npz', 'metadata.json']
    assert not scan_records(tmp_path).records
    assert not tuple(tmp_path.rglob('*.tmp'))


def test_identity_and_status_when_mutated_are_frozen():
    """Given typed identity/status, when reassigned, then mutation is refused."""
    identity = occurrence()
    status = StageStatus(StageState.MISSING, 'student')
    with pytest.raises(FrozenInstanceError):
        identity.epoch = 3
    with pytest.raises(FrozenInstanceError):
        status.owner = 'teacher'


def test_import_when_in_fresh_process_is_dependency_light():
    """Given a fresh process, when importing the reader, then no ML/render stack loads."""
    result = subprocess.run([sys.executable, '-c',
        'import sys; import pcdet.utils.figure6_artifacts; '
        'assert not any(name in sys.modules for name in '
        '("torch", "matplotlib", "pcdet.models"))'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
