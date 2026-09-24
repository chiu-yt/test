import json
from pathlib import Path
import re
from typing import Mapping, Tuple

import numpy as np

from pcdet.utils.figure6_schema import Occurrence
from pcdet.utils.figure7_artifacts import verify_record
from pcdet.utils.figure7_schema import Candidate, RankKey, SCHEMA_VERSION

from .contracts import (
    CaptureBundle, Figure7LoadError, JsonValue, LedgerEntry, LedgerReference, ProvenanceInfo, SelectedRecord,
)
from .validation import validate_arrays as _validate_arrays, validate_observation


def _record_id(value: JsonValue) -> str:
    result = _text(value, 'record_id')
    if re.fullmatch(r'[0-9a-fA-F]{64}', result) is None:
        raise Figure7LoadError('record_id must be exactly 64 hexadecimal characters')
    return result


def _boolean(value: JsonValue, context: str) -> bool:
    if not isinstance(value, bool):
        raise Figure7LoadError(context + ' must be a boolean')
    return value


def _contained(path: Path, parent: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(parent) or resolved == parent:
        raise Figure7LoadError(str(path) + ' escapes its containing directory')
    return resolved


def _mapping(value: JsonValue, context: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise Figure7LoadError(context + ' must be a JSON mapping')
    return value


def _sequence(value: JsonValue, context: str) -> Tuple[JsonValue, ...]:
    if not isinstance(value, list):
        raise Figure7LoadError(context + ' must be a JSON list')
    return tuple(value)


def _text(value: JsonValue, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise Figure7LoadError(context + ' must be nonempty text')
    return value


def _integer(value: JsonValue, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Figure7LoadError(context + ' must be an integer')
    return value


def _number(value: JsonValue, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Figure7LoadError(context + ' must be numeric')
    result = float(value)
    if not np.isfinite(result):
        raise Figure7LoadError(context + ' must be finite')
    return result


def _json(path: Path) -> Mapping[str, JsonValue]:
    try:
        return _mapping(json.loads(path.read_text(encoding='utf-8')), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise Figure7LoadError('%s: %s' % (path, error)) from error


def _occurrence(raw: JsonValue) -> Occurrence:
    value = _mapping(raw, 'candidate identity')
    return Occurrence(
        _text(value.get('token'), 'token'), _text(value.get('frame_id'), 'frame_id'),
        _integer(value.get('epoch'), 'epoch'),
        _integer(value.get('accumulated_iter_before'), 'accumulated_iter_before'),
        _integer(value.get('samples_seen'), 'samples_seen'),
        _integer(value.get('global_rank'), 'global_rank'),
        _integer(value.get('world_size'), 'world_size'),
        _integer(value.get('batch_index'), 'batch_index'),
        _integer(value.get('batch_size'), 'batch_size'),
    )


def _candidate(raw: JsonValue) -> Candidate:
    value = _mapping(raw, 'selected candidate')
    pool = _text(value.get('pool'), 'pool')
    if pool not in ('stable', 'variable'):
        raise Figure7LoadError('candidate pool is invalid')
    raw_rank = _sequence(value.get('rank_key'), 'rank_key')
    if len(raw_rank) != 10:
        raise Figure7LoadError('rank_key must contain ten values')
    rank_key: RankKey = (
        _integer(raw_rank[0], 'rank priority'), _number(raw_rank[1], 'rank score'),
        _number(raw_rank[2], 'rank reliability'), _number(raw_rank[3], 'rank quality'),
        _text(raw_rank[4], 'rank token'), _integer(raw_rank[5], 'rank epoch'),
        _integer(raw_rank[6], 'rank model step'), _integer(raw_rank[7], 'rank global rank'),
        _integer(raw_rank[8], 'rank batch index'), _integer(raw_rank[9], 'rank proposal'),
    )
    return Candidate(
        _record_id(value.get('record_id')), _occurrence(value.get('identity')),
        _integer(value.get('model_step'), 'model_step'),
        _integer(value.get('reference_index'), 'reference_index'),
        _text(value.get('class_name'), 'class_name'), _number(value.get('score'), 'score'),
        _number(value.get('reliability'), 'reliability'), pool, rank_key,
    )


def _reference(raw: JsonValue) -> LedgerReference:
    value = _mapping(raw, 'ledger reference')
    score = value.get('score')
    if any(len(_sequence(value.get(name), name)) != 4 for name in ('matches', 'qualities')):
        raise Figure7LoadError('each reference must have exactly four matches and qualities')
    return LedgerReference(
        _integer(value.get('index'), 'reference index'), _text(value.get('class_name'), 'class'),
        None if score is None else _number(score, 'score'), _boolean(value.get('accepted'), 'accepted'),
        _boolean(value.get('rescued'), 'rescued'),
        tuple(_integer(item, 'match') for item in _sequence(value.get('matches'), 'matches')),
        tuple(_number(item, 'quality') for item in _sequence(value.get('qualities'), 'qualities')),
        _number(value.get('reliability'), 'reliability'),
        None if value.get('stable_key') is None else _sequence(value.get('stable_key'), 'stable_key'),
        None if value.get('variable_key') is None else _sequence(value.get('variable_key'), 'variable_key'),
    )


def _ledger_entry(raw: JsonValue) -> LedgerEntry:
    value = _mapping(raw, 'ledger row')
    if _integer(value.get('schema_version'), 'ledger schema_version') != SCHEMA_VERSION:
        raise Figure7LoadError('unsupported Figure 7 ledger schema')
    identity = _mapping(value.get('identity'), 'ledger identity')
    return LedgerEntry(
        _record_id(value.get('record_id')), _text(identity.get('token'), 'token'),
        _text(identity.get('frame_id'), 'frame_id'), _integer(identity.get('epoch'), 'epoch'),
        _integer(value.get('model_step'), 'model_step'),
        _integer(identity.get('global_rank'), 'global_rank'),
        _integer(identity.get('batch_index'), 'batch_index'), _text(value.get('status'), 'status'),
        tuple(_text(item, 'admitted pool') for item in _sequence(
            value.get('admitted_pools'), 'admitted_pools')),
        tuple(_reference(item) for item in _sequence(value.get('references'), 'references')),
        _occurrence(value.get('identity')), value,
    )


def _load_ledger(path: Path) -> Tuple[LedgerEntry, ...]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
        return tuple(_ledger_entry(json.loads(line)) for line in lines)
    except (OSError, json.JSONDecodeError) as error:
        raise Figure7LoadError('%s: %s' % (path, error)) from error


def _selected(capture: Path, candidate: Candidate, ledger: Tuple[LedgerEntry, ...]) -> SelectedRecord:
    directory = capture / 'records' / candidate.record_id
    try:
        root = _contained(capture / 'records', capture.resolve(strict=True))
        directory = _contained(root / candidate.record_id, root)
        for name in ('metadata.json', 'arrays.npz', 'checksums.json'):
            _contained(directory / name, directory)
        verify_record(directory)
        metadata = _json(directory / 'metadata.json')
        if _integer(metadata.get('schema_version'), 'record schema_version') != SCHEMA_VERSION:
            raise Figure7LoadError('unsupported Figure 7 record schema')
        observation = _mapping(metadata.get('observation'), 'record observation')
        if observation.get('status') != 'complete' or observation.get('record_id') != candidate.record_id:
            raise Figure7LoadError('selected record is not a complete matching observation')
        with np.load(directory / 'arrays.npz', allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        _validate_arrays(arrays, candidate)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise Figure7LoadError('%s: %s' % (directory, error)) from error
    entries = tuple(item for item in ledger if item.record_id == candidate.record_id)
    if len(entries) != 1:
        raise Figure7LoadError('selected record must have exactly one ledger row')
    captured = _ledger_entry(observation)
    if (captured != entries[0]
            or json.dumps(observation, sort_keys=True) != json.dumps(entries[0].observation, sort_keys=True)):
        raise Figure7LoadError('ledger disagrees with checksummed observation')
    validate_observation(candidate, arrays, captured)
    references = tuple(item for item in entries[0].references if item.index == candidate.reference_index)
    if len(references) != 1:
        raise Figure7LoadError('selected proposal is absent from its ledger row')
    if abs(references[0].reliability - candidate.reliability) > 1e-12:
        raise Figure7LoadError('selection reliability disagrees with ledger')
    return SelectedRecord(candidate, directory, arrays, references[0])


def load_capture(capture: Path) -> CaptureBundle:
    run, selection = _json(capture / 'run.json'), _json(capture / 'selection.json')
    if any(_integer(value.get('schema_version'), 'schema_version') != SCHEMA_VERSION
           for value in (run, selection)):
        raise Figure7LoadError('unsupported Figure 7 schema')
    if selection.get('status') != 'complete':
        raise Figure7LoadError('selection.json does not declare a complete capture')
    ledger = _load_ledger(capture / 'ledger.jsonl')
    case_a_candidate = _candidate(selection.get('case_a'))
    case_b_candidate = _candidate(selection.get('case_b'))
    if case_a_candidate.pool != 'variable' or case_b_candidate.pool != 'stable':
        raise Figure7LoadError('selection case semantics are incompatible')
    retained_candidates = tuple(_candidate(item) for item in _sequence(
        selection.get('retained'), 'retained'))
    selected = tuple(_selected(capture, item, ledger) for item in retained_candidates)
    by_key = {(item.candidate.record_id, item.candidate.reference_index,
               item.candidate.pool): item for item in selected}
    case_a = by_key.get((case_a_candidate.record_id, case_a_candidate.reference_index, 'variable'))
    case_b = by_key.get((case_b_candidate.record_id, case_b_candidate.reference_index, 'stable'))
    if case_a is None or case_b is None:
        raise Figure7LoadError('selected pair is absent from retained records')
    if case_a.candidate != case_a_candidate or case_b.candidate != case_b_candidate:
        raise Figure7LoadError('selected cases disagree with retained candidates')
    if len(by_key) != len(selected):
        raise Figure7LoadError('retained candidates contain duplicate keys')
    provenance = _mapping(run.get('provenance'), 'provenance')
    return CaptureBundle(
        capture, ProvenanceInfo(*(_text(provenance.get(name), name) for name in (
            'config', 'command', 'source_checkpoint', 'fixed_seed'))),
        _text(run.get('selection_policy'), 'selection_policy'), case_a, case_b, selected, ledger,
    )
