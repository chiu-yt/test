"""One staged record at a time; checksummed directory publication, numeric NPZ only."""

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .figure6_schema import CaptureArray
from .figure7_schema import Figure7Error, LedgerRecord, Observation, SCHEMA_VERSION


def array_digest(value: CaptureArray) -> str:
    digest = sha256(str((value.dtype.str, value.shape)).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(payload + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_ledger(output: Path, ledger: LedgerRecord) -> None:
    payload = json.dumps(asdict(ledger), sort_keys=True, allow_nan=False)
    with (output / 'ledger.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(payload + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def evidence_arrays(frame: Observation) -> Mapping[str, CaptureArray]:
    evidence, points = frame.evidence, frame.points
    if evidence is None or points is None:
        raise Figure7Error('complete evidence and actual current points are required')
    arrays = {
        'reference_points': points.reference,
        'reference_transform': points.reference_transform,
        'reference_mask': evidence.reference_mask,
        'reference_rescue_mask': evidence.reference_rescue_mask,
        'match_indices': evidence.match_indices, 'view_quality': evidence.view_quality,
        'reliability': evidence.reliability, 'coverage': evidence.coverage,
        'support': evidence.support,
    }
    for prefix, prediction in zip(
            ('reference', 'view_0', 'view_1', 'view_2', 'view_3'),
            (evidence.reference_prediction, *evidence.view_predictions)):
        arrays[prefix + '_boxes'] = prediction.boxes
        arrays[prefix + '_labels'] = prediction.labels
        arrays[prefix + '_scores'] = prediction.scores
        if prediction.camera_support is not None:
            arrays[prefix + '_camera_support'] = prediction.camera_support
    for index in range(4):
        prefix = f'view_{index}'
        arrays[prefix + '_points'] = points.views[index]
        arrays[prefix + '_mask'] = evidence.view_masks[index]
        arrays[prefix + '_rescue_mask'] = evidence.view_rescue_masks[index]
        arrays[prefix + '_retained_indices'] = evidence.retained_indices[index]
        arrays[prefix + '_transform'] = evidence.transforms[index]
    if any(value.dtype.kind not in 'biuf' for value in arrays.values()):
        raise Figure7Error('only real numeric and boolean NPZ arrays are permitted')
    return arrays


def stage_record(output: Path, frame: Observation, ledger: LedgerRecord) -> Path:
    arrays = evidence_arrays(frame)
    staging = output / '_staging'
    staging.mkdir()
    metadata = {'schema_version': SCHEMA_VERSION, 'observation': asdict(ledger), 'arrays': {
        name: {'shape': list(value.shape), 'dtype': value.dtype.str, 'sha256': array_digest(value)}
        for name, value in arrays.items()
    }}
    with (staging / 'arrays.npz').open('wb') as stream:
        with ZipFile(stream, mode='w', compression=ZIP_DEFLATED) as archive:
            for name, value in arrays.items():
                with archive.open(name + '.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_text(staging / 'metadata.json', json.dumps(metadata, sort_keys=True, allow_nan=False))
    hashes = {name: file_digest(staging / name) for name in ('arrays.npz', 'metadata.json')}
    atomic_text(staging / 'checksums.json', json.dumps(hashes, sort_keys=True))
    verify_record(staging)
    return staging


def verify_record(directory: Path) -> None:
    """Reject corrupt, incomplete or nonnumeric archives without enabling pickle."""
    hashes = json.loads((directory / 'checksums.json').read_text(encoding='utf-8'))
    if set(hashes) != {'arrays.npz', 'metadata.json'}:
        raise Figure7Error('invalid record checksum manifest')
    for name in ('arrays.npz', 'metadata.json'):
        if file_digest(directory / name) != hashes[name]:
            raise Figure7Error('record checksum mismatch: ' + name)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['schema_version'] != SCHEMA_VERSION:
        raise Figure7Error('unsupported Figure 7 schema')
    with np.load(directory / 'arrays.npz', allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(metadata['arrays']):
            raise Figure7Error('archive membership mismatch')
        for name in archive.files:
            value = archive[name]
            description = metadata['arrays'][name]
            if (value.dtype.kind not in 'biuf' or list(value.shape) != description['shape']
                    or value.dtype.str != description['dtype']
                    or array_digest(value) != description['sha256']):
                raise Figure7Error('array integrity mismatch: ' + name)
