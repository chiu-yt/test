"""Rank-private persistence; metadata.json is the last-published completion marker."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Dict, List, Mapping, Sequence, Tuple
from zipfile import BadZipFile
import zlib

import numpy as np
from numpy.lib.npyio import NpzFile

from .figure6_codec import decode_record, encode_record
from .figure6_schema import ArtifactError, CaptureRecord


def write_record(capture: Path, record: CaptureRecord) -> Path:
    """Reserve an occurrence exclusively; never replace an existing occurrence."""
    metadata, arrays = encode_record(record)
    identity_bytes = json.dumps(asdict(record.identity), sort_keys=True).encode('utf-8')
    occurrence_name = sha256(identity_bytes).hexdigest()
    directory = capture / ('rank_%05d' % record.identity.global_rank) / 'records' / occurrence_name
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    temporary_paths: List[Path] = []
    try:
        with NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as stream:
            tensor_path = Path(stream.name)
            temporary_paths.append(tensor_path)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        tensor_path.replace(directory / 'tensors.npz')
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                suffix='.tmp', delete=False) as stream:
            metadata_path = Path(stream.name)
            temporary_paths.append(metadata_path)
            json.dump(metadata, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        metadata_path.replace(directory / 'metadata.json')
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return directory


def load_record(directory: Path) -> CaptureRecord:
    """Read only committed records and verify every array without pickle support."""
    try:
        with (directory / 'metadata.json').open(encoding='utf-8') as stream:
            metadata = json.load(stream)
        loaded = np.load(directory / 'tensors.npz', allow_pickle=False)
        if not isinstance(loaded, NpzFile):
            raise ArtifactError('expected NPZ archive')
        with loaded as archive:
            if len(set(archive.files)) != len(archive.files):
                raise ArtifactError('duplicate archive entries')
            arrays = {name: archive[name] for name in archive.files}
            if any(not isinstance(values, np.ndarray) for values in arrays.values()):
                raise ArtifactError('archive member is not an array')
        record = decode_record(metadata, arrays)
        if directory.parent.name != 'records' or directory.parent.parent.name != (
                'rank_%05d' % record.identity.global_rank):
            raise ArtifactError('record rank does not match directory')
        return record
    except (OSError, ValueError, EOFError, BadZipFile, zlib.error) as error:
        raise ArtifactError('%s: %s' % (directory, error)) from error


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class LocatedRecord:
    path: Path
    record: CaptureRecord


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class RecordIssue:
    path: Path
    detail: str


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class RecordCatalog:
    records: Tuple[LocatedRecord, ...]
    issues: Tuple[RecordIssue, ...]


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class OccurrenceSelection:
    selected: LocatedRecord
    alternatives: Tuple[LocatedRecord, ...]


def scan_records(capture: Path) -> RecordCatalog:
    """Report incomplete/corrupt occurrences rather than silently treating them as absent."""
    records: List[LocatedRecord] = []
    issues: List[RecordIssue] = []
    for directory in sorted(capture.glob('rank_*/records/*')):
        if not directory.is_dir():
            continue
        try:
            records.append(LocatedRecord(directory, load_record(directory)))
        except ArtifactError as error:
            issues.append(RecordIssue(directory, str(error)))
    return RecordCatalog(tuple(records), tuple(issues))


def select_occurrences(records: Sequence[LocatedRecord]) -> Mapping[str, OccurrenceSelection]:
    """Earliest per token, independent of scan order; return every alternative."""
    grouped: Dict[str, List[LocatedRecord]] = {}
    for located in sorted(records, key=lambda item: (
            item.record.identity.selection_key, str(item.path))):
        grouped.setdefault(located.record.identity.token, []).append(located)
    return MappingProxyType({
        token: OccurrenceSelection(items[0], tuple(items[1:]))
        for token, items in grouped.items()
    })
