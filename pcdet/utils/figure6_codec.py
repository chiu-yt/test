"""Versioned JSON boundary and per-array integrity for Figure 6 artifacts."""

from dataclasses import asdict, fields
from hashlib import sha256
from typing import Dict, List, Mapping, Set, Tuple, Union

from .figure6_schema import (
    ArtifactError, CaptureArray, CaptureRecord, Occurrence, SCHEMA_VERSION,
    StageCapture, StageState, StageStatus,
)


JsonValue = Union[None, bool, int, float, str, List['JsonValue'], Dict[str, 'JsonValue']]


def json_map(value: JsonValue) -> Dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ArtifactError('expected JSON mapping')
    return value


def json_text(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ArtifactError('expected JSON string')
    return value


def json_int(value: JsonValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactError('expected JSON integer')
    return value


def array_metadata(values: CaptureArray) -> Dict[str, JsonValue]:
    return {
        'shape': list(values.shape),
        'dtype': values.dtype.str,
        'sha256': sha256(values.tobytes(order='C')).hexdigest(),
    }


def encode_record(record: CaptureRecord) -> Tuple[Dict[str, JsonValue], Dict[str, CaptureArray]]:
    arrays = {
        '%s/%s' % (stage_name, name): values
        for stage_name, stage in record.stages.items()
        for name, values in stage.arrays.items()
    }
    stages: Dict[str, JsonValue] = {}
    for name, stage in record.stages.items():
        stages[name] = {
            'state': stage.status.state.value,
            'owner': stage.status.owner,
            'detail': stage.status.detail,
            'arrays': list(stage.arrays),
        }
    descriptors: Dict[str, JsonValue] = {
        name: array_metadata(values) for name, values in arrays.items()
    }
    metadata: Dict[str, JsonValue] = {
        'schema_version': SCHEMA_VERSION,
        'identity': asdict(record.identity),
        'protocol': {key: value for key, value in record.protocol.items()},
        'stages': stages,
        'arrays': descriptors,
    }
    return metadata, arrays


def decode_record(metadata: JsonValue, arrays: Mapping[str, CaptureArray]) -> CaptureRecord:
    """Parse once at the filesystem boundary, rejecting inconsistent declarations."""
    root = json_map(metadata)
    if set(root) != {'schema_version', 'identity', 'protocol', 'stages', 'arrays'}:
        raise ArtifactError('invalid metadata fields')
    if json_int(root['schema_version']) != SCHEMA_VERSION:
        raise ArtifactError('unsupported schema version')
    raw_identity = json_map(root['identity'])
    if set(raw_identity) != {field.name for field in fields(Occurrence)}:
        raise ArtifactError('invalid occurrence fields')
    identity = Occurrence(
        token=json_text(raw_identity['token']), frame_id=json_text(raw_identity['frame_id']),
        epoch=json_int(raw_identity['epoch']),
        accumulated_iter_before=json_int(raw_identity['accumulated_iter_before']),
        samples_seen=json_int(raw_identity['samples_seen']),
        global_rank=json_int(raw_identity['global_rank']),
        world_size=json_int(raw_identity['world_size']),
        batch_index=json_int(raw_identity['batch_index']),
        batch_size=json_int(raw_identity['batch_size']),
    )
    descriptors = json_map(root['arrays'])
    if set(descriptors) != set(arrays):
        raise ArtifactError('array inventory mismatch')
    for name, values in arrays.items():
        descriptor = json_map(descriptors[name])
        if set(descriptor) != {'shape', 'dtype', 'sha256'}:
            raise ArtifactError('invalid array descriptor')
        shape = descriptor['shape']
        if not isinstance(shape, list) or any(type(size) is not int for size in shape):
            raise ArtifactError('invalid array shape')
        if shape != list(values.shape) or descriptor['dtype'] != values.dtype.str:
            raise ArtifactError('array shape/dtype mismatch: %s' % name)
        if descriptor['sha256'] != array_metadata(values)['sha256']:
            raise ArtifactError('array checksum mismatch: %s' % name)
    stages: Dict[str, StageCapture] = {}
    referenced: Set[str] = set()
    for name, raw_stage in json_map(root['stages']).items():
        stage = json_map(raw_stage)
        if set(stage) != {'state', 'owner', 'detail', 'arrays'}:
            raise ArtifactError('invalid stage fields')
        names = stage['arrays']
        if not isinstance(names, list):
            raise ArtifactError('invalid stage array list')
        array_names = tuple(json_text(value) for value in names)
        if len(set(array_names)) != len(array_names):
            raise ArtifactError('duplicate stage array name')
        keys = tuple('%s/%s' % (name, value) for value in array_names)
        if not set(keys).issubset(arrays):
            raise ArtifactError('stage references missing arrays')
        referenced.update(keys)
        stages[name] = StageCapture(
            StageStatus(StageState(json_text(stage['state'])),
                        json_text(stage['owner']), json_text(stage['detail'])),
            {array_name: arrays[key] for array_name, key in zip(array_names, keys)},
        )
    if referenced != set(arrays):
        raise ArtifactError('unowned arrays')
    protocol = {key: json_text(value) for key, value in json_map(root['protocol']).items()}
    return CaptureRecord(identity, protocol, stages)
