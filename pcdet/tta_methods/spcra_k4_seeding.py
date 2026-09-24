from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SeedingInputError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


class ViewCollisionError(RuntimeError):
    def __init__(self, token: str, view_index: int, max_attempts: int) -> None:
        self.token = token
        self.view_index = view_index
        self.max_attempts = max_attempts
        super().__init__(str(self))

    def __str__(self) -> str:
        return f'{self.token}: view {self.view_index} exhausted {self.max_attempts} attempts'


@dataclass(frozen=True)  # noqa: SLOTS_OK - Python 3.8 runtime.
class ViewSamples:
    indices: tuple[NDArray[np.int64], ...]
    transforms: tuple[NDArray[np.float32], ...]
    fingerprints: tuple[str, ...]
    attempts: tuple[int, ...]
    law_identifier: str


def seed_for_view(
    token: str, view_index: int, *, base_seed: int, schema_version: str,
    attempt: int = 0, stream: Literal['drop', 'geometry'],
) -> int:
    """Hash compact UTF-8 JSON identity; interpret the first 128 digest bits little-endian."""
    if type(base_seed) is not int or type(view_index) is not int or type(attempt) is not int:
        raise SeedingInputError('base_seed, view_index and attempt must be integers')
    if not 0 <= view_index < 4 or attempt < 0:
        raise SeedingInputError('view_index must be in [0, 3] and attempt nonnegative')
    if not isinstance(token, str) or not token:
        raise SeedingInputError('token must be a nonempty string')
    if not isinstance(schema_version, str) or not schema_version:
        raise SeedingInputError('schema_version must be a nonempty string')
    if stream not in ('drop', 'geometry'):
        raise SeedingInputError('stream must be drop or geometry')
    identity = (f'spcra-k4/{stream}', base_seed, token, view_index, schema_version, attempt)
    payload = json.dumps(identity, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], 'little')


def rng_for_view(
    token: str, view_index: int, *, base_seed: int, schema_version: str,
    attempt: int = 0, stream: Literal['drop', 'geometry'],
) -> np.random.Generator:
    seed = seed_for_view(token, view_index, base_seed=base_seed, schema_version=schema_version,
                         attempt=attempt, stream=stream)
    return np.random.Generator(np.random.PCG64(seed))


def _canonical_view(
    retained_indices: ArrayLike, transform: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    indices, matrix = np.asarray(retained_indices), np.asarray(transform)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise SeedingInputError('retained_indices must be a one-dimensional integer array')
    if np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max):
        raise SeedingInputError('retained_indices must fit nonnegative int64')
    if np.unique(indices).size != indices.size:
        raise SeedingInputError('retained_indices must not repeat a point')
    if matrix.shape != (4, 4) or matrix.dtype.kind not in 'iuf':
        raise SeedingInputError('transform must be a real numeric 4x4 matrix')
    if not np.isfinite(matrix).all() or np.any(np.abs(matrix) > np.finfo(np.float32).max):
        raise SeedingInputError('transform must be finite and representable as float32')
    indices = np.array(indices, dtype='<i8', order='C', copy=True)
    matrix = np.array(matrix, dtype='<f4', order='C', copy=True)
    if not np.array_equal(matrix[3], [0., 0., 0., 1.]) or np.linalg.det(matrix[:3, :3]) == 0:
        raise SeedingInputError('transform must be an invertible affine matrix')
    indices.setflags(write=False)
    matrix.setflags(write=False)
    return indices, matrix


def _fingerprint(
    reference_digest: str, view: tuple[NDArray[np.int64], NDArray[np.float32]],
    schema_version: str, law_identifier: str,
) -> str:
    if not isinstance(reference_digest, str) or re.fullmatch('[0-9a-f]{64}', reference_digest) is None:
        raise SeedingInputError('reference_digest must be a lowercase SHA256 hex digest')
    if not isinstance(schema_version, str) or not schema_version:
        raise SeedingInputError('schema_version must be a nonempty string')
    if not isinstance(law_identifier, str) or not law_identifier:
        raise SeedingInputError('law_identifier must be a nonempty string')
    indices, transform = view
    metadata = ('spcra-k4/view', law_identifier, reference_digest, schema_version,
                ('<i8', indices.shape), ('<f4', transform.shape))
    header = json.dumps(metadata, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(header + b'\x00' + indices.tobytes(order='C') + transform.tobytes(order='C')).hexdigest()


def fingerprint_view(
    reference_digest: str, retained_indices: ArrayLike, transform: ArrayLike,
    schema_version: str, law_identifier: str,
) -> str:
    """Fingerprint realized content, not RNG identity, using canonical <i8/<f4 C-order arrays."""
    return _fingerprint(
        reference_digest, _canonical_view(retained_indices, transform), schema_version, law_identifier,
    )


def sample_unique_views(
    token: str, *, base_seed: int, schema_version: str, reference_digest: str, law_identifier: str,
    build_view: Callable[[np.random.Generator, np.random.Generator], tuple[ArrayLike, ArrayLike]],
    max_attempts: int = 100,
) -> ViewSamples:
    """Call build_view(drop_rng, geometry_rng); max_attempts includes the initial attempt per view."""
    if type(max_attempts) is not int or max_attempts <= 0:
        raise SeedingInputError('max_attempts must be a positive integer')
    if not isinstance(law_identifier, str) or not law_identifier:
        raise SeedingInputError('law_identifier must be a nonempty string')
    retained: list[NDArray[np.int64]] = []
    transforms: list[NDArray[np.float32]] = []
    fingerprints: list[str] = []
    attempts: list[int] = []
    for view_index in range(4):
        for attempt in range(max_attempts):
            drop = rng_for_view(token, view_index, base_seed=base_seed, schema_version=schema_version,
                                attempt=attempt, stream='drop')
            geometry = rng_for_view(token, view_index, base_seed=base_seed, schema_version=schema_version,
                                    attempt=attempt, stream='geometry')
            view = _canonical_view(*build_view(drop, geometry))
            fingerprint = _fingerprint(reference_digest, view, schema_version, law_identifier)
            if fingerprint in fingerprints:
                continue
            retained.append(view[0])
            transforms.append(view[1])
            fingerprints.append(fingerprint)
            attempts.append(attempt)
            break
        else:
            raise ViewCollisionError(token, view_index, max_attempts)
    return ViewSamples(
        tuple(retained), tuple(transforms), tuple(fingerprints), tuple(attempts), law_identifier,
    )
