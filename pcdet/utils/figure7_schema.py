"""Versioned, GT-free values for current-pre-update Figure 7 observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from .figure6_schema import CaptureArray, Occurrence

if TYPE_CHECKING:
    from pcdet.tta_methods.spcra_k4_evidence import K4Evidence

SCHEMA_VERSION: Final[int] = 1
CLASS_NAMES: Final[tuple[str, ...]] = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
)
Pool = Literal['stable', 'variable']
Status = Literal['complete', 'empty', 'ineligible', 'incomplete', 'failed']
RankKey = tuple[int, float, float, float, str, int, int, int, int, int]


class Figure7Error(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class Provenance:
    config: str
    command: str
    source_checkpoint: str
    fixed_seed: str

    def __post_init__(self) -> None:
        if not all((self.config, self.command, self.source_checkpoint, self.fixed_seed)):
            raise Figure7Error('explicit run provenance is required')


@dataclass(frozen=True, slots=True)
class PoolLimits:
    stable: int = 10
    variable: int = 10

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (self.stable, self.variable)):
            raise Figure7Error('both pool capacities must be positive integers')

    @property
    def capacity(self) -> int:
        return self.stable + self.variable


@dataclass(frozen=True, slots=True)
class CurrentPoints:
    """Owned CPU arrays, in the actual reference/view input coordinate frames.

    Reference xyz occupy columns 0:3 (no collated batch column). Views must be
    supplied by the caller, never reconstructed from retained indices here.
    """

    reference: CaptureArray
    views: tuple[CaptureArray, ...]
    reference_transform: CaptureArray

    def __post_init__(self) -> None:
        if len(self.views) != 4:
            raise Figure7Error('exactly four actual view point arrays are required')
        arrays = (self.reference, *self.views, self.reference_transform)
        if any(value.dtype.kind not in 'iuf' or not np.isfinite(value).all() for value in arrays):
            raise Figure7Error('point evidence must contain finite real numeric arrays')
        if any(value.ndim != 2 or value.shape[1] < 3 for value in arrays[:5]):
            raise Figure7Error('point arrays must have shape (N, D>=3)')
        if self.reference_transform.shape != (4, 4):
            raise Figure7Error('reference transform must have shape (4, 4)')
        copies = tuple(np.array(value, copy=True, order='C') for value in arrays)
        for value in copies:
            value.setflags(write=False)
        object.__setattr__(self, 'reference', copies[0])
        object.__setattr__(self, 'views', copies[1:5])
        object.__setattr__(self, 'reference_transform', copies[5])


@dataclass(frozen=True, slots=True)
class Observation:
    identity: Occurrence
    model_step: int
    evidence: K4Evidence | None
    points: CurrentPoints | None
    failure: str = ''

    def __post_init__(self) -> None:
        if type(self.model_step) is not int or self.model_step < 0:
            raise Figure7Error('model_step must identify the nonnegative pre-update step')


@dataclass(frozen=True, slots=True)
class ReferenceRow:
    index: int
    class_name: str
    score: float | None
    accepted: bool
    rescued: bool
    matches: tuple[int, ...]
    qualities: tuple[float, ...]
    reliability: float
    range_m: float | None
    stable_key: RankKey | None
    variable_key: RankKey | None
    stable_eligible: bool = field(init=False)
    variable_eligible: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'stable_eligible', self.stable_key is not None)
        object.__setattr__(self, 'variable_eligible', self.variable_key is not None)


@dataclass(frozen=True, slots=True)
class Candidate:
    record_id: str
    identity: Occurrence
    model_step: int
    reference_index: int
    class_name: str
    score: float
    reliability: float
    pool: Pool
    rank_key: RankKey


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    schema_version: int
    record_id: str
    identity: Occurrence
    model_step: int
    status: Status
    detail: str
    point_counts: tuple[int, ...]
    prediction_counts: tuple[int, ...]
    accepted_counts: tuple[int, ...]
    rescued_counts: tuple[int, ...]
    transforms: tuple[tuple[float, ...], ...]
    fingerprints: tuple[str, ...]
    point_checksums: tuple[str, ...]
    attempts: tuple[int, ...]
    law_identifier: str
    completion: tuple[bool, ...]
    coverage: tuple[float, ...]
    support: tuple[float, ...]
    references: tuple[ReferenceRow, ...]
    admitted_pools: tuple[Pool, ...] = ()
