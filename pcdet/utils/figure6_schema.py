"""NumPy-only Figure 6 capture values; no runtime hooks or rendering dependencies."""

from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Final, Mapping, Sequence, Tuple
import re

import numpy as np


if TYPE_CHECKING:
    CaptureArray = np.ndarray[Tuple[int, ...], np.dtype[np.generic]]
else:
    CaptureArray = np.ndarray
SCHEMA_VERSION: Final[int] = 1
FIXED_TOKENS: Final[Tuple[str, ...]] = (
    '9a476217cc324813a5760c9852643324',
    '3425c66163af46cca4c96006d425e0eb',
    '2128fc958907421ca888ab014a73348a',
)


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class ArtifactError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def require_text(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ArtifactError('expected nonempty text')


def require_name(value: str) -> None:
    require_text(value)
    if re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', value) is None:
        raise ArtifactError('invalid stage or array name: %s' % value)


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class Occurrence:
    """batch_index is the token's zero-based position in this rank's batch."""

    token: str
    frame_id: str
    epoch: int
    accumulated_iter_before: int
    samples_seen: int
    global_rank: int
    world_size: int
    batch_index: int
    batch_size: int

    def __post_init__(self) -> None:
        require_text(self.token)
        require_text(self.frame_id)
        for field in fields(self)[2:]:
            value = getattr(self, field.name)
            if type(value) is not int or value < 0:
                raise ArtifactError('%s must be a nonnegative integer' % field.name)
        if not 0 <= self.global_rank < self.world_size:
            raise ArtifactError('global_rank must be smaller than world_size')
        if not 0 <= self.batch_index < self.batch_size:
            raise ArtifactError('batch_index must be smaller than batch_size')

    @property
    def selection_key(self) -> Tuple[int, int, int, int, str]:
        return (self.epoch, self.accumulated_iter_before, self.global_rank,
                self.batch_index, self.token)


class StageState(str, Enum):
    OBSERVED_EMPTY = 'observed_empty'
    MISSING = 'missing'
    FAILED = 'failed'
    COMPLETE = 'complete'


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class StageStatus:
    state: StageState
    owner: str
    detail: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.state, StageState):
            raise ArtifactError('stage state must be a StageState')
        require_text(self.owner)
        if not isinstance(self.detail, str):
            raise ArtifactError('stage detail must be text')
        if self.state is StageState.FAILED and not self.detail:
            raise ArtifactError('failed stage requires a detail')


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class StageCapture:
    """Snapshot arrays on input; absent/failed stages cannot masquerade as empty."""

    status: StageStatus
    arrays: Mapping[str, CaptureArray]

    def __post_init__(self) -> None:
        copied: Dict[str, CaptureArray] = {}
        for name, values in self.arrays.items():
            require_name(name)
            if not isinstance(values, np.ndarray) or values.dtype.kind not in 'biufcSU':
                raise ArtifactError('arrays must have numeric, boolean or string dtype')
            snapshot = np.array(values, copy=True, order='C')
            snapshot.setflags(write=False)
            copied[name] = snapshot
        has_values = any(values.size for values in copied.values())
        valid = {
            StageState.MISSING: not copied,
            StageState.FAILED: not copied,
            StageState.OBSERVED_EMPTY: bool(copied) and not has_values,
            StageState.COMPLETE: bool(copied) and has_values,
        }
        if not valid[self.status.state]:
            raise ArtifactError('stage arrays disagree with state %s' % self.status.state.value)
        object.__setattr__(self, 'arrays', MappingProxyType(copied))


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class CaptureRecord:
    """Protocol is an immutable string map; each stage explicitly names its owner."""

    identity: Occurrence
    protocol: Mapping[str, str]
    stages: Mapping[str, StageCapture]

    def __post_init__(self) -> None:
        if not self.protocol or not self.stages:
            raise ArtifactError('protocol and stages must be explicit and nonempty')
        for name, value in self.protocol.items():
            require_text(name)
            require_text(value)
        for name in self.stages:
            require_name(name)
        object.__setattr__(self, 'protocol', MappingProxyType(dict(self.protocol)))
        object.__setattr__(self, 'stages', MappingProxyType(dict(self.stages)))


def selected_batch_indices(
        tokens: Sequence[str], requested: Sequence[str] = FIXED_TOKENS,
) -> Tuple[int, ...]:
    selected = frozenset(requested)
    return tuple(index for index, token in enumerate(tokens) if token in selected)
