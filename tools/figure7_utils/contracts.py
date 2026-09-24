from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union

import numpy as np

from pcdet.utils.figure6_schema import Occurrence
from pcdet.utils.figure7_schema import Candidate


JsonValue = Union[None, bool, int, float, str, Sequence['JsonValue'], Mapping[str, 'JsonValue']]
Crop = Tuple[float, float, float, float]


class Figure7LoadError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class ProvenanceInfo:
    config: str
    command: str
    source_checkpoint: str
    fixed_seed: str


@dataclass(frozen=True, slots=True)
class LedgerReference:
    index: int
    class_name: str
    score: float | None
    accepted: bool
    rescued: bool
    matches: Tuple[int, ...]
    qualities: Tuple[float, ...]
    reliability: float
    stable_key: Tuple[JsonValue, ...] | None
    variable_key: Tuple[JsonValue, ...] | None


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    record_id: str
    token: str
    frame_id: str
    epoch: int
    model_step: int
    global_rank: int
    batch_index: int
    status: str
    admitted_pools: Tuple[str, ...]
    references: Tuple[LedgerReference, ...]
    identity: Occurrence
    observation: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SelectedRecord:
    candidate: Candidate
    directory: Path
    arrays: Mapping[str, np.ndarray]
    reference: LedgerReference

    @staticmethod
    def inverse_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        aligned = np.array(points, copy=True)
        aligned[:, :3] = (aligned[:, :3] - matrix[:3, 3]) @ np.linalg.inv(matrix[:3, :3]).T
        return aligned

    @staticmethod
    def inverse_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        aligned = np.array(boxes, copy=True)
        linear = matrix[:3, :3]
        inverse = np.linalg.inv(linear)
        scale = float(np.linalg.norm(linear[:, 0]))
        aligned[:, :3] = (aligned[:, :3] - matrix[:3, 3]) @ inverse.T
        aligned[:, 3:6] /= scale
        directions = np.column_stack((np.cos(aligned[:, 6]), np.sin(aligned[:, 6])))
        directions = directions @ inverse[:2, :2].T
        aligned[:, 6] = np.arctan2(directions[:, 1], directions[:, 0])
        return aligned


@dataclass(frozen=True, slots=True)
class CaptureBundle:
    capture_dir: Path
    provenance: ProvenanceInfo
    selection_policy: str
    case_a: SelectedRecord
    case_b: SelectedRecord
    retained: Tuple[SelectedRecord, ...]
    ledger: Tuple[LedgerEntry, ...]
