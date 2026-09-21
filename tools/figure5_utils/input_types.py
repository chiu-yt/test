"""CPU-only input contracts shared by artifact and point readers (Python 3.8)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from .domain import Detection, FrameRecord, SampleToken, SceneToken


class ArtifactFormatError(ValueError):
    def __init__(self, context: str, detail: str) -> None:
        self.context = context
        self.detail = detail
        super().__init__('%s: %s' % (context, detail))


class MissingTokenError(ArtifactFormatError):
    pass


class DuplicateTokenError(ArtifactFormatError):
    pass


class TokenMismatchError(ArtifactFormatError):
    pass


@dataclass(frozen=True)
class ResultPaths:
    source_only: Path
    codemerge: Path
    refuse_tta: Path


@dataclass(frozen=True)
class SceneMetadata:
    scene_token: Optional[SceneToken] = None
    scene_name: Optional[str] = None
    frame_index: Optional[int] = None


@dataclass(frozen=True)
class SweepInfo:
    lidar_path: Path
    transform_matrix: Optional[np.ndarray]
    time_lag: float


@dataclass(frozen=True)
class ValidationInfo:
    token: SampleToken
    ground_truth: Tuple[Detection, ...]
    lidar_path: Path
    sweeps: Tuple[SweepInfo, ...]
    scene: SceneMetadata


@dataclass(frozen=True)
class ArtifactDataset:
    frames: Mapping[SampleToken, FrameRecord]
    infos: Mapping[SampleToken, ValidationInfo]


@dataclass(frozen=True)
class PointSource:
    """Root resolves info lidar paths; optional NPY directory contains sparse rows."""
    data_root: Path
    seed: int = 1024
    sparse_directory: Optional[Path] = None
    sparsity_mode: str = 'density_dec_global'
    sparsity_severity: int = 5


@dataclass(frozen=True)
class LoadedFrame:
    frame: FrameRecord
    sparse_points: np.ndarray
    dense_points: Optional[np.ndarray]
    sweep_indices: Tuple[int, ...]
