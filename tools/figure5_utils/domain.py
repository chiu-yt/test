from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Final, NewType, Optional, Tuple

from .palette import DetectionClass


SampleToken = NewType('SampleToken', str)
SceneToken = NewType('SceneToken', str)
DetectionScore = NewType('DetectionScore', float)
DistanceMeters = NewType('DistanceMeters', float)
YawRadians = NewType('YawRadians', float)
Vector3 = Tuple[float, float, float]
ImageRoi = Tuple[float, float, float, float]


def _immutable_vector3(values: Vector3) -> Vector3:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != 3:
        raise Vector3LengthError(actual_length=len(normalized))
    return normalized[0], normalized[1], normalized[2]


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class Vector3LengthError(ValueError):
    actual_length: int

    def __str__(self) -> str:
        return '3D vectors require exactly three values; received %d' % self.actual_length


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class Box3D:
    center: Vector3
    size: Vector3
    yaw: YawRadians

    def __post_init__(self) -> None:
        object.__setattr__(self, 'center', _immutable_vector3(self.center))
        object.__setattr__(self, 'size', _immutable_vector3(self.size))
        object.__setattr__(self, 'yaw', YawRadians(float(self.yaw)))


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class Detection:
    sample_token: SampleToken
    class_name: DetectionClass
    score: DetectionScore
    box: Box3D


@unique
class FigureColumn(str, Enum):
    GT = 'GT'
    SOURCE_ONLY = 'Source-only'
    CODEMERGE = 'CodeMerge'
    REFUSE_TTA = 'ReFuse-TTA'


FIGURE_COLUMNS: Final[Tuple[FigureColumn, ...]] = (
    FigureColumn.GT,
    FigureColumn.SOURCE_ONLY,
    FigureColumn.CODEMERGE,
    FigureColumn.REFUSE_TTA,
)


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class FrameRecord:
    sample_token: SampleToken
    ground_truth: Tuple[Detection, ...]
    source_only: Tuple[Detection, ...]
    codemerge: Tuple[Detection, ...]
    refuse_tta: Tuple[Detection, ...]
    scene_name: Optional[str] = None
    scene_token: Optional[SceneToken] = None
    frame_index: Optional[int] = None

    def detections(self, column: FigureColumn) -> Tuple[Detection, ...]:
        return {
            FigureColumn.GT: self.ground_truth,
            FigureColumn.SOURCE_ONLY: self.source_only,
            FigureColumn.CODEMERGE: self.codemerge,
            FigureColumn.REFUSE_TTA: self.refuse_tta,
        }[column]

    def columns(self) -> Tuple[FigureColumn, ...]:
        return FIGURE_COLUMNS


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class DetectionMatch:
    ground_truth: Detection
    prediction: Detection
    center_distance_m: DistanceMeters


@unique
class CalloutKind(str, Enum):
    FAR_RANGE_RECOVERY = 'far_range_recovery'
    SMALL_OBJECT_RECOVERY = 'small_object_recovery'
    FALSE_POSITIVE_REDUCTION = 'false_positive_reduction'
    BETTER_LOCALIZATION = 'better_localization'


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class Callout:
    roi: ImageRoi
    class_name: DetectionClass
    distance_m: DistanceMeters
    kind: CalloutKind


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class CandidateEvidence:
    recovered_from_source: int = 0
    recovered_from_codemerge: int = 0
    false_positive_removed: int = 0
    far_small_detected: int = 0
    better_localization: int = 0
    num_gt_objects: int = 0
    num_source_preds: int = 0
    num_codemerge_preds: int = 0
    num_refuse_preds: int = 0
    involved_classes: Tuple[DetectionClass, ...] = ()
    primary_distances: Tuple[DistanceMeters, ...] = ()
