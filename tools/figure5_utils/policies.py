from dataclasses import dataclass
from typing import Tuple

from .domain import (
    FIGURE_COLUMNS,
    CalloutKind,
    CandidateEvidence,
    DistanceMeters,
    FigureColumn,
    SampleToken,
)


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class MatchingPolicy:
    max_center_distance_m: DistanceMeters = DistanceMeters(2.0)
    min_bev_iou: float = 0.0
    class_aware: bool = True


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class CandidatePolicy:
    recovered_from_source_weight: float = 3.0
    recovered_from_codemerge_weight: float = 2.0
    false_positive_removed_weight: float = 1.5
    far_small_detected_weight: float = 1.0
    gt_object_penalty: float = 0.1

    def score(self, evidence: CandidateEvidence) -> float:
        return (
            self.recovered_from_source_weight * evidence.recovered_from_source
            + self.recovered_from_codemerge_weight * evidence.recovered_from_codemerge
            + self.false_positive_removed_weight * evidence.false_positive_removed
            + self.far_small_detected_weight * evidence.far_small_detected
            - self.gt_object_penalty * evidence.num_gt_objects
        )


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class FrameCandidate:
    sample_token: SampleToken
    evidence: CandidateEvidence
    policy: CandidatePolicy = CandidatePolicy()

    def score(self) -> float:
        return self.policy.score(self.evidence)


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class CalloutPolicy:
    kinds: Tuple[CalloutKind, ...] = tuple(CalloutKind)


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class RenderPolicy:
    far_distance_m: DistanceMeters = DistanceMeters(30.0)
    very_far_distance_m: DistanceMeters = DistanceMeters(40.0)
    columns: Tuple[FigureColumn, ...] = FIGURE_COLUMNS
