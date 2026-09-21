"""Pure qualitative candidate evidence, not an official detection evaluator.

Far/small counts ReFuse-matched GT that are both at >=30m and in small classes.
Localization requires >=0.5m improvement over every baseline matching that GT.
FP reduction is the sum of per-class minimum excesses over ReFuse across both
baselines, capped by the minimum overall excess. Counts cannot locate individual
removed FPs: a class-level ROI is emitted only when ReFuse has zero unmatched
predictions of that class, and encloses all baseline unmatched boxes of it.
"""

from dataclasses import dataclass
from math import hypot
from typing import Dict, Final, Iterable, List, Tuple, TypedDict

from .domain import (
    Box3D, Callout, CalloutKind, CandidateEvidence, Detection, DistanceMeters,
    SampleToken,
)
from .domain import FrameRecord
from .matching import FrameMatchingResult, rotated_bev_corners
from .palette import DetectionClass
from .policies import CalloutPolicy, CandidatePolicy, RenderPolicy


SMALL_CLASSES: Final = frozenset((DetectionClass.PEDESTRIAN, DetectionClass.BICYCLE,
                                DetectionClass.MOTORCYCLE, DetectionClass.TRAFFIC_CONE))
LOCALIZATION_MARGIN_M: Final = 0.5
ROI_PADDING_M: Final = 2.0


class CandidateCsvRecord(TypedDict):
    sample_token: str
    scene_name: str
    scene_token: str
    frame_index: str
    score: float
    recovered_from_source: int
    recovered_from_codemerge: int
    false_positive_removed: int
    far_small_detected: int
    better_localization: int
    num_gt_objects: int
    num_source_preds: int
    num_codemerge_preds: int
    num_refuse_preds: int
    involved_classes: str
    primary_distances: str
    memberships: str


@dataclass(frozen=True)
class CandidateEvaluation:
    frame: FrameRecord
    evidence: CandidateEvidence
    score: float
    memberships: Tuple[CalloutKind, ...]
    available_callouts: Tuple[Callout, ...]

    @property
    def sample_token(self) -> SampleToken:
        return self.frame.sample_token

    @property
    def callouts(self) -> Tuple[Callout, ...]:
        return generate_callouts(self)

    @property
    def very_far_distances(self) -> Tuple[DistanceMeters, ...]:
        return tuple(distance for distance in self.evidence.primary_distances
                     if distance >= RenderPolicy().very_far_distance_m)

    def to_csv_record(self) -> CandidateCsvRecord:
        evidence = self.evidence
        return {
            'sample_token': self.sample_token,
            'scene_name': self.frame.scene_name or '',
            'scene_token': self.frame.scene_token or '',
            'frame_index': '' if self.frame.frame_index is None else str(self.frame.frame_index),
            'score': self.score,
            'recovered_from_source': evidence.recovered_from_source,
            'recovered_from_codemerge': evidence.recovered_from_codemerge,
            'false_positive_removed': evidence.false_positive_removed,
            'far_small_detected': evidence.far_small_detected,
            'better_localization': evidence.better_localization,
            'num_gt_objects': evidence.num_gt_objects,
            'num_source_preds': evidence.num_source_preds,
            'num_codemerge_preds': evidence.num_codemerge_preds,
            'num_refuse_preds': evidence.num_refuse_preds,
            'involved_classes': ';'.join(kind.value for kind in evidence.involved_classes),
            'primary_distances': ';'.join(str(value) for value in evidence.primary_distances),
            'memberships': ';'.join(kind.value for kind in self.memberships),
        }


@dataclass(frozen=True)
class CalloutLimitError(ValueError):
    limit: int

    def __str__(self) -> str:
        return 'Callout limit must be nonnegative; received %d' % self.limit


def _distance(detection: Detection) -> DistanceMeters:
    return DistanceMeters(hypot(*detection.box.center[:2]))


def _callout(kind: CalloutKind, anchor: Detection, boxes: Iterable[Box3D]) -> Callout:
    corners = tuple(corner for box in boxes for corner in rotated_bev_corners(box))
    horizontal, vertical = zip(*corners)
    return Callout(
        (min(horizontal) - ROI_PADDING_M, min(vertical) - ROI_PADDING_M,
         max(horizontal) + ROI_PADDING_M, max(vertical) + ROI_PADDING_M),
        anchor.class_name, _distance(anchor), kind,
    )


def evaluate_frame(frame: FrameRecord, matching: FrameMatchingResult,
                   policy: CandidatePolicy = CandidatePolicy()) -> CandidateEvaluation:
    """Consume matches for this frame with shared original GT object references."""
    source = {id(item.ground_truth): item for item in matching.source_only.matches}
    codemerge = {id(item.ground_truth): item for item in matching.codemerge.matches}
    recovered_source = recovered_codemerge = far_small = localized = 0
    relevant: Dict[int, Detection] = {}
    annotations: Dict[CalloutKind, List[Callout]] = {kind: [] for kind in CalloutKind}
    for item in matching.refuse_tta.matches:
        target = item.ground_truth
        identity = id(target)
        recovered_source += int(identity not in source)
        recovered_codemerge += int(identity not in codemerge)
        recovered = identity not in source or identity not in codemerge
        far = _distance(target) >= RenderPolicy().far_distance_m
        small = target.class_name in SMALL_CLASSES
        far_small += int(far and small)
        baselines = tuple(baseline[identity] for baseline in (source, codemerge)
                          if identity in baseline)
        improved = bool(baselines) and all(
            baseline.center_distance_m - item.center_distance_m >= LOCALIZATION_MARGIN_M
            for baseline in baselines
        )
        localized += int(improved)
        if recovered or far or small or improved:
            relevant[identity] = target
        eligibility = {
            CalloutKind.FAR_RANGE_RECOVERY: recovered and far,
            CalloutKind.SMALL_OBJECT_RECOVERY: recovered and small,
            CalloutKind.BETTER_LOCALIZATION: improved,
        }
        boxes = (target.box, item.prediction.box) + tuple(
            baseline.prediction.box for baseline in baselines)
        for kind, eligible in eligibility.items():
            if eligible:
                annotations[kind].append(_callout(kind, target, boxes))

    unmatched = (matching.source_only.unmatched_predictions,
                 matching.codemerge.unmatched_predictions,
                 matching.refuse_tta.unmatched_predictions)
    overall_excess = max(0, min(len(unmatched[0]), len(unmatched[1])) - len(unmatched[2]))
    class_excess = 0
    for class_name in DetectionClass:
        groups = tuple(tuple(item for item in group if item.class_name == class_name)
                       for group in unmatched)
        excess = max(0, min(len(groups[0]), len(groups[1])) - len(groups[2]))
        class_excess += excess
        if excess and overall_excess:
            for target in groups[0] + groups[1]:
                relevant[id(target)] = target
            if not groups[2]:
                anchor = min(groups[0] + groups[1], key=lambda item: (_distance(item), item.box.center))
                annotations[CalloutKind.FALSE_POSITIVE_REDUCTION].append(_callout(
                    CalloutKind.FALSE_POSITIVE_REDUCTION, anchor,
                    tuple(item.box for item in groups[0] + groups[1]),
                ))
    removed = min(overall_excess, class_excess)
    evidence = CandidateEvidence(
        recovered_from_source=recovered_source, recovered_from_codemerge=recovered_codemerge,
        false_positive_removed=removed, far_small_detected=far_small,
        better_localization=localized, num_gt_objects=len(frame.ground_truth),
        num_source_preds=len(frame.source_only), num_codemerge_preds=len(frame.codemerge),
        num_refuse_preds=len(frame.refuse_tta),
        involved_classes=tuple(sorted({item.class_name for item in relevant.values()},
                                      key=lambda kind: kind.value)),
        primary_distances=tuple(sorted((_distance(item) for item in relevant.values()), reverse=True)),
    )
    memberships = tuple(kind for kind in CalloutKind if annotations[kind] or
                        (kind == CalloutKind.FALSE_POSITIVE_REDUCTION and removed > 0))
    available = tuple(callout for kind in CalloutKind for callout in sorted(
        annotations[kind], key=lambda item: (-item.distance_m, item.class_name.value, item.roi)))
    return CandidateEvaluation(frame, evidence, policy.score(evidence), memberships, available)


def generate_callouts(candidate: CandidateEvaluation, policy: CalloutPolicy = CalloutPolicy(),
                      limit: int = 3) -> Tuple[Callout, ...]:
    """Prioritize requested kinds, then distance descending; never invent a ROI."""
    if limit < 0:
        raise CalloutLimitError(limit)
    priorities = {kind: index for index, kind in reversed(tuple(enumerate(policy.kinds)))}
    eligible = (item for item in candidate.available_callouts if item.kind in priorities)
    return tuple(sorted(eligible, key=lambda item: (
        priorities[item.kind], -item.distance_m, item.class_name.value, item.roi,
    )))[:limit]


def rank_candidates(candidates: Iterable[CandidateEvaluation]) -> Tuple[CandidateEvaluation, ...]:
    """Rank all supplied evaluated frames; this does not select final scenes."""
    return tuple(sorted(candidates, key=lambda item: (-item.score, item.sample_token)))


def bucket_candidates(candidates: Iterable[CandidateEvaluation]) -> Dict[CalloutKind, Tuple[CandidateEvaluation, ...]]:
    ranked = rank_candidates(candidates)
    return {kind: tuple(item for item in ranked if kind in item.memberships) for kind in CalloutKind}
