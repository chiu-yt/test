from dataclasses import replace
import math

import pytest

from tools.figure5_utils.domain import (
    Box3D, CalloutKind, Detection, DetectionScore, FrameRecord, SampleToken,
    YawRadians,
)
from tools.figure5_utils.palette import DetectionClass
from tools.figure5_utils.matching import match_frame
from tools.figure5_utils.policies import CalloutPolicy, CandidatePolicy
from tools.figure5_utils.candidates import (
    CalloutLimitError, evaluate_frame, rank_candidates, bucket_candidates, generate_callouts,
)


def detection(distance: float, kind: DetectionClass = DetectionClass.CAR) -> Detection:
    return Detection(SampleToken('frame'), kind, DetectionScore(0.9),
                     Box3D((distance, 0.0, 0.0), (2.0, 4.0, 1.0), YawRadians(0.0)))


def test_metrics_when_recovery_and_localization_coexist() -> None:
    """Given distinct recoveries; when evaluated; then counts and score are exact."""
    far = detection(40.0, DetectionClass.PEDESTRIAN)
    small = detection(10.0, DetectionClass.BICYCLE)
    shared = detection(20.0)
    frame = FrameRecord(SampleToken('frame'), (far, small, shared),
                        (detection(21.0),), (small, detection(21.5)), (far, small, shared))
    result = evaluate_frame(frame, match_frame(frame))
    assert (result.evidence.recovered_from_source,
            result.evidence.recovered_from_codemerge,
            result.evidence.better_localization,
            result.evidence.far_small_detected) == (2, 1, 1, 1)
    assert result.score == pytest.approx(8.7)
    assert result.evidence.involved_classes == (DetectionClass.BICYCLE, DetectionClass.CAR,
                                                DetectionClass.PEDESTRIAN)
    assert result.evidence.primary_distances == (40.0, 20.0, 10.0)
    assert set(result.memberships) == {CalloutKind.FAR_RANGE_RECOVERY,
                                     CalloutKind.SMALL_OBJECT_RECOVERY,
                                     CalloutKind.BETTER_LOCALIZATION}
    assert result.to_csv_record()['num_gt_objects'] == 3
    assert result.to_csv_record()['num_source_preds'] == 1
    assert result.to_csv_record()['num_codemerge_preds'] == 2
    assert result.to_csv_record()['num_refuse_preds'] == 3
    assert result.callouts[0].distance_m == 40.0
    assert result.very_far_distances == (40.0,)


def test_identity_when_equal_ground_truth_objects_are_distinct() -> None:
    """Given equal-valued GT objects; when matched; then recovery uses identity."""
    first, second = detection(30.0), detection(30.0)
    frame = FrameRecord(SampleToken('frame'), (first, second), (first,), (), (first, second))
    result = evaluate_frame(frame, match_frame(frame))
    assert result.evidence.recovered_from_source == 1
    assert result.evidence.recovered_from_codemerge == 2


@pytest.mark.parametrize('distance,expected', [(29.99, 0), (30.0, 0), (40.0, 0)])
def test_far_boundary_when_large_object_recovered(distance: float, expected: int) -> None:
    """Given a car at the range boundary; when evaluated; then far is inclusive."""
    target = detection(distance)
    frame = FrameRecord(SampleToken('frame'), (target,), (), (), (target,))
    result = evaluate_frame(frame, match_frame(frame))
    assert result.evidence.far_small_detected == expected
    assert (CalloutKind.FAR_RANGE_RECOVERY in result.memberships) == (distance >= 30.0)
    assert bool(result.callouts) == (distance >= 30.0)


@pytest.mark.parametrize('distance,expected', [(10.0, 0), (29.99, 0), (30.0, 1), (40.0, 1)])
def test_far_small_when_pedestrian_is_matched(distance: float, expected: int) -> None:
    target = detection(distance, DetectionClass.PEDESTRIAN)
    frame = FrameRecord(SampleToken('frame'), (target,), (), (), (target,))
    result = evaluate_frame(frame, match_frame(frame))
    assert result.evidence.far_small_detected == expected
    assert result.score == pytest.approx(4.9 + expected)
    assert result.evidence.involved_classes == (DetectionClass.PEDESTRIAN,)
    assert result.evidence.primary_distances == (distance,)


def test_far_small_when_far_pedestrian_is_unmatched() -> None:
    target = detection(40.0, DetectionClass.PEDESTRIAN)
    frame = FrameRecord(SampleToken('frame'), (target,), (), (), ())
    result = evaluate_frame(frame, match_frame(frame))
    assert result.evidence.far_small_detected == 0


@pytest.mark.parametrize('offset,expected', [(0.49, 0), (0.5, 1)])
def test_localization_margin_when_baselines_share_gt(offset: float, expected: int) -> None:
    """Given shared GT; when improvement crosses 0.5m; then count it once."""
    target = detection(0.0)
    frame = FrameRecord(SampleToken('frame'), (target,), (detection(offset),), (), (target,))
    result = evaluate_frame(frame, match_frame(frame))
    assert result.evidence.better_localization == expected


def test_fp_reduction_when_both_baselines_have_same_class_excess() -> None:
    """Given baseline-only unmatched boxes; when evaluated; then use minimum excess."""
    first, second = detection(10.0), detection(20.0)
    frame = FrameRecord(SampleToken('frame'), (), (first, second), (first,), ())
    result = evaluate_frame(frame, match_frame(frame))
    assert result.evidence.false_positive_removed == 1
    assert result.memberships == (CalloutKind.FALSE_POSITIVE_REDUCTION,)
    assert result.callouts[0].roi == (7.0, -4.0, 23.0, 4.0)
    assert result.evidence.involved_classes == (DetectionClass.CAR,)
    assert result.evidence.primary_distances == (20.0, 10.0)
    assert result.score == 1.5


def test_fp_reduction_when_classes_disagree_is_not_invented() -> None:
    """Given different unmatched classes; when evaluated; then no shared reduction."""
    frame = FrameRecord(SampleToken('frame'), (), (detection(10.0),),
                        (detection(10.0, DetectionClass.TRUCK),), ())
    result = evaluate_frame(frame, match_frame(frame))
    assert result.evidence.false_positive_removed == 0
    assert result.callouts == ()


def test_roi_when_box_rotated_encloses_physical_extent() -> None:
    """Given a rotated recovered box; when evaluated; then ROI includes box and padding."""
    target = detection(30.0)
    target = replace(target, box=replace(target.box, yaw=YawRadians(math.pi / 2)))
    frame = FrameRecord(SampleToken('frame'), (target,), (), (), (target,))
    result = evaluate_frame(frame, match_frame(frame))
    assert result.callouts[0].roi == pytest.approx((26.0, -3.0, 34.0, 3.0))


def test_ranking_when_scores_tie_uses_token_and_keeps_multiple_buckets() -> None:
    """Given tied frames; when ranked and bucketed; then order and membership persist."""
    target = detection(40.0, DetectionClass.TRAFFIC_CONE)
    frame = FrameRecord(SampleToken('z'), (target,), (), (), (target,))
    candidates = tuple(evaluate_frame(replace(frame, sample_token=SampleToken(token)),
                                      match_frame(frame)) for token in ('z', 'a'))
    ranked = rank_candidates(candidates)
    assert tuple(item.sample_token for item in ranked) == ('a', 'z')
    assert bucket_candidates(ranked)[CalloutKind.FAR_RANGE_RECOVERY] == ranked
    assert bucket_candidates(ranked)[CalloutKind.SMALL_OBJECT_RECOVERY] == ranked


def test_empty_frame_when_custom_score_policy_used() -> None:
    """Given no matching gains; when evaluated; then no fabricated annotations."""
    frame = FrameRecord(SampleToken('empty'), (detection(1.0),), (), (), ())
    result = evaluate_frame(frame, match_frame(frame), CandidatePolicy(gt_object_penalty=2.0))
    assert result.score == -2.0
    assert result.callouts == ()
    assert result.memberships == ()


def test_callout_priority_when_user_requests_small_first() -> None:
    """Given overlapping themes; when prioritized; then honor order and limit."""
    target = detection(40.0, DetectionClass.PEDESTRIAN)
    frame = FrameRecord(SampleToken('frame'), (target,), (), (), (target,))
    candidate = evaluate_frame(frame, match_frame(frame))
    callouts = generate_callouts(candidate, CalloutPolicy((CalloutKind.SMALL_OBJECT_RECOVERY,
                                                         CalloutKind.FAR_RANGE_RECOVERY)), 1)
    assert tuple(item.kind for item in callouts) == (CalloutKind.SMALL_OBJECT_RECOVERY,)


def test_callout_limit_when_negative_is_rejected() -> None:
    """Given an evaluated frame; when the limit is negative; then reject it."""
    frame = FrameRecord(SampleToken('frame'), (), (), (), ())
    candidate = evaluate_frame(frame, match_frame(frame))
    with pytest.raises(CalloutLimitError):
        generate_callouts(candidate, limit=-1)


def test_fp_count_when_some_unmatched_remain_has_no_fabricated_roi() -> None:
    """Given aggregate reduction only; when evaluated; then retain count without ROI."""
    baseline = (detection(10.0), detection(20.0))
    frame = FrameRecord(SampleToken('frame'), (), baseline, baseline, (detection(15.0),))
    candidate = evaluate_frame(frame, match_frame(frame))
    assert candidate.evidence.false_positive_removed == 1
    assert candidate.memberships == (CalloutKind.FALSE_POSITIVE_REDUCTION,)
    assert candidate.callouts == ()


def test_localization_when_refuse_worse_than_one_baseline_is_not_improved() -> None:
    """Given mixed baseline comparisons; when evaluated; then avoid cherry-picking."""
    target = detection(0.0)
    frame = FrameRecord(SampleToken('frame'), (target,), (detection(1.5),),
                        (target,), (detection(0.5),))
    candidate = evaluate_frame(frame, match_frame(frame))
    assert candidate.evidence.better_localization == 0


def test_ranking_when_scores_differ_uses_score_before_token() -> None:
    """Given score order opposes token order; when ranked; then score wins."""
    target = detection(40.0)
    better = FrameRecord(SampleToken('z'), (target,), (), (), (target,))
    worse = FrameRecord(SampleToken('a'), (target,), (), (), ())
    ranked = rank_candidates(evaluate_frame(frame, match_frame(frame)) for frame in (worse, better))
    assert tuple(item.sample_token for item in ranked) == ('z', 'a')
