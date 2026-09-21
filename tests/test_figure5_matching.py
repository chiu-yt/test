import math

import pytest

from tools.figure5_utils import (
    Box3D,
    Detection,
    DetectionClass,
    DetectionScore,
    DistanceMeters,
    FigureColumn,
    FrameRecord,
    MatchingPolicy,
    SampleToken,
    YawRadians,
)
from tools.figure5_utils.matching import (
    bev_iou,
    match_detections,
    match_frame,
    rotated_bev_corners,
)


def _detection(index, center, class_name=DetectionClass.CAR):
    return Detection(
        sample_token=SampleToken('sample-%d' % index),
        class_name=class_name,
        score=DetectionScore(0.9),
        box=Box3D(
            center=(center[0], center[1], 0.0),
            size=(2.0, 2.0, 1.0),
            yaw=YawRadians(0.0),
        ),
    )


def test_rotated_bev_corners_use_dx_dy_and_lidar_yaw():
    # Given a two-by-four box rotated counterclockwise by 90 degrees.
    box = Box3D(
        center=(3.0, 5.0, 0.0),
        size=(2.0, 4.0, 1.0),
        yaw=YawRadians(math.pi / 2.0),
    )

    # When its BEV corners are computed without detector operators.
    corners = rotated_bev_corners(box)

    # Then dx, dy, and LiDAR yaw define a counterclockwise rectangle.
    assert tuple(value for corner in corners for value in corner) == pytest.approx(
        (5.0, 4.0, 5.0, 6.0, 1.0, 6.0, 1.0, 4.0),
    )


def test_bev_iou_measures_rotated_rectangle_overlap():
    # Given equal axis-aligned boxes shifted by half their width.
    first = Box3D(
        center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 1.0), yaw=YawRadians(0.0),
    )
    second = Box3D(
        center=(1.0, 0.0, 0.0), size=(2.0, 2.0, 1.0), yaw=YawRadians(0.0),
    )

    # When their pure geometric BEV IoU is computed.
    overlap = bev_iou(first, second)

    # Then the two-square intersection and union are measured exactly.
    assert overlap == pytest.approx(1.0 / 3.0)
    rotated = Box3D(
        center=(4.0, -3.0, 0.0),
        size=(3.0, 1.0, 1.0),
        yaw=YawRadians(math.pi / 3.0),
    )
    assert bev_iou(rotated, rotated) == pytest.approx(1.0)


def test_match_detections_never_matches_wrong_class():
    # Given colocated ground truth and prediction from different classes.
    ground_truth = _detection(1, (0.0, 0.0), DetectionClass.CAR)
    prediction = _detection(2, (0.0, 0.0), DetectionClass.TRUCK)

    # When class-constrained Hungarian matching runs.
    result = match_detections((ground_truth,), (prediction,))

    # Then both detections remain unmatched despite perfect geometry.
    assert result.matches == ()
    assert result.unmatched_ground_truth == (ground_truth,)
    assert result.unmatched_predictions == (prediction,)


@pytest.mark.parametrize(
    ('ground_truth', 'predictions', 'expected_counts'),
    [
        ((), (_detection(1, (0.0, 0.0)),), (0, 1)),
        ((_detection(2, (0.0, 0.0)),), (), (1, 0)),
        ((), (), (0, 0)),
    ],
)
def test_match_detections_handles_empty_sets(
        ground_truth, predictions, expected_counts):
    # Given one or both matching inputs are empty.

    # When matching runs.
    result = match_detections(ground_truth, predictions)

    # Then every supplied detection is returned as unmatched.
    assert result.matches == ()
    assert len(result.unmatched_ground_truth) == expected_counts[0]
    assert len(result.unmatched_predictions) == expected_counts[1]


def test_hungarian_matching_selects_best_competing_prediction():
    # Given two same-class predictions competing for one ground truth.
    ground_truth = _detection(1, (0.0, 0.0))
    localized = _detection(2, (0.2, 0.0))
    displaced = _detection(3, (1.5, 0.0))

    # When one-to-one assignment minimizes center distance plus one minus IoU.
    result = match_detections((ground_truth,), (displaced, localized))

    # Then the better-localized prediction wins and the other stays unmatched.
    assert tuple(match.prediction for match in result.matches) == (localized,)
    assert result.unmatched_predictions == (displaced,)


def test_match_records_bev_localization_distance():
    # Given a valid prediction offset in both BEV axes.
    ground_truth = _detection(1, (1.0, 2.0))
    prediction = _detection(2, (1.3, 2.4))

    # When matching creates the typed correspondence.
    result = match_detections((ground_truth,), (prediction,))

    # Then localization is the Euclidean BEV center distance.
    assert len(result.matches) == 1
    assert result.matches[0].center_distance_m == pytest.approx(0.5)


def test_matching_policy_gates_center_distance_and_minimum_iou():
    # Given one prediction outside the center gate and one below the IoU gate.
    ground_truth = (
        _detection(1, (0.0, 0.0)),
        _detection(2, (10.0, 0.0)),
    )
    predictions = (
        _detection(3, (1.1, 0.0)),
        _detection(4, (10.9, 0.0)),
    )
    policy = MatchingPolicy(
        max_center_distance_m=DistanceMeters(1.0), min_bev_iou=0.4,
    )

    # When Hungarian matching applies both geometric validity gates.
    result = match_detections(ground_truth, predictions, policy)

    # Then neither clearly invalid pair is emitted as a match.
    assert result.matches == ()
    assert result.unmatched_ground_truth == ground_truth
    assert result.unmatched_predictions == predictions


def test_hungarian_ties_are_deterministic_and_ordered_by_ground_truth():
    # Given identical geometry for two ground truths and two predictions.
    ground_truth = (_detection(1, (0.0, 0.0)), _detection(2, (0.0, 0.0)))
    predictions = (_detection(3, (0.0, 0.0)), _detection(4, (0.0, 0.0)))

    # When tied Hungarian assignment is repeated.
    assignments = tuple(
        tuple(match.prediction.sample_token for match in match_detections(
            ground_truth, predictions,
        ).matches)
        for _ in range(5)
    )

    # Then the same input-order tie break and GT ordering are stable every time.
    assert assignments == (('sample-3', 'sample-4'),) * 5


def test_crowded_hungarian_ties_are_bounded_and_deterministic():
    # Given 200 same-class objects with tied pairs and mostly invalid cross-pairs.
    ground_truth = tuple(
        _detection(index, (float(index // 2) * 10.0, 0.0))
        for index in range(200)
    )
    predictions = tuple(
        _detection(index + 200, (float(index // 2) * 10.0, 0.0))
        for index in range(200)
    )
    expected = tuple('sample-%d' % (index + 200) for index in range(200))

    # When crowded Hungarian assignment is repeated.
    assignments = tuple(
        tuple(match.prediction.sample_token for match in match_detections(
            ground_truth, predictions,
        ).matches)
        for _ in range(2)
    )

    # Then tie-breaking neither overflows nor changes across runs.
    assert assignments == (expected, expected)


def test_match_frame_returns_typed_results_for_each_prediction_method():
    # Given one frame with a different prediction set for each Figure 5 method.
    ground_truth = _detection(1, (0.0, 0.0))
    source = _detection(2, (0.1, 0.0))
    codemerge = _detection(3, (5.0, 0.0))
    refuse_tta = _detection(4, (0.2, 0.0))
    frame = FrameRecord(
        sample_token=SampleToken('frame-token'),
        ground_truth=(ground_truth,),
        source_only=(source,),
        codemerge=(codemerge,),
        refuse_tta=(refuse_tta,),
    )

    # When the complete frame is matched.
    results = match_frame(frame)

    # Then each method has independent typed matches and unmatched detections.
    assert results.for_column(FigureColumn.SOURCE_ONLY).matches[0].prediction is source
    assert results.for_column(FigureColumn.CODEMERGE).unmatched_predictions == (codemerge,)
    assert results.for_column(FigureColumn.REFUSE_TTA).matches[0].prediction is refuse_tta
