from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from scipy.optimize import linear_sum_assignment

from .domain import Box3D, Detection, DetectionMatch, DistanceMeters, FigureColumn, FrameRecord
from .policies import MatchingPolicy


Point2D = Tuple[float, float]
Line2D = Tuple[Point2D, Point2D]
BevCorners = Tuple[Point2D, Point2D, Point2D, Point2D]


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class MatchingResult:
    matches: Tuple[DetectionMatch, ...]
    unmatched_ground_truth: Tuple[Detection, ...]
    unmatched_predictions: Tuple[Detection, ...]


@dataclass(frozen=True)  # noqa: SLOTS_OK - dataclass slots require Python 3.10.
class FrameMatchingResult:
    source_only: MatchingResult
    codemerge: MatchingResult
    refuse_tta: MatchingResult

    def for_column(self, column: FigureColumn) -> MatchingResult:
        return {
            FigureColumn.SOURCE_ONLY: self.source_only,
            FigureColumn.CODEMERGE: self.codemerge,
            FigureColumn.REFUSE_TTA: self.refuse_tta,
        }[column]


def rotated_bev_corners(box: Box3D) -> BevCorners:
    half_dx = box.size[0] / 2.0
    half_dy = box.size[1] / 2.0
    local_corners = (
        (-half_dx, -half_dy),
        (half_dx, -half_dy),
        (half_dx, half_dy),
        (-half_dx, half_dy),
    )
    cosine = math.cos(box.yaw)
    sine = math.sin(box.yaw)
    center_x, center_y = box.center[:2]
    rotated = tuple(
        (
            center_x + local_x * cosine - local_y * sine,
            center_y + local_x * sine + local_y * cosine,
        )
        for local_x, local_y in local_corners
    )
    return rotated[0], rotated[1], rotated[2], rotated[3]


def _cross(first: Point2D, second: Point2D) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _inside(point: Point2D, edge_start: Point2D, edge_end: Point2D) -> bool:
    edge = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
    relative = (point[0] - edge_start[0], point[1] - edge_start[1])
    return _cross(edge, relative) >= -1e-12


def _line_intersection(segment_line: Line2D, edge_line: Line2D) -> Point2D:
    segment_start, segment_end = segment_line
    edge_start, edge_end = edge_line
    segment = (
        segment_end[0] - segment_start[0],
        segment_end[1] - segment_start[1],
    )
    edge = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
    offset = (edge_start[0] - segment_start[0], edge_start[1] - segment_start[1])
    denominator = _cross(segment, edge)
    if abs(denominator) <= 1e-15:
        return segment_end
    fraction = _cross(offset, edge) / denominator
    return (
        segment_start[0] + fraction * segment[0],
        segment_start[1] + fraction * segment[1],
    )


def _clip_polygon(subject: Sequence[Point2D], clip: BevCorners) -> Tuple[Point2D, ...]:
    output = tuple(subject)
    for edge_index, edge_start in enumerate(clip):
        if not output:
            break
        edge_end = clip[(edge_index + 1) % len(clip)]
        clipped: List[Point2D] = []
        previous = output[-1]
        previous_inside = _inside(previous, edge_start, edge_end)
        for current in output:
            current_inside = _inside(current, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    clipped.append(_line_intersection(
                        (previous, current), (edge_start, edge_end),
                    ))
                clipped.append(current)
            elif previous_inside:
                clipped.append(_line_intersection(
                    (previous, current), (edge_start, edge_end),
                ))
            previous = current
            previous_inside = current_inside
        output = tuple(clipped)
    return output


def _polygon_area(polygon: Sequence[Point2D]) -> float:
    if len(polygon) < 3:
        return 0.0
    twice_area = sum(
        point[0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * point[1]
        for index, point in enumerate(polygon)
    )
    return abs(twice_area) / 2.0


def bev_iou(first: Box3D, second: Box3D) -> float:
    first_corners = rotated_bev_corners(first)
    second_corners = rotated_bev_corners(second)
    first_area = _polygon_area(first_corners)
    second_area = _polygon_area(second_corners)
    intersection = _polygon_area(_clip_polygon(first_corners, second_corners))
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return min(1.0, max(0.0, intersection / union))


def _center_distance(first: Detection, second: Detection) -> float:
    delta_x = first.box.center[0] - second.box.center[0]
    delta_y = first.box.center[1] - second.box.center[1]
    return math.hypot(delta_x, delta_y)


def _match_class(
        ground_truth: Sequence[Detection],
        predictions: Sequence[Detection],
        policy: MatchingPolicy) -> Tuple[Tuple[int, int, float], ...]:
    assignment_count = min(len(ground_truth), len(predictions))
    max_valid_cost = float(policy.max_center_distance_m) + 1.0
    invalid_cost = (max_valid_cost + 1.0) * (assignment_count + 1)
    max_pair_rank = max(
        1,
        (max(len(ground_truth), len(predictions)) - 1) ** 2
        * (len(predictions) + 1) + len(predictions),
    )
    # Keep the complete assignment's secondary cost at or below 1e-12.
    tie_denominator = assignment_count * max_pair_rank
    valid_pairs: List[List[bool]] = []
    costs: List[List[float]] = []
    for ground_index, ground_detection in enumerate(ground_truth):
        validity_row: List[bool] = []
        cost_row: List[float] = []
        for prediction_index, prediction in enumerate(predictions):
            distance = _center_distance(ground_detection, prediction)
            overlap = bev_iou(ground_detection.box, prediction.box)
            valid = (
                distance <= float(policy.max_center_distance_m)
                and overlap >= policy.min_bev_iou
            )
            pair_rank = (
                (ground_index - prediction_index) ** 2 * (len(predictions) + 1)
                + prediction_index
            )
            tie_break = 1e-12 * pair_rank / tie_denominator
            validity_row.append(valid)
            cost_row.append(distance + 1.0 - overlap + tie_break if valid else invalid_cost)
        valid_pairs.append(validity_row)
        costs.append(cost_row)

    matched_rows, matched_columns = linear_sum_assignment(costs)
    matches: List[Tuple[int, int, float]] = []
    for ground_index, prediction_index in zip(matched_rows, matched_columns):
        ground_position = int(ground_index)
        prediction_position = int(prediction_index)
        if valid_pairs[ground_position][prediction_position]:
            matches.append((
                ground_position,
                prediction_position,
                _center_distance(
                    ground_truth[ground_position], predictions[prediction_position],
                ),
            ))
    return tuple(matches)


def match_detections(
        ground_truth: Sequence[Detection],
        predictions: Sequence[Detection],
        policy: MatchingPolicy = MatchingPolicy()) -> MatchingResult:
    indexed_matches: List[Tuple[int, int, float]] = []
    class_names = sorted(
        {detection.class_name for detection in ground_truth}.union(
            detection.class_name for detection in predictions
        ),
        key=lambda class_name: class_name.value,
    )
    for class_name in class_names:
        ground_indices = tuple(
            index for index, detection in enumerate(ground_truth)
            if detection.class_name is class_name
        )
        prediction_indices = tuple(
            index for index, detection in enumerate(predictions)
            if detection.class_name is class_name
        )
        if not ground_indices or not prediction_indices:
            continue
        class_ground_truth = tuple(ground_truth[index] for index in ground_indices)
        class_predictions = tuple(predictions[index] for index in prediction_indices)
        for ground_index, prediction_index, distance in _match_class(
                class_ground_truth, class_predictions, policy):
            indexed_matches.append((
                ground_indices[ground_index],
                prediction_indices[prediction_index],
                distance,
            ))

    indexed_matches.sort(key=lambda item: (item[0], item[1]))
    matched_ground = {item[0] for item in indexed_matches}
    matched_predictions = {item[1] for item in indexed_matches}
    return MatchingResult(
        matches=tuple(
            DetectionMatch(
                ground_truth=ground_truth[ground_index],
                prediction=predictions[prediction_index],
                center_distance_m=DistanceMeters(distance),
            )
            for ground_index, prediction_index, distance in indexed_matches
        ),
        unmatched_ground_truth=tuple(
            detection for index, detection in enumerate(ground_truth)
            if index not in matched_ground
        ),
        unmatched_predictions=tuple(
            detection for index, detection in enumerate(predictions)
            if index not in matched_predictions
        ),
    )


def match_frame(
        frame: FrameRecord,
        policy: MatchingPolicy = MatchingPolicy()) -> FrameMatchingResult:
    return FrameMatchingResult(
        source_only=match_detections(frame.ground_truth, frame.source_only, policy),
        codemerge=match_detections(frame.ground_truth, frame.codemerge, policy),
        refuse_tta=match_detections(frame.ground_truth, frame.refuse_tta, policy),
    )
