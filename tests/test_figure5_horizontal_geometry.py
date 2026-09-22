import numpy as np

from tools.figure5_utils import final_rendering
from tools.figure5_utils.domain import (
    Box3D, Detection, DetectionScore, SampleToken, YawRadians,
)
from tools.figure5_utils.palette import DetectionClass


def _corners(center, size, yaw):
    detection = Detection(
        SampleToken('geometry-regression'), DetectionClass.CAR, DetectionScore(0.9),
        Box3D(center, size, YawRadians(yaw)),
    )
    return final_rendering._box_bev_corners(detection)


def test_rotated_near_corner_box_is_excluded_when_only_aabbs_overlap():
    # Given the oracle regression whose AABB overlaps the unit reference crop.
    corners = _corners((1.5, 1.5, 0.0), (2.0, 0.1, 1.0), -np.pi / 4.0)
    crop = (0.0, 0.0, 1.0, 1.0)
    assert corners[:, 0].min() < crop[2]
    assert corners[:, 1].min() < crop[3]

    # When exact polygon intersection is evaluated, then the separated box is excluded.
    assert not final_rendering._corners_intersect_crop(corners, crop)


def test_box_touching_reference_boundary_counts_as_intersection():
    # Given a box whose left edge lies exactly on the reference crop's right edge.
    corners = _corners((1.5, 0.5, 0.0), (1.0, 0.2, 1.0), 0.0)

    # When intersection is evaluated, then boundary contact is retained.
    assert final_rendering._corners_intersect_crop(corners, (0.0, 0.0, 1.0, 1.0))
