import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest

from tools.figure5_utils import (
    NUSCENES_DETECTION_CLASSES,
    NUSCENES_DETECTION_PALETTE,
    Box3D,
    Callout,
    CalloutKind,
    CalloutPolicy,
    CandidateEvidence,
    CandidatePolicy,
    Detection,
    DetectionClass,
    DetectionMatch,
    FigureColumn,
    FrameCandidate,
    FrameRecord,
    MatchingPolicy,
    RenderPolicy,
    SceneToken,
    SampleToken,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE5_PACKAGE = REPO_ROOT / 'tools' / 'figure5_utils'


def _detection(sample_token, score=0.8):
    return Detection(
        sample_token=SampleToken(sample_token),
        class_name=DetectionClass.CAR,
        score=score,
        box=Box3D(center=(1.0, 2.0, 3.0), size=(4.0, 2.0, 1.5), yaw=0.25),
    )


def test_nuscenes_detection_order_and_palette_are_fixed():
    # Given the public nuScenes Figure 5 class metadata.
    expected_classes = (
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
    )

    # When class values and colors are read in display order.
    class_values = tuple(class_name.value for class_name in NUSCENES_DETECTION_CLASSES)
    colors = tuple(NUSCENES_DETECTION_PALETTE[class_name]
                   for class_name in NUSCENES_DETECTION_CLASSES)

    # Then all ten official detection classes have stable, distinct RGB colors.
    assert class_values == expected_classes
    assert len(colors) == len(expected_classes)
    assert len(set(colors)) == len(colors)
    assert all(len(color) == 3 and all(0 <= channel <= 255 for channel in color)
               for color in colors)


def test_box3d_copies_numpy_inputs_into_immutable_vectors():
    # Given mutable NumPy vectors owned by a future loader.
    center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    size = np.array([4.0, 2.0, 1.5], dtype=np.float32)

    # When a box contract is constructed and the loader mutates its arrays.
    box = Box3D(center=center, size=size, yaw=0.25)
    center[0] = 99.0
    size[0] = 99.0

    # Then the contract retains immutable value copies rather than array aliases.
    assert box.center == (1.0, 2.0, 3.0)
    assert box.size == (4.0, 2.0, 1.5)
    assert isinstance(box.center, tuple)
    with pytest.raises(FrozenInstanceError):
        box.yaw = 1.0


def test_detection_and_frame_preserve_sample_token_identity():
    # Given detections from one nuScenes sample in each fixed Figure 5 column.
    token = SampleToken('sample-token-001')
    detection = _detection(token)

    # When the immutable frame record is assembled.
    frame = FrameRecord(
        sample_token=token,
        ground_truth=(detection,),
        source_only=(detection,),
        codemerge=(detection,),
        refuse_tta=(detection,),
        scene_name='scene-001',
        scene_token=SceneToken('scene-token-001'),
        frame_index=7,
    )

    # Then identity, score, geometry, and fixed column access remain explicit.
    assert detection.sample_token == frame.sample_token
    assert detection.score == 0.8
    assert detection.box.center == (1.0, 2.0, 3.0)
    assert frame.detections(FigureColumn.CODEMERGE) == (detection,)
    assert frame.columns() == RenderPolicy().columns
    assert frame.scene_name == 'scene-001'
    assert frame.scene_token == SceneToken('scene-token-001')
    assert frame.frame_index == 7
    with pytest.raises(FrozenInstanceError):
        frame.sample_token = SampleToken('other-token')


def test_frame_optional_scene_metadata_defaults_to_none():
    # Given detections without optional scene-level metadata.
    detection = _detection('sample-token-without-scene')

    # When the immutable frame record is assembled.
    frame = FrameRecord(
        sample_token=detection.sample_token,
        ground_truth=(detection,),
        source_only=(),
        codemerge=(),
        refuse_tta=(),
    )

    # Then scene identity and sequence position remain explicitly optional.
    assert frame.scene_name is None
    assert frame.scene_token is None
    assert frame.frame_index is None


def test_matching_candidate_callout_and_render_defaults_are_stable():
    # Given default policies for the offline pipeline.
    matching = MatchingPolicy()
    candidate = CandidatePolicy()
    callout = CalloutPolicy()
    render = RenderPolicy()

    # When their public defaults are inspected.
    columns = tuple(column.value for column in render.columns)

    # Then matching, ranking, callout, range, and four-column contracts are fixed.
    assert matching.max_center_distance_m == 2.0
    assert matching.min_bev_iou == 0.0
    assert matching.class_aware is True
    assert candidate.recovered_from_source_weight == 3.0
    assert candidate.recovered_from_codemerge_weight == 2.0
    assert candidate.false_positive_removed_weight == 1.5
    assert candidate.far_small_detected_weight == 1.0
    assert candidate.gt_object_penalty == 0.1
    assert callout.kinds == tuple(CalloutKind)
    assert render.far_distance_m == 30.0
    assert render.very_far_distance_m == 40.0
    assert columns == ('GT', 'Source-only', 'CodeMerge', 'ReFuse-TTA')


def test_candidate_evidence_exposes_offline_selection_statistics():
    # Given the candidate evidence contract used by offline selection.
    expected_fields = (
        'recovered_from_source', 'recovered_from_codemerge',
        'false_positive_removed', 'far_small_detected', 'better_localization',
        'num_gt_objects', 'num_source_preds', 'num_codemerge_preds',
        'num_refuse_preds', 'involved_classes', 'primary_distances',
    )

    # When its immutable dataclass fields are inspected.
    field_names = tuple(field.name for field in fields(CandidateEvidence))

    # Then all required evidence buckets and diagnostics are present exactly once.
    assert field_names == expected_fields


def test_candidate_score_uses_exact_requested_formula():
    # Given evidence including localization and prediction-count diagnostics.
    evidence = CandidateEvidence(
        recovered_from_source=2,
        recovered_from_codemerge=1,
        false_positive_removed=2,
        far_small_detected=4,
        better_localization=99,
        num_gt_objects=10,
        num_source_preds=20,
        num_codemerge_preds=18,
        num_refuse_preds=16,
        involved_classes=(DetectionClass.CAR, DetectionClass.PEDESTRIAN),
        primary_distances=(12.0, 45.0),
    )

    # When the default candidate policy ranks the frame.
    frame_candidate = FrameCandidate(
        sample_token=SampleToken('sample-token-002'),
        evidence=evidence,
    )

    # Then localization remains evidence and only the exact requested terms score.
    assert frame_candidate.score() == 14.0


def test_match_contract_links_typed_detections():
    # Given a same-sample ground truth and prediction.
    ground_truth = _detection('sample-token-003', score=1.0)
    prediction = _detection('sample-token-003', score=0.7)

    # When a matching record references them.
    match = DetectionMatch(
        ground_truth=ground_truth,
        prediction=prediction,
        center_distance_m=0.5,
    )

    # Then downstream matching can trace sample, score, and geometry identity.
    assert match.prediction.sample_token == SampleToken('sample-token-003')
    assert match.prediction.score == 0.7


def test_callout_represents_physical_roi_class_distance_and_kind():
    # Given a far-range recovery selected for a physical image region.
    callout = Callout(
        roi=(10.0, 20.0, 30.0, 40.0),
        class_name=DetectionClass.PEDESTRIAN,
        distance_m=45.0,
        kind=CalloutKind.FAR_RANGE_RECOVERY,
    )

    # When callout metadata is consumed by a future renderer.
    values = tuple(kind.value for kind in CalloutKind)

    # Then it carries physical evidence rather than a method-column reference.
    assert callout.roi == (10.0, 20.0, 30.0, 40.0)
    assert callout.class_name is DetectionClass.PEDESTRIAN
    assert callout.distance_m == 45.0
    assert not hasattr(callout, 'column')
    assert values == (
        'far_range_recovery',
        'small_object_recovery',
        'false_positive_reduction',
        'better_localization',
    )


def test_figure5_contract_sources_parse_as_python_38():
    # Given every Python source in the new package.
    source_paths = sorted(FIGURE5_PACKAGE.glob('*.py'))

    # When Python 3.8 parses the source and dataclass options are inspected.
    modules = [
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=8)
        for path in source_paths
    ]
    dataclass_decorators = [
        decorator
        for module in modules
        for node in module.body
        if isinstance(node, ast.ClassDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == 'dataclass'
    ]

    # Then contracts avoid PEP 604 and Python-3.10-only dataclass slots.
    assert source_paths
    assert dataclass_decorators
    assert not any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
        for module in modules for node in ast.walk(module)
    )
    assert all(
        keyword.arg != 'slots'
        for decorator in dataclass_decorators
        for keyword in decorator.keywords
    )
