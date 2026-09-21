"""Native OpenPCDet artifact inputs. Only load trusted local pickle files.

All three runs must contain exactly the validation-info token set. No positional
alignment, class filtering, score thresholding, or box-coordinate conversion is
performed. nuScenes info ``ignore`` annotations are omitted because they are not
detection classes; all ten detection classes are retained.
"""
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .domain import Box3D, Detection, DetectionScore, FrameRecord, SampleToken, SceneToken, YawRadians
from .palette import DetectionClass
from .input_types import (
    ArtifactDataset, ArtifactFormatError, DuplicateTokenError, LoadedFrame,
    MissingTokenError, PointSource, ResultPaths, SceneMetadata, SweepInfo,
    TokenMismatchError, ValidationInfo,
)
from .scene_io import load_scene_metadata


def _records(path: Path) -> list:
    with path.open('rb') as stream:
        rows = pickle.load(stream)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ArtifactFormatError(str(path), 'expected a list of records')
    return rows


def _token(raw, context: str) -> SampleToken:
    if not isinstance(raw, str) or not raw.strip():
        raise MissingTokenError(context, 'expected a nonempty token')
    return SampleToken(raw)


def _detections(row, token: SampleToken, ground_truth: bool) -> Tuple[Detection, ...]:
    names_key, boxes_key = ('gt_names', 'gt_boxes') if ground_truth else ('name', 'boxes_lidar')
    try:
        names = np.asarray(row[names_key])
        boxes = np.asarray(row[boxes_key], dtype=np.float64)
        scores = np.ones(len(names)) if ground_truth else np.asarray(row['score'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactFormatError(token, 'invalid detection arrays') from error
    if names.ndim != 1 or boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ArtifactFormatError(token, 'expected names [N] and boxes [N, >=7]')
    if len(boxes) != len(names) or scores.shape != names.shape:
        raise ArtifactFormatError(token, 'names, boxes and scores have different lengths')
    if not np.isfinite(boxes[:, :7]).all() or not np.isfinite(scores).all():
        raise ArtifactFormatError(token, 'non-finite geometry or scores')
    result = []
    for name, box, score in zip(names, boxes, scores):
        if ground_truth and name == 'ignore':
            continue
        if np.any(box[3:6] <= 0.0):
            raise ArtifactFormatError(token, 'box dimensions must be positive')
        try:
            detection_class = DetectionClass(str(name))
        except ValueError as error:
            raise ArtifactFormatError(token, 'unknown detection class: %s' % name) from error
        result.append(Detection(token, detection_class, DetectionScore(float(score)), Box3D(
            (float(box[0]), float(box[1]), float(box[2])),
            (float(box[3]), float(box[4]), float(box[5])), YawRadians(float(box[6])))))
    return tuple(result)


def load_predictions(path: Path) -> Mapping[SampleToken, Tuple[Detection, ...]]:
    """Index native result.pkl rows strictly by metadata.token."""
    result: Dict[SampleToken, Tuple[Detection, ...]] = {}
    for row in _records(Path(path)):
        metadata = row.get('metadata')
        if not isinstance(metadata, dict):
            raise MissingTokenError(str(path), 'missing metadata.token')
        token = _token(metadata.get('token'), str(path))
        if token in result:
            raise DuplicateTokenError(str(path), token)
        if 'token' in row and row['token'] != token:
            raise TokenMismatchError(str(path), 'token differs from metadata.token')
        result[token] = _detections(row, token, False)
    return MappingProxyType(result)


def _sweep(row, token: SampleToken) -> SweepInfo:
    try:
        path = Path(row['lidar_path'])
        time = float(row['time_lag'])
        transform = row['transform_matrix']
        matrix = None if transform is None else np.array(transform, dtype=np.float64, copy=True)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactFormatError(token, 'invalid sweep metadata') from error
    if not np.isfinite(time) or (matrix is not None and
                                (matrix.shape != (4, 4) or not np.isfinite(matrix).all())):
        raise ArtifactFormatError(token, 'expected finite time and 4x4 transform')
    return SweepInfo(path, matrix, time)


def load_validation_infos(path: Path) -> Mapping[SampleToken, ValidationInfo]:
    """Parse validation ground truth and LiDAR reconstruction metadata."""
    result: Dict[SampleToken, ValidationInfo] = {}
    for row in _records(Path(path)):
        token = _token(row.get('token'), str(path))
        if token in result:
            raise DuplicateTokenError(str(path), token)
        try:
            lidar_path = Path(row['lidar_path'])
            sweeps = tuple(_sweep(sweep, token) for sweep in row['sweeps'])
        except (KeyError, TypeError) as error:
            raise ArtifactFormatError(token, 'missing lidar_path or sweeps') from error
        scene_token = row.get('scene_token')
        scene_name = row.get('scene_name')
        frame_index = row.get('frame_index')
        if scene_token is not None:
            scene_token = SceneToken(_token(scene_token, token))
        if scene_name is not None and not isinstance(scene_name, str):
            raise ArtifactFormatError(token, 'scene_name must be a string')
        if frame_index is not None and (not isinstance(frame_index, int) or frame_index < 0):
            raise ArtifactFormatError(token, 'frame_index must be nonnegative')
        result[token] = ValidationInfo(token, _detections(row, token, True), lidar_path,
                                       sweeps, SceneMetadata(scene_token, scene_name, frame_index))
    return MappingProxyType(result)


def load_artifacts(paths: ResultPaths, info_path: Path,
                   table_directory: Optional[Path] = None) -> ArtifactDataset:
    """Join three complete runs; optionally enrich scenes from SDK JSON tables."""
    infos = load_validation_infos(info_path)
    runs = tuple(load_predictions(path) for path in
                 (paths.source_only, paths.codemerge, paths.refuse_tta))
    for path, run in zip((paths.source_only, paths.codemerge, paths.refuse_tta), runs):
        if set(run) != set(infos):
            raise TokenMismatchError(str(path), 'missing=%s extra=%s' % (
                sorted(set(infos) - set(run)), sorted(set(run) - set(infos))))
    scenes = {} if table_directory is None else load_scene_metadata(table_directory)
    frames = {}
    for token, info in infos.items():
        if table_directory is not None and token not in scenes:
            raise MissingTokenError(str(table_directory), token)
        scene = scenes.get(token, info.scene)
        if token in scenes:
            for field in ('scene_token', 'scene_name', 'frame_index'):
                value = getattr(info.scene, field)
                if value is not None and value != getattr(scene, field):
                    raise TokenMismatchError(token, 'info and scene tables disagree: ' + field)
        frames[token] = FrameRecord(token, info.ground_truth, runs[0][token],
                                    runs[1][token], runs[2][token], scene.scene_name,
                                    scene.scene_token, scene.frame_index)
    return ArtifactDataset(MappingProxyType(frames), infos)


def load_frame(dataset: ArtifactDataset, token: str, source: PointSource) -> LoadedFrame:
    """Return the aligned frame with reconstructed or supplied sparse points."""
    from .points import load_points

    sample_token = _token(token, 'load_frame')
    if sample_token not in dataset.frames:
        raise MissingTokenError('load_frame', token)
    dense, sparse, indices = load_points(dataset.infos[sample_token], source)
    return LoadedFrame(dataset.frames[sample_token], sparse, dense, indices)
