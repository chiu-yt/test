import hashlib
import importlib
import json
import pickle
from types import ModuleType
from typing import Callable, Dict, Union

import numpy as np
import pytest


def _write(path, records):
    with path.open('wb') as stream:
        pickle.dump(records, stream)
    return path


def _prediction(token, x=1.0, size=(4, 5, 6)):
    return dict(metadata=dict(token=token), name=np.array(['car']),
                score=np.array([0.8]),
                boxes_lidar=np.array([[x, 2, 3, size[0], size[1], size[2], 0.25, 99, 99]]))


def _info(token, size=(4, 5, 6)):
    return dict(token=token, gt_names=np.array(['car']),
                gt_boxes=np.array([[1, 2, 3, size[0], size[1], size[2],
                                    0.25, np.nan, np.nan]]),
                lidar_path='key.bin', sweeps=[])


def _paths(tmp_path, records):
    return tuple(_write(tmp_path / ('run%d.pkl' % index), rows)
                 for index, rows in enumerate(records))


def test_frames_join_by_token_when_results_are_reordered(tmp_path):
    """Given independently ordered runs, when loaded, then tokens own geometry."""
    io = importlib.import_module('tools.figure5_utils.io')
    paths = _paths(tmp_path, [[_prediction('b', 20), _prediction('a', 10)],
                              [_prediction('a', 30), _prediction('b', 40)],
                              [_prediction('b', 60), _prediction('a', 50)]])
    info_path = _write(tmp_path / 'infos.pkl', [_info('a'), _info('b')])
    dataset = io.load_artifacts(io.ResultPaths(*paths), info_path)
    frame = dataset.frames['a']
    assert [frame.source_only[0].box.center[0], frame.codemerge[0].box.center[0],
            frame.refuse_tta[0].box.center[0]] == [10, 30, 50]
    assert frame.ground_truth[0].box.size == (4, 5, 6)
    assert frame.ground_truth[0].box.yaw == 0.25
    assert frame.ground_truth[0].score == 1.0
    assert frame.scene_token is None


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'mismatch', 'info_duplicate'])
def test_tokens_rejected_when_artifact_identity_is_invalid(tmp_path, failure):
    """Given malformed identities, when loading, then raise a typed error."""
    io = importlib.import_module('tools.figure5_utils.io')
    rows = [_prediction('a')]
    infos = [_info('a')]
    errors = dict(missing=io.MissingTokenError, duplicate=io.DuplicateTokenError,
                  mismatch=io.TokenMismatchError, info_duplicate=io.DuplicateTokenError)
    if failure == 'missing':
        rows[0]['metadata'] = {}
    if failure == 'duplicate':
        rows *= 2
    if failure == 'mismatch':
        rows = [_prediction('b')]
    if failure == 'info_duplicate':
        infos *= 2
    paths = _paths(tmp_path, [rows, [_prediction('a')], [_prediction('a')]])
    with pytest.raises(errors[failure]):
        io.load_artifacts(io.ResultPaths(*paths), _write(tmp_path / 'infos.pkl', infos))


def test_scene_metadata_uses_sample_chain_not_json_order(tmp_path):
    """Given shuffled SDK tables, when loaded, then frame index follows next."""
    io = importlib.import_module('tools.figure5_utils.io')
    (tmp_path / 'sample.json').write_text(json.dumps([
        dict(token='b', scene_token='scene', next='', prev='a'),
        dict(token='a', scene_token='scene', next='b', prev='')]))
    (tmp_path / 'scene.json').write_text(json.dumps([
        dict(token='scene', name='scene-1', first_sample_token='a')]))
    paths = _paths(tmp_path, [[_prediction('b')]] * 3)
    dataset = io.load_artifacts(io.ResultPaths(*paths),
                                _write(tmp_path / 'infos.pkl', [_info('b')]), tmp_path)
    frame = dataset.frames['b']
    assert (frame.scene_token, frame.scene_name, frame.frame_index) == ('scene', 'scene-1', 1)


def test_points_match_native_geometry_and_token_seeded_drop(tmp_path):
    """Given binary sweeps, when reconstructed, then native rows/time survive."""
    io = importlib.import_module('tools.figure5_utils.io')
    key = np.array([[0, 0, 1, 11, 9], [5, 6, 7, 12, 9]], dtype=np.float32)
    key.tofile(tmp_path / 'key.bin')
    rows = np.array([[0.5, -0.5, 3, 21, 9], [1, 0, 3, 22, 9],
                     [2, 3, 4, 23, 9]], dtype=np.float32)
    rows.tofile(tmp_path / 'sweep.bin')
    transform = np.array([[0, -1, 0, 10], [1, 0, 0, 20],
                          [0, 0, 1, 30], [0, 0, 0, 1]], dtype=np.float64)
    info = _info('a')
    info['sweeps'] = [dict(lidar_path='sweep.bin', transform_matrix=transform,
                           time_lag=float(index + 1)) for index in range(6)]
    paths = _paths(tmp_path, [[_prediction('a')]] * 3)
    dataset = io.load_artifacts(io.ResultPaths(*paths),
                                _write(tmp_path / 'infos.pkl', [info]))
    loaded = io.load_frame(dataset, 'a', io.PointSource(tmp_path))
    selected = np.random.RandomState(1024).choice(6, 4, replace=False)
    expected = np.concatenate([np.column_stack((key[:, :4], [0, 0]))] + [
        np.array([[10, 21, 33, 22, index + 1], [7, 22, 34, 23, index + 1]])
        for index in selected]).astype(np.float32)
    seed = int.from_bytes(hashlib.sha256(b'1024:a').digest()[:4], 'little')
    keep = np.random.RandomState(seed).choice(10, 7, replace=False)
    np.testing.assert_array_equal(loaded.dense_points, expected)
    np.testing.assert_array_equal(loaded.sparse_points, expected[keep])
    np.testing.assert_array_equal(io.load_frame(dataset, 'a', io.PointSource(tmp_path)).sparse_points,
                                  loaded.sparse_points)
    assert loaded.frame.sample_token == 'a'
    assert loaded.sweep_indices == tuple(selected)


def test_precomputed_points_bypass_raw_reconstruction(tmp_path):
    """Given token-named NPY input, when requested, then use it without dropping again."""
    io = importlib.import_module('tools.figure5_utils.io')
    expected = np.arange(15, dtype=np.float32).reshape(3, 5)
    np.save(tmp_path / 'a.npy', expected)
    paths = _paths(tmp_path, [[_prediction('a')]] * 3)
    dataset = io.load_artifacts(io.ResultPaths(*paths),
                                _write(tmp_path / 'infos.pkl', [_info('a')]))
    loaded = io.load_frame(dataset, 'a', io.PointSource(tmp_path, sparse_directory=tmp_path))
    np.testing.assert_array_equal(loaded.sparse_points, expected)
    assert loaded.dense_points is None
    with pytest.raises(io.MissingTokenError):
        io.load_frame(dataset, 'absent', io.PointSource(tmp_path))


def test_insufficient_sweeps_are_rejected(tmp_path):
    """Given fewer than four historical sweeps, when reconstructing, then reject."""
    io = importlib.import_module('tools.figure5_utils.io')
    paths = _paths(tmp_path, [[_prediction('a')]] * 3)
    dataset = io.load_artifacts(io.ResultPaths(*paths),
                                _write(tmp_path / 'infos.pkl', [_info('a')]))
    with pytest.raises(io.ArtifactFormatError):
        io.load_frame(dataset, 'a', io.PointSource(tmp_path))


@pytest.mark.parametrize('field,value', [
    ('name', np.array(['car', 'truck'])),
    ('score', np.array([np.nan])),
    ('boxes_lidar', np.ones((1, 6))),
    ('name', np.array(['unrecognized'])),
])
def test_invalid_detection_arrays_raise_typed_error(tmp_path, field, value):
    """Given malformed array fields, when parsed, then fail at the boundary."""
    io = importlib.import_module('tools.figure5_utils.io')
    row = _prediction('a')
    row[field] = value
    with pytest.raises(io.ArtifactFormatError):
        io.load_predictions(_write(tmp_path / 'result.pkl', [row]))


@pytest.mark.parametrize(('ground_truth', 'dimension_index', 'value'), [
    (True, 3, 0.0),
    (True, 4, -1.0),
    (True, 5, 0.0),
    (False, 3, -1.0),
    (False, 4, 0.0),
    (False, 5, -1.0),
])
def test_nonpositive_retained_box_dimensions_raise_typed_error(
        tmp_path, ground_truth, dimension_index, value):
    # Given a retained GT or prediction with a nonpositive dx, dy, or dz.
    io = importlib.import_module('tools.figure5_utils.io')
    size = [4.0, 5.0, 6.0]
    size[dimension_index - 3] = value
    row = _info('a', size) if ground_truth else _prediction('a', size=size)
    loader = io.load_validation_infos if ground_truth else io.load_predictions

    # When the native artifact is parsed at the I/O boundary.
    with pytest.raises(io.ArtifactFormatError):
        loader(_write(tmp_path / 'artifact.pkl', [row]))

def test_empty_detections_and_ignore_gt_are_supported(tmp_path):
    """Given empty predictions and ignored GT, when joined, then preserve emptiness."""
    io = importlib.import_module('tools.figure5_utils.io')
    row = dict(metadata=dict(token='a'), name=np.array([]),
               score=np.array([]), boxes_lidar=np.empty((0, 9)))
    info = _info('a', size=(0, 5, 6))
    info['gt_names'] = np.array(['ignore'])
    paths = _paths(tmp_path, [[row]] * 3)
    frame = io.load_artifacts(io.ResultPaths(*paths),
                              _write(tmp_path / 'infos.pkl', [info])).frames['a']
    assert frame.ground_truth == frame.source_only == frame.codemerge == frame.refuse_tta == ()


def test_sweep_without_transform_preserves_boundary_points(tmp_path):
    """Given an untransformed sweep, when read, then only strict ego-square points vanish."""
    points = importlib.import_module('tools.figure5_utils.points')
    types = importlib.import_module('tools.figure5_utils.input_types')
    raw = np.array([[0, 0, 0, 10, 0], [-1, 0, 2, 11, 0],
                    [0, 1, 2, 12, 0]], dtype=np.float32)
    raw.tofile(tmp_path / 'sweep.bin')
    actual, times = points.get_sweep(types.SweepInfo(tmp_path / 'sweep.bin', None, 0.2), tmp_path)
    np.testing.assert_array_equal(actual, raw[1:, :4])
    np.testing.assert_array_equal(times, [[0.2], [0.2]])


def test_native_density_function_has_exact_row_parity(tmp_path):
    """Given native sparsity code, when both paths run, then every selected row agrees."""
    import ast
    from pathlib import Path

    io = importlib.import_module('tools.figure5_utils.io')
    native_path = Path(__file__).resolve().parents[1] / 'pcdet/datasets/augmentor/augmentor_utils.py'
    tree = ast.parse(native_path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'apply_lidar_sparsity')
    namespace: Dict[str, Union[ModuleType, Callable[..., np.ndarray]]] = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(native_path), 'exec'), namespace)
    raw = np.arange(55, dtype=np.float32).reshape(11, 5)
    raw.tofile(tmp_path / 'key.bin')
    info = _info('a')
    info['sweeps'] = [dict(lidar_path='key.bin', transform_matrix=None, time_lag=0.1)] * 4
    paths = _paths(tmp_path, [[_prediction('a')]] * 3)
    dataset = io.load_artifacts(io.ResultPaths(*paths), _write(tmp_path / 'infos.pkl', [info]))
    loaded = io.load_frame(dataset, 'a', io.PointSource(tmp_path))
    seed = int.from_bytes(hashlib.sha256(b'1024:a').digest()[:4], 'little')
    native_sparsity = namespace['apply_lidar_sparsity']
    assert callable(native_sparsity)
    expected = native_sparsity(
        loaded.dense_points, severity=5, rng=np.random.RandomState(seed), mode='density_dec_global')
    np.testing.assert_array_equal(loaded.sparse_points, expected)


def test_random_keep_s5_is_deterministic_and_scales_intensity(tmp_path):
    """Given random_keep S5, when reconstructed, then rows and intensity follow native policy."""
    io = importlib.import_module('tools.figure5_utils.io')
    raw = np.arange(500, dtype=np.float32).reshape(100, 5)
    raw.tofile(tmp_path / 'key.bin')
    info = _info('a')
    info['sweeps'] = [dict(lidar_path='key.bin', transform_matrix=None, time_lag=0.1)] * 4
    paths = _paths(tmp_path, [[_prediction('a')]] * 3)
    dataset = io.load_artifacts(io.ResultPaths(*paths), _write(tmp_path / 'infos.pkl', [info]))
    source = io.PointSource(tmp_path, sparsity_mode='random_keep', sparsity_severity=5)

    first = io.load_frame(dataset, 'a', source)
    second = io.load_frame(dataset, 'a', source)

    np.testing.assert_array_equal(first.sparse_points, second.sparse_points)
    assert len(first.sparse_points) >= int(len(first.dense_points) * 0.2)
    dense_intensities = set(np.round(first.dense_points[:, 3] * 0.85, 5))
    assert set(np.round(first.sparse_points[:, 3], 5)).issubset(dense_intensities)


@pytest.mark.parametrize('severity', [0, 6])
def test_reconstruction_rejects_invalid_sparsity_severity(tmp_path, severity):
    """Given an unsupported severity, when loading points, then reject at the boundary."""
    io = importlib.import_module('tools.figure5_utils.io')
    np.arange(25, dtype=np.float32).reshape(5, 5).tofile(tmp_path / 'key.bin')
    info = _info('a')
    info['sweeps'] = [dict(lidar_path='key.bin', transform_matrix=None, time_lag=0.1)] * 4
    paths = _paths(tmp_path, [[_prediction('a')]] * 3)
    dataset = io.load_artifacts(io.ResultPaths(*paths), _write(tmp_path / 'infos.pkl', [info]))
    with pytest.raises(io.ArtifactFormatError):
        io.load_frame(dataset, 'a', io.PointSource(tmp_path, sparsity_severity=severity))


def test_import_has_no_heavy_dependencies_and_supports_python38():
    """Given the input package, when imported fresh, then no model stack is loaded."""
    import ast
    from pathlib import Path
    import subprocess
    import sys

    package = Path(__file__).resolve().parents[1] / 'tools/figure5_utils'
    for filename in ('io.py', 'points.py', 'scene_io.py', 'input_types.py'):
        ast.parse((package / filename).read_text(), feature_version=(3, 8))
    subprocess.run([sys.executable, '-c',
                    'import sys; import tools.figure5_utils.io; '
                    'assert not {"torch", "pcdet", "nuscenes"}.intersection(sys.modules)'],
                   check=True)
