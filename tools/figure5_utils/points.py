"""NumPy-only five-sweep reconstruction with selectable LiDAR sparsity.

Sweep selection uses a fresh RandomState(seed) per frame, independent of access
order. This is deterministic reconstruction, not a claim to replay a training or
evaluation DataLoader's worker RNG history. For exact run points, supply saved
token-named NPY arrays. Density dropping matches the native token hash exactly;
random_keep uses the native policy with a deterministic token-derived RNG.
"""
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .input_types import ArtifactFormatError, MissingTokenError, PointSource, SweepInfo, ValidationInfo


_DROP_RATIOS = (0.06, 0.12, 0.18, 0.24, 0.30)
_KEEP_RATIOS = (0.90, 0.80, 0.70, 0.60, 0.50)
_INTENSITY_SCALES = (0.98, 0.95, 0.92, 0.88, 0.85)


def _lidar(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 5:
        raise ArtifactFormatError(str(path), 'expected float32 rows with five channels')
    return raw.reshape(-1, 5)[:, :4]


def get_sweep(sweep: SweepInfo, data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the sweep ego square before applying the homogeneous transform."""
    points = _lidar(data_root / sweep.lidar_path)
    points = points[~((np.abs(points[:, 0]) < 1.0) & (np.abs(points[:, 1]) < 1.0))].T
    if sweep.transform_matrix is not None:
        points[:3, :] = sweep.transform_matrix.dot(
            np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]
    times = sweep.time_lag * np.ones((points.shape[1], 1))
    return points.T, times


def load_points(info: ValidationInfo, source: PointSource
                 ) -> Tuple[Optional[np.ndarray], np.ndarray, Tuple[int, ...]]:
    """Return dense XYZI-time, sparse XYZI-time and selected info sweep indices."""
    if source.sparse_directory is not None:
        if Path(info.token).name != info.token or info.token in ('.', '..'):
            raise ArtifactFormatError(info.token, 'token must be a plain filename for NPY lookup')
        path = source.sparse_directory / (info.token + '.npy')
        if not path.is_file():
            raise MissingTokenError(str(source.sparse_directory), info.token)
        points = np.load(str(path), allow_pickle=False)
        if (not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 5
                or points.dtype.kind != 'f' or not np.isfinite(points).all()):
            raise ArtifactFormatError(str(path), 'expected finite floating [N,5] XYZI-time array')
        return None, points, ()
    if len(info.sweeps) < 4:
        raise ArtifactFormatError(info.token, 'five-sweep reconstruction requires four historical sweeps')
    indices = tuple(int(index) for index in
                    np.random.RandomState(source.seed).choice(len(info.sweeps), 4, replace=False))
    key = _lidar(source.data_root / info.lidar_path)
    point_parts = [key]
    time_parts = [np.zeros((key.shape[0], 1))]
    for index in indices:
        points, times = get_sweep(info.sweeps[index], source.data_root)
        point_parts.append(points)
        time_parts.append(times)
    dense = np.concatenate(point_parts, axis=0)
    times = np.concatenate(time_parts, axis=0).astype(dense.dtype)
    dense = np.concatenate((dense, times), axis=1)
    if source.sparsity_severity < 1 or source.sparsity_severity > 5:
        raise ArtifactFormatError(info.token, 'sparsity severity must be in [1, 5]')
    digest = hashlib.sha256(('%d:%s' % (source.seed, info.token)).encode('utf-8')).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], byteorder='little'))
    severity_index = source.sparsity_severity - 1
    if source.sparsity_mode == 'density_dec_global':
        drop_ratio = _DROP_RATIOS[severity_index]
        keep = rng.choice(len(dense), size=len(dense) - int(len(dense) * drop_ratio),
                          replace=False)
        sparse = dense[keep]
    elif source.sparsity_mode == 'random_keep':
        keep_mask = rng.rand(len(dense)) < _KEEP_RATIOS[severity_index]
        min_keep = min(max(64, int(len(dense) * 0.2)), len(dense))
        if keep_mask.sum() < min_keep:
            keep_mask[:] = False
            keep_mask[rng.choice(len(dense), size=min_keep, replace=False)] = True
        sparse = dense[keep_mask].copy()
        if sparse.shape[1] > 3:
            sparse[:, 3] *= _INTENSITY_SCALES[severity_index]
    else:
        raise ArtifactFormatError(info.token, 'unknown sparsity mode: %s' % source.sparsity_mode)
    return dense, sparse.astype(dense.dtype, copy=False), indices
