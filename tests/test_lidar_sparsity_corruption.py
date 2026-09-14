"""Regression tests for LiDAR sparsity corruption modes."""

import ast
import copy
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUGMENTOR_UTILS_PATH = (
    REPO_ROOT / "pcdet" / "datasets" / "augmentor" / "augmentor_utils.py"
)
OFFICIAL_DROP_FRACTIONS = [0.06, 0.12, 0.18, 0.24, 0.30]
N_POINTS = 1000
RANDOM_KEEP_S5_RATIO = 0.50
RANDOM_KEEP_S5_INTENSITY_SCALE = 0.85


def _extract_module_function(path, root_name, namespace):
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    assignments = [node for node in module.body if isinstance(node, ast.Assign)]
    assert root_name in definitions, "%s not found in %s" % (root_name, path)

    wanted = {root_name}
    growing = True
    while growing:
        growing = False
        for name in list(wanted):
            for node in ast.walk(definitions[name]):
                if isinstance(node, ast.Name) and node.id in definitions:
                    if node.id not in wanted:
                        wanted.add(node.id)
                        growing = True

    body = list(assignments) + [definitions[name] for name in sorted(wanted)]
    extracted = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace[root_name]


APPLY_LIDAR_SPARSITY = _extract_module_function(
    AUGMENTOR_UTILS_PATH,
    "apply_lidar_sparsity",
    {"np": np, "math": math, "copy": copy},
)


def _make_points(n_points=N_POINTS, dtype=np.float32):
    idx = np.arange(n_points, dtype=dtype)
    x = idx * dtype(0.01)
    y = (idx % dtype(97.0)) * dtype(0.02)
    z = (idx % dtype(13.0)) * dtype(0.03)
    intensity = (idx + dtype(1.0)) * dtype(0.5)
    return np.stack([x, y, z, intensity], axis=1).astype(dtype, copy=False)


def _row_bytes(points):
    return {row.tobytes() for row in points}


def test_density_dec_global_uses_official_drop_fractions():
    points = _make_points()
    for severity, drop in enumerate(OFFICIAL_DROP_FRACTIONS, start=1):
        rng = np.random.RandomState(1234)
        out = APPLY_LIDAR_SPARSITY(
            points, severity=severity, rng=rng, mode="density_dec_global"
        )
        expected_keep = N_POINTS - int(N_POINTS * drop)
        assert out.shape[0] == expected_keep, (severity, drop, out.shape[0])


def test_density_dec_global_s5_retains_seventy_percent():
    points = _make_points()
    out = APPLY_LIDAR_SPARSITY(
        points, severity=5, rng=np.random.RandomState(7), mode="density_dec_global"
    )
    assert out.shape[0] == 700
    assert math.isclose(out.shape[0] / points.shape[0], 0.70, rel_tol=0, abs_tol=1e-9)


def test_density_dec_global_preserves_dtype_and_feature_shape():
    points = _make_points()
    for severity in range(1, 6):
        out = APPLY_LIDAR_SPARSITY(
            points,
            severity=severity,
            rng=np.random.RandomState(severity),
            mode="density_dec_global",
        )
        assert out.dtype == points.dtype
        assert out.shape[1] == points.shape[1]


def test_density_dec_global_keeps_retained_rows_unchanged_including_intensity():
    points = _make_points()
    valid_rows = _row_bytes(points)
    for severity in range(1, 6):
        out = APPLY_LIDAR_SPARSITY(
            points,
            severity=severity,
            rng=np.random.RandomState(99 + severity),
            mode="density_dec_global",
        )
        assert all(row.tobytes() in valid_rows for row in out)


def test_density_dec_global_seeded_rng_is_deterministic():
    points = _make_points()
    first = APPLY_LIDAR_SPARSITY(
        points, severity=3, rng=np.random.RandomState(2024), mode="density_dec_global"
    )
    second = APPLY_LIDAR_SPARSITY(
        points, severity=3, rng=np.random.RandomState(2024), mode="density_dec_global"
    )
    assert np.array_equal(first, second)


def test_random_keep_mode_is_selectable_and_matches_default():
    points = _make_points()
    explicit = APPLY_LIDAR_SPARSITY(
        points, severity=5, rng=np.random.RandomState(11), mode="random_keep"
    )
    default = APPLY_LIDAR_SPARSITY(points, severity=5, rng=np.random.RandomState(11))
    assert np.array_equal(explicit, default)


def test_random_keep_s5_keeps_about_half_and_scales_intensity():
    points = _make_points()
    out = APPLY_LIDAR_SPARSITY(
        points, severity=5, rng=np.random.RandomState(5), mode="random_keep"
    )
    keep_ratio = out.shape[0] / points.shape[0]
    assert abs(keep_ratio - RANDOM_KEEP_S5_RATIO) < 0.08

    intensity_by_x = {row[0].tobytes(): row[3] for row in points}
    for row in out:
        source_intensity = intensity_by_x[row[0].tobytes()]
        expected = source_intensity * np.float32(RANDOM_KEEP_S5_INTENSITY_SCALE)
        assert np.isclose(row[3], expected, rtol=1e-5, atol=1e-6)


def test_unknown_mode_raises_value_error():
    points = _make_points()
    with pytest.raises(ValueError):
        APPLY_LIDAR_SPARSITY(
            points, severity=1, rng=np.random.RandomState(0), mode="not_a_real_mode"
        )
