import ast
import hashlib
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
NUSCENES_DATASET_PATH = (
    REPO_ROOT / 'pcdet' / 'datasets' / 'nuscenes' / 'nuscenes_dataset.py'
)
SAR_CONFIG_PATH = (
    REPO_ROOT / 'tools' / 'cfgs' / 'nuscenes_models' / 'bevfusion_sar.yaml'
)


def _parse_dataset():
    return ast.parse(NUSCENES_DATASET_PATH.read_text(), str(NUSCENES_DATASET_PATH))


def _load_rng_builder():
    tree = _parse_dataset()
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == '_build_lidar_corruption_rng'
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {'hashlib': hashlib, 'np': np}
    exec(compile(module, str(NUSCENES_DATASET_PATH), 'exec'), namespace)
    return namespace[function.name]


def _choice(sample_id):
    rng = _load_rng_builder()(sample_id, 1024)
    return rng.choice(1000, size=700, replace=False)


def test_sample_rng_is_stable_and_independent_of_global_numpy_state():
    np.random.seed(1)
    first = _choice('sample-token')
    np.random.seed(999)
    np.random.random(4096)
    second = _choice('sample-token')
    assert np.array_equal(first, second)


def test_sample_rng_changes_with_sample_identity():
    assert not np.array_equal(_choice('sample-a'), _choice('sample-b'))


def test_rng_builder_uses_sha256_and_not_python_hash():
    function = next(
        node for node in _parse_dataset().body
        if isinstance(node, ast.FunctionDef)
        and node.name == '_build_lidar_corruption_rng'
    )
    source = ast.unparse(function)
    assert 'hashlib.sha256' in source
    assert 'hash(' not in source
    assert 'np.random.RandomState' in source


def test_getitem_seeds_only_density_dec_global_and_passes_rng():
    tree = _parse_dataset()
    dataset_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'NuScenesDataset'
    )
    getitem = next(
        node for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == '__getitem__'
    )
    source = ast.unparse(getitem)
    assert "sparsity_mode == 'density_dec_global'" in source
    assert '_build_lidar_corruption_rng' in source
    assert 'rng=sparsity_rng' in source
    assert 'sparsity_rng = None' in source


def test_sar_config_sets_density_corruption_seed_1024():
    config = yaml.safe_load(SAR_CONFIG_PATH.read_text())
    sparsity = config['DATA_CONFIG']['CORRUPTION']['LIDAR_SPARSITY']
    assert sparsity['MODE'] == 'density_dec_global'
    assert sparsity['SEED'] == 1024
