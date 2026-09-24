"""Opt-in contract: VERSION=k4_v1; validate_k4_config(cfg, world_size=1).

Validation returns True only for enabled formal K4, False for legacy/versionless
SPCRA, and raises ValueError for incompatible formal execution. AST checks bind
the CPU composition APIs to MOS and TransFusion without importing CUDA modules.
"""

from __future__ import annotations

import ast
import copy
from types import SimpleNamespace
import unittest

import yaml

from test_spcra_k4_runtime import ROOT, load_contract_module


CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
INCOMPATIBLE = ('HARD_PSEUDO_MINING', 'PS_FILTER', 'ADAPTIVE_CAP', 'DEPTH_FILTER',
                'DEPTH_ENTROPY_FILTER', 'GEOMETRY_FILTER', 'CONFLICT_FILTER', 'RG_PLM')


def parsed(path):
    return ast.parse((ROOT / path).read_text())


def function(tree, name):
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if not matches:
        raise AssertionError(f'Missing formal integration function: {name}')
    return matches[0]


def calls(tree):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted_name(node.value) + '.' + node.attr
    return ast.dump(node)


def call_names(tree):
    return [node.func.attr if isinstance(node.func, ast.Attribute)
            else dotted_name(node.func) for node in calls(tree)]


def formal_config():
    return yaml.safe_load((ROOT / 'tools/cfgs/nuscenes_models/bevfusion_spcra_k4.yaml').read_text())


class TestK4Integration(unittest.TestCase):
    def test_version_dispatch_when_formal_is_explicitly_enabled(self):
        """Given legacy/disabled/formal configs; when validated; then opt in only explicitly."""
        runtime = load_contract_module('spcra_k4_runtime')
        for version, enabled, expected in ((None, True, False), ('legacy', True, False),
                                           ('k4_v1', False, False), ('k4_v1', True, True)):
            with self.subTest(version=version, enabled=enabled):
                config = formal_config()
                config['TTA']['SPCRA']['ENABLED'] = enabled
                if version is None:
                    config['TTA']['SPCRA'].pop('VERSION')
                else:
                    config['TTA']['SPCRA']['VERSION'] = version
                self.assertIs(runtime.validate_k4_config(config, world_size=1), expected)

    def test_rejection_when_formal_execution_can_replace_evidence(self):
        """Given incompatible formal features; when validated; then reject before inference."""
        runtime = load_contract_module('spcra_k4_runtime')
        paths = [('TTA', 'MOS_SETTING', 'AGGREGATION_ENABLED'),
                 ('TTA', 'FIGURE6_CAPTURE', 'ENABLED'), ('TTA', 'DPO_MATCHER', 'ENABLED')]
        paths += [('SELF_TRAIN', name, 'ENABLED') for name in (*INCOMPATIBLE, 'MEMORY_ENSEMBLE')]
        for section, name, key in paths:
            with self.subTest(feature=(section, name, key)):
                config = formal_config()
                config[section][name][key] = True
                with self.assertRaises(ValueError):
                    runtime.validate_k4_config(config, world_size=1)
        with self.assertRaises(ValueError):
            runtime.validate_k4_config(formal_config(), world_size=2)

    def test_legacy_validation_when_legacy_features_are_enabled(self):
        """Given versionless legacy MOS; when formal validation runs; then leave it alone."""
        runtime = load_contract_module('spcra_k4_runtime')
        config = formal_config()
        config['TTA']['SPCRA'].pop('VERSION')
        config['SELF_TRAIN']['MEMORY_ENSEMBLE']['ENABLED'] = True
        config['TTA']['FIGURE6_CAPTURE']['ENABLED'] = True
        self.assertIs(runtime.validate_k4_config(config, world_size=2), False)

    def test_dispatch_at_spcra_seam_when_formal_branch_is_added(self):
        """Given MOS's existing seam; when formal dispatch is wired; then retain legacy code."""
        tree = parsed('pcdet/tta_methods/mos.py')
        seam = function(tree, '_run_spcra_diagnostic')
        branches = [node for node in seam.body if isinstance(node, ast.If)
                    and '_spcra_k4_enabled' in call_names(node.test)]
        self.assertTrue(branches, 'MOS SPCRA seam lacks explicit formal version dispatch')
        formal = ast.Module(body=branches[0].body, type_ignores=[])
        self.assertIn('_run_spcra_k4', call_names(formal))
        self.assertTrue(any(isinstance(node, ast.Return) for node in ast.walk(formal)))
        self.assertIn('compute_spcra_reliability', call_names(seam))
        self.assertIn('run_current_k4', call_names(function(tree, '_run_spcra_k4')))
        bridge = parsed('pcdet/tta_methods/spcra_k4_mos.py')
        self.assertIn('run_k4_views', call_names(function(bridge, 'run_current_k4')))
        self.assertIn('validate_k4_config', call_names(function(tree, '__init__')))

    def test_bevfusion_bridge_uses_project_absolute_imports(self):
        """Given synthetic-package tests; when loading the bridge; then imports stay resolvable."""
        tree = parsed('pcdet/tta_methods/spcra_k4_bevfusion.py')
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        inference_imports = [node for node in imports
                             if any(alias.name == 'forward_without_annotations' for alias in node.names)]
        self.assertEqual(len(inference_imports), 1)
        self.assertEqual(inference_imports[0].level, 0)
        self.assertEqual(inference_imports[0].module, 'pcdet.utils.inference_utils')

    def test_mos_version_predicate_when_legacy_or_formal_is_selected(self):
        """Given MOS's real predicate; when versions vary; then preserve legacy dispatch."""
        predicate = function(parsed('pcdet/tta_methods/mos.py'), '_spcra_k4_enabled')
        namespace = {}
        exec(compile(ast.Module(body=[predicate], type_ignores=[]), '<mos-predicate>', 'exec'), namespace)
        for version, enabled, expected in ((None, True, False), ('legacy', True, False),
                                           ('k4_v1', False, False), ('k4_v1', True, True)):
            with self.subTest(version=version, enabled=enabled):
                spcra: dict[str, bool | str] = {'ENABLED': enabled}
                if version is not None:
                    spcra['VERSION'] = version
                owner = SimpleNamespace(tta_cfg={'SPCRA': spcra})
                self.assertIs(namespace['_spcra_k4_enabled'](owner), expected)

    def test_temp_shell_is_skipped_when_formal_is_active(self):
        """Given formal mode and a dataset; when evaluating shell guards; then skip shell."""
        optimize = function(parsed('pcdet/tta_methods/mos.py'), 'optimize')
        guards = [node for node in ast.walk(optimize) if isinstance(node, ast.If)
                  and '_init_temp_model' in call_names(ast.Module(body=node.body, type_ignores=[]))]
        self.assertTrue(guards, 'Expected guarded lazy shell initialization')
        for formal in (True, False):
            owner = SimpleNamespace(temp_model_shell=None, dataset=SimpleNamespace(),
                                    _spcra_k4_enabled=lambda: formal)
            enabled = all(eval(compile(ast.Expression(guard.test), '<shell-guard>', 'eval'),
                               {'self': owner}) for guard in guards)
            self.assertEqual(enabled, not formal, 'Formal K4 must not allocate an aggregation shell')

    def test_weight_helpers_are_connected_when_mos_trains_transfusion(self):
        """Given tested helpers; when production integrates them; then do not leave them unused."""
        mos = parsed('pcdet/tta_methods/mos.py')
        self.assertIn('build_k4_pseudo_info', call_names(function(mos, 'save_pseudo_label_batch')))
        self.assertIn('inject_k4_pseudo_labels', call_names(function(mos, '_inject_pseudo_labels')))
        head = parsed('pcdet/models/dense_heads/transfusion_head.py')
        self.assertIn('positive_query_weights', call_names(function(head, 'get_targets_single')))

    def test_optimize_ownership_when_formal_helpers_are_composed(self):
        """Given MOS optimize; when K4 is added; then retain training/backward/global-step order."""
        optimize = function(parsed('pcdet/tta_methods/mos.py'), 'optimize')
        ordered = sorted(calls(optimize), key=lambda node: (node.lineno, node.col_offset))
        names = [dotted_name(node.func) for node in ordered]
        expected = ['self._run_spcra_diagnostic', 'self._inject_pseudo_labels',
                    'self.model.train', "batch_dict['optimizer'].zero_grad", 'TTA_augmentation',
                    'self.model', 'loss.mean', 'final_loss.backward', 'self.model.update_global_step']
        positions = [names.index(dotted_name(ast.parse(name, mode='eval').body)) for name in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(names.count('final_loss.backward'), 1)
        self.assertFalse(any(name.endswith('.step') for name in names))
        weighted = [node for node in ast.walk(optimize) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'final_loss' for target in node.targets)]
        self.assertEqual(ast.dump(weighted[0].value), ast.dump(ast.parse('loss * tar_loss_weight', mode='eval').body))
        weight = [node for node in ast.walk(optimize) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'tar_loss_weight' for target in node.targets)]
        self.assertTrue(any(isinstance(node, ast.Attribute)
                            and dotted_name(node) == 'cfg.SELF_TRAIN.TAR'
                            for node in ast.walk(weight[0].value)))

    def test_formal_yaml_when_launching_the_paper_protocol(self):
        """Given a dedicated formal YAML; when loaded; then require all fixed controls."""
        path = ROOT / 'tools/cfgs/nuscenes_models/bevfusion_spcra_k4.yaml'
        self.assertTrue(path.is_file(), 'Missing formal bevfusion_spcra_k4.yaml')
        config = yaml.safe_load(path.read_text())
        self.assertEqual(config['CLASS_NAMES'], CLASS_NAMES)
        spcra = config['TTA']['SPCRA']
        for key, expected in {'ENABLED': True, 'VERSION': 'k4_v1', 'K': 4,
                              'RELIABILITY_FLOOR': 0., 'RELIABILITY_CLAMP_MAX': 1.,
                              'TARGET_CLASSES': CLASS_NAMES, 'TOPK_PER_CLASS': 0}.items():
            self.assertEqual(spcra[key], expected, key)
        self.assertFalse(config['TTA']['MOS_SETTING']['AGGREGATION_ENABLED'])
        self.assertFalse(config['TTA']['FIGURE6_CAPTURE']['ENABLED'])
        self.assertFalse(config['SELF_TRAIN']['MEMORY_ENSEMBLE']['ENABLED'])
        self.assertTrue(config['MODEL']['TTA_FUSION_ADAPTER']['ENABLED'])
        self.assertTrue(config['MODEL']['TTA_FUSION_ADAPTER']['SG_DFA']['ENABLED'])
        self.assertGreater(config['MODEL']['TTA_FUSION_ADAPTER']['RESIDUAL_SCALE_INIT'], 0)
        self.assertTrue(config['TTA']['FREEZE']['ADAPTER_ONLY'])
        self.assertEqual(config['TTA']['TTA_STRENGTH'], 'mid')
        self.assertTrue(spcra['CAMERA_RESCUE_ENABLED'])
        self.assertEqual((spcra['CAMERA_LOW_SCORE'], spcra['CAMERA_THRESH'], spcra['CAMERA_SUPPORT_SCALE']),
                         (.1, .6, 4.))
        self.assertEqual(len(config['DATA_CONFIG']['TTA_DATA_AUGMENTOR']['AUG_CONFIG_LIST']), 4)
        capture = config['TTA']['FIGURE7_CAPTURE']
        self.assertFalse(capture['ENABLED'])
        self.assertEqual(capture['STABLE_CAPACITY'] + capture['VARIABLE_CAPACITY'], 20)
        corruption = config['DATA_CONFIG']['CORRUPTION']
        self.assertTrue(corruption['ENABLED'])
        self.assertTrue(corruption['LIDAR_SPARSITY']['ENABLED'])
        self.assertEqual(corruption['LIDAR_SPARSITY']['MODE'], 'density_dec_global')
        self.assertEqual(corruption['LIDAR_SPARSITY']['SEVERITY'], 5)
        for name in ('IMAGE_FOG', 'IMAGE_STYLE', 'IMAGE_GEOMETRY', 'LIDAR_FOG'):
            self.assertFalse(corruption[name]['ENABLED'])
        self.assertTrue(config['DATA_CONFIG']['CAMERA_CONFIG']['USE_CAMERA'])
        self.assertEqual(config['DATA_CONFIG']['MAX_SWEEPS'], 5)
        runtime = load_contract_module('spcra_k4_runtime')
        self.assertIs(runtime.validate_k4_config(copy.deepcopy(config), world_size=1), True)


if __name__ == '__main__':
    unittest.main()
