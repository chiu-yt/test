"""Formal full-method controls are mandatory; versionless configs stay untouched."""

import ast
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np

from test_spcra_k4_integration import formal_config
from test_spcra_k4_runtime import ROOT, FakeModel, load_contract_module


def bevfusion_namespace():
    namespace = {'__name__': __name__, 'dataclass': dataclass, 'np': np,
                 'K4Policy': load_contract_module('spcra_k4_core').K4Policy,
                 'K4ConfigurationError': load_contract_module('spcra_k4_config').K4ConfigurationError}
    for filename, names in (
        ('spcra_k4_context.py', ('DensityMapSpec', 'K4ContextError')),
        ('spcra_k4_bevfusion.py', ('K4BEVFusionViews',)),
    ):
        path = ROOT / 'pcdet/tta_methods' / filename
        tree = ast.parse(path.read_text())
        selected: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def configured_policy(config):
    namespace = bevfusion_namespace()
    processor = SimpleNamespace(func=SimpleNamespace(__name__='transform_points_to_voxels'))
    dataset = SimpleNamespace(
        point_cloud_range=[-4., -4., -1., 4., 4., 1.],
        data_processor=SimpleNamespace(data_processor_queue=[processor]),
    )
    return namespace['K4BEVFusionViews'](dataset, config).policy


class TestK4Protocol(unittest.TestCase):
    def test_exception_identity_and_message_when_propagated_through_context(self):
        """Given each K4 error; when unwinding inference state; then preserve identity/message."""
        errors = [getattr(load_contract_module(module), name)('original detail') for module, name in (
            ('spcra_k4_core', 'K4InputError'),
            ('spcra_k4_config', 'K4ConfigurationError'),
            ('spcra_k4_augmentation', 'AugmentationConfigurationError'),
            ('spcra_k4_seeding', 'SeedingInputError'),
            ('spcra_k4_training', 'K4TrainingError'),
        )]
        errors.extend((load_contract_module('spcra_k4_seeding').ViewCollisionError('token', 2, 100),
                       bevfusion_namespace()['K4ContextError']('original detail')))
        preserve = load_contract_module('spcra_k4_runtime').preserve_inference_state
        for error in errors:
            with self.subTest(error=type(error).__name__):
                message = str(error)
                model = FakeModel()
                with self.assertRaises(type(error)) as caught:
                    with preserve(model):
                        raise error
                self.assertIs(caught.exception, error)
                self.assertEqual(str(caught.exception), message)
                self.assertEqual([module.training for module in model.modules()], [True, False])

    def test_support_tau_when_default_or_explicit_config_is_used(self):
        """Given omitted/numeric/string tau; when constructing views; then forward a float."""
        for config, expected in (({}, 3.0), ({'SUPPORT_TAU': .5}, .5), ({'SUPPORT_TAU': '7.5'}, 7.5)):
            with self.subTest(config=config):
                policy = configured_policy(config)
                self.assertEqual(policy.support_tau, expected)
                self.assertIsInstance(policy.support_tau, float)

    def test_support_tau_rejected_when_nonpositive_or_nonfinite(self):
        """Given invalid configured tau; when constructing views; then use K4Policy validation."""
        error_type = load_contract_module('spcra_k4_core').K4InputError
        for tau in (0., -1., float('nan'), float('inf'), -float('inf')):
            with self.subTest(tau=tau):
                with self.assertRaisesRegex(error_type, 'support_tau must be finite and positive'):
                    configured_policy({'SUPPORT_TAU': tau})

    def test_only_support_diagnostics_change_when_configured_tau_changes(self):
        """Given fixed matches; when tau changes; then reliability and all other diagnostics stay fixed."""
        core = load_contract_module('spcra_k4_core')
        reference = core.Predictions([[0., 0., 0., 2., 2., 2., 0.]], [1], [.9])
        view = core.Predictions([[.25, 0., 0., 2., 2., 2., 0.]], [1], [.7])
        baseline = core.compute_k4_reliability(reference, (view,) * 4, configured_policy({}))
        changed = core.compute_k4_reliability(reference, (view,) * 4, configured_policy({'SUPPORT_TAU': .5}))
        np.testing.assert_allclose(baseline.support, np.full(4, -np.expm1(-1. / 3.)))
        np.testing.assert_allclose(changed.support, np.full(4, -np.expm1(-2.)))
        for field in ('reliability', 'view_quality', 'match_indices', 'coverage', 'reference_mask',
                      'view_masks', 'reference_rescue_mask', 'view_rescue_masks'):
            np.testing.assert_array_equal(getattr(changed, field), getattr(baseline, field))

    def test_rejects_missing_or_disabled_full_method_controls(self):
        validate = load_contract_module('spcra_k4_config').validate_k4_config
        paths = (('MODEL', 'TTA_FUSION_ADAPTER', 'ENABLED'),
                 ('MODEL', 'TTA_FUSION_ADAPTER', 'SG_DFA', 'ENABLED'),
                 ('TTA', 'FREEZE', 'ENABLED'), ('TTA', 'FREEZE', 'ADAPTER_ONLY'),
                 ('TTA', 'SPCRA', 'CAMERA_RESCUE_ENABLED'))
        for path in paths:
            for missing in (False, True):
                with self.subTest(path=path, missing=missing):
                    config = formal_config()
                    parent = config
                    for key in path[:-1]:
                        parent = parent[key]
                    if missing:
                        parent.pop(path[-1])
                    else:
                        parent[path[-1]] = False
                    with self.assertRaises(ValueError):
                        validate(config)
                    config['TTA']['SPCRA'].pop('VERSION')
                    self.assertFalse(validate(config))

    def test_rejects_missing_disabled_reordered_or_unsupported_law(self):
        validate = load_contract_module('spcra_k4_config').validate_k4_config
        for drift in ('missing', 'disabled', 'reordered', 'unsupported', 'empty', 'invalid_scale', 'target_missing'):
            with self.subTest(drift=drift):
                config = formal_config()
                augmentation = config['DATA_CONFIG']['TTA_DATA_AUGMENTOR']
                queue = augmentation['AUG_CONFIG_LIST']
                if drift == 'missing':
                    config['DATA_CONFIG'].pop('TTA_DATA_AUGMENTOR')
                if drift == 'disabled':
                    augmentation['DISABLE_AUG_LIST'] = ['random_world_flip']
                if drift == 'reordered':
                    queue.reverse()
                if drift == 'unsupported':
                    queue.append({'NAME': 'imgaug'})
                if drift == 'empty':
                    queue.clear()
                if drift == 'invalid_scale':
                    queue[2]['WORLD_SCALE_RANGE'] = [-1., 1.]
                if drift == 'target_missing':
                    config['DATA_CONFIG_TAR'] = {}
                with self.assertRaises(ValueError):
                    validate(config)

    def test_yaml_declares_exact_active_law_without_ignored_view_controls(self):
        config = formal_config()
        augmentation = config['DATA_CONFIG']['TTA_DATA_AUGMENTOR']
        self.assertEqual(augmentation['DISABLE_AUG_LIST'], [])
        self.assertEqual([item['NAME'] for item in augmentation['AUG_CONFIG_LIST']],
                         ['random_world_flip', 'random_world_rotation',
                          'random_world_scaling', 'random_world_translation'])
        for key in ('VIEW_YAW_LIMIT', 'VIEW_SCALE_LIMIT', 'VIEW_TRANSLATION_LIMIT'):
            self.assertNotIn(key, config['TTA']['SPCRA'])
