import numpy as np
import unittest
import ast
from types import SimpleNamespace

from test_spcra_k4_runtime import ROOT, load_contract_module
from test_spcra_k4_runtime import FakeModel


def check_student_rows_when_scaling_changes_class_preserve_velocity_and_class():
    module = load_contract_module('spcra_k4_mos')
    classes = np.array([[8., 10., 0.]])
    augmented = np.array([[[1., 2., 3., 2., 3., 4., .2, -2., 5., 8.8],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [4., 4., 4., 0., 0., 0., .2, 0., 0., 0.]]])
    result = module.restore_student_rows(augmented, classes)
    np.testing.assert_array_equal(result[0, 0, 7:], [-2., 5., 8.])
    np.testing.assert_array_equal(result[0, 1:], np.zeros((2, 10)))


def check_student_rows_when_augmentation_changes_row_identity_reject():
    module = load_contract_module('spcra_k4_mos')
    with unittest.TestCase().assertRaises(ValueError):
        module.restore_student_rows(np.zeros((1, 1, 10)), np.zeros((1, 2)))


def check_adapter_when_formal_enabled_is_compatible():
    from test_spcra_k4_integration import formal_config

    config = formal_config()
    config['MODEL']['TTA_FUSION_ADAPTER']['ENABLED'] = True
    assert load_contract_module('spcra_k4_config').validate_k4_config(config)


class TestK4MOS(unittest.TestCase):
    def test_four_views_share_one_context_when_capture_is_on_or_off(self):
        module = load_contract_module('spcra_k4_mos')
        core = load_contract_module('spcra_k4_core')
        seed = load_contract_module('spcra_k4_seeding')

        class TensorArray(np.ndarray):
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.asarray(self)

        prediction = {'pred_boxes': np.array([[0., 0., 0., 2., 2., 2., 0., 3., -4.]]),
                      'pred_labels': np.array([8]), 'pred_scores': np.array([.9])}
        context = ('immutable-reference-context',)
        samples = seed.ViewSamples(tuple(np.arange(2) for _ in range(4)),
                                   tuple(np.eye(4) for _ in range(4)),
                                   ('a', 'b', 'c', 'd'), (0, 0, 0, 0), 'test-law')
        outcomes = []
        for enabled in (False, True):
            contexts, evidence, points, builds = [], [], [], []

            def build_context(predictions):
                builds.append(predictions)
                return context

            def build_view(pristine, indices, transforms, proposal_context):
                contexts.append(proposal_context)
                pristine['points'][:, 1] += len(contexts)
                return pristine

            capture = SimpleNamespace(reference=lambda *values: None,
                                      view=lambda value: points.append(value.copy()),
                                      evidence=evidence.append) if enabled else None
            views = SimpleNamespace(predictions=lambda predictions, batch: predictions,
                                    build_context=build_context, build_view=build_view,
                                    sample=lambda batch: (samples,), policy=core.K4Policy(),
                                    forward_view=lambda model, batch: [prediction])
            reference = {'batch_size': 1, 'points': np.zeros((2, 4)).view(TensorArray),
                         'lidar_aug_matrix': np.eye(4)[None].view(TensorArray)}
            batch = dict(reference, _spcra_k4_reference=reference, spatial_features_img=np.ones(10))
            owner = SimpleNamespace(k4_views=views, figure7_capture=capture, model=FakeModel())
            outcomes.append(module.run_current_k4(owner, batch, [prediction]))
            self.assertEqual(len(builds), 1)
            self.assertEqual(contexts, [context] * 4)
            self.assertNotIn('spatial_features_img', batch)
            self.assertEqual(len(points), 4 if enabled else 0)
            self.assertEqual(len(evidence), 1 if enabled else 0)
        for key in ('pred_boxes', 'pred_labels', 'pred_scores', 'spcra_reliability', 'spcra_k4_accepted'):
            np.testing.assert_array_equal(outcomes[0][0][key], outcomes[1][0][key])

    def test_formal_adapter_freeze_keeps_head_parameters_trainable(self):
        tree = ast.parse((ROOT / 'tools/train.py').read_text())
        names = {'apply_tta_freeze_strategy', '_unfreeze_module', '_freeze_module_params_keep_train'}
        namespace = {}
        exec(compile(ast.Module(body=[node for node in tree.body
                                     if isinstance(node, ast.FunctionDef) and node.name in names],
                                type_ignores=[]), '<freeze>', 'exec'), namespace)

        class Module:
            def __init__(self):
                self.parameter = SimpleNamespace(requires_grad=True, numel=lambda: 1)

            def parameters(self):
                return [self.parameter]

            def train(self):
                return self

        adapter, head = Module(), Module()
        model = SimpleNamespace(tta_fusion_adapter=adapter,
                                named_children=lambda: [('tta_fusion_adapter', adapter), ('dense_head', head)],
                                parameters=lambda: [adapter.parameter, head.parameter])
        for version, expected in (('k4_v1', True), ('legacy', False)):
            config = {'TTA': {'FREEZE': {'ENABLED': True, 'ADAPTER_ONLY': True},
                              'SPCRA': {'ENABLED': True, 'VERSION': version}}}
            namespace['apply_tta_freeze_strategy'](model, config, SimpleNamespace(info=lambda value: None))
            self.assertEqual(head.parameter.requires_grad, expected)

    def test_student_and_config_contracts(self):
        for case in (check_student_rows_when_scaling_changes_class_preserve_velocity_and_class,
                     check_student_rows_when_augmentation_changes_row_identity_reject,
                     check_adapter_when_formal_enabled_is_compatible):
            with self.subTest(case=case.__name__):
                case()
