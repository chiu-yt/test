"""Exercise MOS up to the actual reference-forward seam without importing CUDA."""

import ast
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import numpy as np

from test_spcra_k4_runtime import ROOT, load_contract_module


class TensorArray(np.ndarray):
    def new_zeros(self, shape):
        return np.zeros(shape, dtype=self.dtype).view(TensorArray)

    def bool(self):
        return self.astype(bool)


class ReferenceReached(RuntimeError):
    pass


class TestK4Reference(unittest.TestCase):
    def test_reference_forward_receives_fresh_empty_context_with_or_without_stale_inputs(self):
        module = load_contract_module('spcra_k4_mos')
        tree = ast.parse((ROOT / 'pcdet/tta_methods/mos.py').read_text())
        optimize = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == 'optimize')
        for stale in (False, True):
            with self.subTest(stale=stale):
                batch = {'batch_size': 2, 'points': np.zeros((3, 4), dtype=np.float32).view(TensorArray)}
                if stale:
                    batch.update(tta_density_map=np.full((2, 1, 2, 2), 99.),
                                 tta_proposal_boxes=np.ones((2, 3, 9)),
                                 tta_proposal_mask=np.ones((2, 3), dtype=bool))
                density = np.zeros((2, 1, 2, 2))

                def reference_forward(model, current):
                    self.assertIs(current['tta_density_map'], density)
                    self.assertEqual(current['tta_proposal_boxes'].shape, (2, 0, 9))
                    self.assertEqual(current['tta_proposal_mask'].shape, (2, 0))
                    self.assertEqual(current['tta_proposal_mask'].dtype, np.dtype(bool))
                    raise ReferenceReached()

                owner = SimpleNamespace(
                    total_samples_seen=0, figure6_collector=None,
                    _spcra_k4_enabled=lambda: True,
                    _prepare_sg_dfa_density_map=lambda current: current.update(tta_density_map=density),
                    model=SimpleNamespace(eval=lambda: None),
                )
                namespace = {}
                namespace.update(load_data_to_gpu=lambda current: None,
                                 torch=SimpleNamespace(no_grad=nullcontext), nullcontext=nullcontext,
                                 snapshot_reference=lambda current: {},
                                 forward_without_annotations=reference_forward)
                if hasattr(module, 'prepare_k4_reference'):
                    namespace['prepare_k4_reference'] = module.prepare_k4_reference
                exec(compile(ast.Module(body=[optimize], type_ignores=[]), '<mos-reference>', 'exec'), namespace)
                with self.assertRaises(ReferenceReached):
                    namespace['optimize'](owner, batch)
