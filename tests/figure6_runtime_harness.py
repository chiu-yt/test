"""CPU execution of production MOS/adapter bodies without CUDA package imports."""

import ast
import copy
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import numpy as np
import torch

from pcdet.utils.figure6_runtime import Figure6RuntimeCollector, stage
from pcdet.utils.inference_utils import forward_without_annotations

ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    def __getattr__(self, name):
        return self[name]


def definitions(path, namespace):
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, str(ROOT / path), 'exec'), namespace)
    return namespace


def adapter(enabled=True):
    namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional, 'stage': stage}
    definitions('pcdet/models/backbones_2d/fuser/tta_fusion_adapter_utils.py', namespace)
    definitions('pcdet/models/backbones_2d/fuser/tta_fusion_adapter.py', namespace)
    module = namespace['BEVFusionTTAAdapter']({
        'IN_CHANNEL': 2, 'HIDDEN_CHANNEL': 2, 'IMAGE_CHANNEL': 2, 'LIDAR_CHANNEL': 2,
        'GROUP_NORM_GROUPS': 1, 'SG_DFA': {'ENABLED': enabled},
    })
    with torch.no_grad():
        module.shared_refine[-1].bias.fill_(.4)
    return module


class FakeDetector(torch.nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.adapter = adapter()
        self.calls = 0
        self.offset = offset
        self.predictions = [
            {'pred_boxes': torch.ones(2, 7), 'pred_scores': torch.tensor([.2, .8]),
             'pred_labels': torch.tensor([1, 2])} for _ in range(2)
        ]

    def forward(self, batch):
        self.calls += 1
        batch['spatial_features'] = torch.ones(2, 2, 2, 2)
        self.adapter(batch)
        for prediction in self.predictions:
            prediction['pred_boxes'].fill_(self.offset + self.calls)
        if self.training:
            return {'loss': batch['spatial_features'].sum(), 'tb_dict': {}}
        return self.predictions, {}


def mos_namespace():
    config = Config(CLASS_NAMES=['a', 'b'], SELF_TRAIN=Config(
        SCORE_THRESH=[.5, .5], TAR=Config(LOSS_WEIGHT=1.),
        RG_PLM=Config(ENABLED=True, MODE='reweight', SCORE_BLEND=.5),
    ))
    namespace = vars(ModuleType('figure6_mos_harness'))
    namespace.update(torch=torch, np=np, cfg=config, copy=copy, OrderedDict=OrderedDict,
                     Optional=Optional, Figure6RuntimeCollector=Figure6RuntimeCollector,
                     nullcontext=nullcontext, forward_without_annotations=forward_without_annotations,
                     commu_utils=SimpleNamespace(get_rank=lambda: 1),
                     TTA_augmentation=lambda dataset, batch: batch)
    memory = definitions('pcdet/utils/memory_ensemble_utils.py', dict(torch=torch, np=np, cfg=config))
    namespace['memory_ensemble_utils'] = SimpleNamespace(**memory)
    definitions('pcdet/tta_methods/mos.py', namespace)
    namespace.update(NEW_PSEUDO_LABELS={}, PSEUDO_LABELS={}, RG_PLM_STATS={},
                     HARD_PSEUDO_STATS={}, GEOMETRY_FILTER_STATS={})
    return namespace


def engine(observer=None):
    namespace = mos_namespace()
    model = FakeDetector()
    aggregate = FakeDetector(10)
    tta = Config(METHOD='mos', MOS_SETTING=Config(AGGREGATE_START_CKPT=0))
    instance = namespace['MOS'](model, tta, SimpleNamespace(), figure6_collector=observer)
    instance.run_ckpt_dir = 'fake'
    instance._prepare_sg_dfa_density_map = lambda batch: batch.update(tta_density_map=torch.ones(2, 1, 1, 1))
    instance._collect_hist_from_gt_boxes = lambda boxes: {}
    instance._spcra_enabled = lambda: True
    instance._should_memory_update = lambda: False
    instance._collect_aggregation_ckpts = lambda path: ['a', 'b', 'c']
    instance._perform_aggregation = lambda paths, batch, pred: aggregate
    instance._dpo_matcher_enabled = lambda: False
    instance._build_tta_proposal_boxes = lambda batch: None
    for name in ('_add_dpo_matcher_stats_to_logs', '_add_spcra_stats_to_logs',
                 '_add_rg_plm_stats_to_logs', '_add_sg_dfa_stats_to_logs'):
        setattr(instance, name, lambda *args: None)

    def diagnostic(batch, predictions, diagnostic_model=None):
        forward_without_annotations(model if diagnostic_model is None else diagnostic_model, batch)
        return [dict(prediction, spcra_reliability=np.array([0., .2]))
                for prediction in predictions], {}

    instance._run_spcra_diagnostic = diagnostic
    return instance, namespace, aggregate


def batch():
    return {'frame_id': ['other', 'selected'], 'batch_size': 2,
            'points': torch.tensor([[0., 0., 0., 0.], [1., 1., 2., 3.]]),
            'gt_boxes': torch.full((2, 1, 10), 99.),
            'lidar_aug_matrix': torch.eye(4).repeat(2, 1, 1)}
