import importlib

import numpy as np
import pytest
import torch

from pcdet.utils.figure6_schema import Occurrence, StageState


def collector():
    runtime = importlib.import_module('pcdet.utils.figure6_runtime')
    observer = runtime.Figure6RuntimeCollector()
    observer.begin([Occurrence('token', 'selected', 0, 1, 2, 0, 1, 1, 2)], {'method': 'mos'})
    return observer


def test_snapshots_when_sources_mutate():
    """Given selected tensors, when captured then mutations cannot alter records."""
    observer = collector()
    points = torch.tensor([[0., 9., 8.], [1., 2., float('nan')]])
    batch = {'points': points, 'tta_density_map': torch.ones(2, 1, 2, 2),
             'lidar_aug_matrix': torch.eye(4).repeat(2, 1, 1)}
    observer.inputs(batch)
    pred = {'pred_boxes': torch.ones(2, 7), 'pred_scores': torch.tensor([.2, .8]),
            'pred_labels': torch.tensor([1, 2]), 'spcra_reliability': np.array([0., .2])}
    observer.final_detection([{}, pred])
    observer.spcra([{}, pred], 'current_pre_update')
    points.fill_(8)
    pred['pred_boxes'].fill_(9)
    pred['spcra_reliability'].fill(9)
    record, = observer.finalize()
    assert np.isnan(record.stages['input'].arrays['points'][0, 2])
    assert record.stages['final_detection'].arrays['pred_boxes'][0, 0] == 1
    np.testing.assert_array_equal(record.stages['spcra.current_pre_update'].arrays['spcra_reliability'], [0., .2])


def test_pass_ownership_when_aggregated_pseudo_replaces_current():
    """Given two pseudo sources, when observed then both passes remain separate."""
    observer = collector()
    infos = {'gt_boxes': np.array([[0., -2., .8]]), 'pseudo_cls_weights': np.array([.3])}
    for owner in ('current_pre_update', 'aggregated_pseudo_source'):
        observer.pseudo('selected', infos, (owner, 'rg_plm_before'))
        infos['gt_boxes'][0, -1] /= 2
        observer.pseudo('selected', infos, (owner, 'rg_plm_after'))
        observer.effective({'selected': infos}, owner)
    record, = observer.finalize()
    assert record.protocol['final_pseudo_source'] == 'aggregated_pseudo_source'
    assert record.stages['rg_plm_before.current_pre_update'].arrays['gt_boxes'][0, -1] == .8
    assert record.stages['rg_plm_after.current_pre_update'].arrays['gt_boxes'][0, -1] == .4
    assert record.stages['effective_pseudo.aggregated_pseudo_source'].arrays['gt_boxes'][0, 1] == -2


@pytest.mark.parametrize('reason', ['missing_frame_ids', 'empty_pseudo_labels'])
def test_skip_when_original_gt_exists(reason):
    """Given an injection skip, when captured then no original GT is exported."""
    observer = collector()
    observer.injection(None, reason)
    record, = observer.finalize()
    stage = record.stages['injection']
    assert stage.status.state is StageState.MISSING
    assert stage.status.detail == reason
    assert not stage.arrays


def test_inactive_when_no_transaction():
    """Given no selection, when first forward is entered then no request is added."""
    runtime = importlib.import_module('pcdet.utils.figure6_runtime')
    observer = runtime.Figure6RuntimeCollector()
    batch = {}
    with observer.first_forward(batch):
        assert batch == {}
    assert observer.finalize() == ()
