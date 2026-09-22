"""Execute the real outer loop with CPU collaborators and the real artifact writer."""

import json
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import tqdm
import yaml

from figure6_runtime_harness import Config, definitions
from pcdet.utils.figure6_artifacts import scan_records
from pcdet.utils.figure6_schema import ArtifactError, FIXED_TOKENS


@pytest.fixture
def stream(tmp_path, monkeypatch):
    from pcdet.utils.figure6_stream import Figure6StreamCapture

    events = []
    workers = []

    class Worker:
        def __init__(self, model, config, logger, dataset=None, figure6_collector=None):
            self.collector = figure6_collector
            workers.append(self)

        def optimize(self, batch):
            events.append(('optimize', batch['samples_seen'], tuple(batch['frame_id'])))
            if self.collector is not None:
                self.collector.inputs(batch)
            return 0., {}, {}

    class Loader(list):
        dataset = None
        batch_size = 2

    class Profiler:
        def __init__(self, *args, **kwargs):
            pass

        def begin_batch(self, size):
            pass

        def begin_segment(self):
            pass

        def end_segment(self):
            pass

        def end_batch(self):
            pass

        def finalize(self):
            pass

    namespace = vars(ModuleType('figure6_train_harness'))
    namespace.update(torch=torch, tqdm=tqdm, MOS=Worker, CodeMergeTTA=Worker,
                     Figure6StreamCapture=Figure6StreamCapture,
                     EfficiencyProfiler=Profiler, EfficiencyProfileRun=lambda **kw: kw,
                     clip_grad_norm_=lambda *args: events.append(('clip',)))
    definitions('tools/train_utils/train_st_utils.py', namespace)
    namespace.update(
        checkpoint_state=lambda *args: {},
        save_checkpoint=lambda state, filename: events.append(('checkpoint', str(filename))),
    )
    optimizer = SimpleNamespace(
        param_groups=[{'lr': .1}], zero_grad=lambda: events.append(('zero',)),
        step=lambda: events.append(('step',)),
    )
    config = Config(EXP_GROUP_PATH='nuscenes_models', TAG='bevfusion_mos',
                    DATA_CONFIG=Config(MAX_SWEEPS=5,
                    POINT_CLOUD_RANGE=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                    CORRUPTION=Config(
                        ENABLED=True, APPLY_IN=['train', 'test'],
                        LIDAR_SPARSITY=Config(ENABLED=True, MODE='random_keep', SEVERITY=5))))
    tta = Config(ENABLED=True, METHOD='mos', SAVE_CKPT=[0, 2, 4], SAVE_CKPT_INTERVAL=0,
                 FIGURE6_CAPTURE=Config(ENABLED=True, OUTPUT_DIR='', TOKENS=list(FIXED_TOKENS)))
    config['TTA'] = tta
    kwargs = dict(model=torch.nn.Linear(1, 1), optimizer=optimizer,
                  model_func=None, lr_scheduler=SimpleNamespace(
                      step=lambda iteration: events.append(('scheduler', iteration))),
                  optim_cfg=Config(GRAD_NORM_CLIP=1), start_epoch=0, total_epochs=1,
                  start_iter=7, rank=0, tb_log=None, ckpt_save_dir=tmp_path / 'ckpt',
                  ckpt_save_interval=99, logger=logging.getLogger('stream-test'),
                  tta_cfg=tta, cfg=config, capture_provenance={
                      'config': 'cfgs/exact.yaml', 'command': "python train.py --ckpt 'source file.pth'",
                      'source_checkpoint': 'source file.pth', 'fixed_seed': 'True'})
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)

    def run(batches):
        return namespace['train_model_st'](train_loader=Loader(batches), **kwargs)

    return SimpleNamespace(run=run, kwargs=kwargs, config=config, tta=tta, namespace=namespace,
                           events=events, workers=workers, output=tmp_path / 'figure6_capture')


def batch(tokens):
    return {'batch_size': len(tokens), 'metadata': [{'token': token} for token in tokens],
            'frame_id': ['lidar_%d' % index for index in range(len(tokens))],
            'points': torch.tensor([[float(index), 1., 2., 3.] for index in range(len(tokens))])}


def test_disabled_when_targets_present(stream):
    # Given capture disabled despite a matching token.
    stream.tta.FIGURE6_CAPTURE['ENABLED'] = False
    # When the normal stream runs.
    assert stream.run([batch(FIXED_TOKENS[:2])]) == 8
    # Then no collector or capture directory exists.
    assert stream.workers[0].collector is None
    assert not stream.output.exists()


@pytest.mark.parametrize('method', ['mos', 'codemerge'])
def test_all_occurrences_when_targets_on_nonzero_global_rank(stream, monkeypatch, method):
    # Given local rank zero on global rank three and repeated mixed batches.
    stream.tta['METHOD'] = method
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 3)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 8)
    batches = [batch(['ordinary', FIXED_TOKENS[0]]), batch(FIXED_TOKENS[1:]),
               batch(['ordinary', FIXED_TOKENS[0]]), batch(['tail', 'tail2'])]
    # When every batch including the tail completes.
    assert stream.run(batches) == 11
    catalog = scan_records(stream.output)
    # Then tokens, rank identity, pre-update counters and repeated occurrences survive.
    assert not catalog.issues
    identities = sorted((entry.record.identity for entry in catalog.records),
                        key=lambda identity: identity.selection_key)
    assert [identity.token for identity in identities] == [FIXED_TOKENS[0], *FIXED_TOKENS[1:], FIXED_TOKENS[0]]
    assert [identity.samples_seen for identity in identities] == [0, 2, 2, 4]
    assert [identity.accumulated_iter_before for identity in identities] == [7, 8, 8, 9]
    assert all(identity.global_rank == 3 and identity.world_size == 8 for identity in identities)
    assert all(entry.path.parent.parent.name == 'rank_00003' for entry in catalog.records)
    assert identities[0].frame_id == 'lidar_1' and identities[0].batch_index == 1
    assert len([event for event in stream.events if event[0] == 'optimize']) == 4


def test_frame_id_is_not_token_when_metadata_is_non_target(stream):
    # Given a filename that happens to equal a requested token.
    ordinary = batch(['ordinary'])
    ordinary['frame_id'] = [FIXED_TOKENS[0]]
    # When metadata identifies a different sample.
    stream.run([ordinary])
    # Then capture stays idle.
    assert not stream.output.exists()


@pytest.mark.parametrize('mode', ['random_keep', 'density_dec_global'])
def test_protocol_when_resolved_mode_changes(stream, mode):
    # Given resolved corruption settings, not a tag-derived mode.
    stream.config.DATA_CONFIG.CORRUPTION.LIDAR_SPARSITY['MODE'] = mode
    # When a record is persisted.
    stream.run([batch(FIXED_TOKENS[:1])])
    protocol = scan_records(stream.output).records[0].record.protocol
    # Then provenance records the actual protocol and authoritative CLI values.
    assert protocol['lidar_sparsity.mode'] == mode
    assert protocol['max_sweeps'] == '5'
    assert protocol['config'] == 'cfgs/exact.yaml'
    assert protocol['source_checkpoint'] == 'source file.pth'
    assert protocol['fixed_seed'] == 'True'
    assert json.loads(protocol['point_cloud_range']) == [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    assert all(isinstance(value, str) and value for value in protocol.values())


@pytest.mark.parametrize('malformed', ['frames', 'metadata_length', 'metadata_entry', 'metadata_missing'])
def test_selected_malformed_batch_fails_before_execution(stream, malformed):
    # Given a target batch with corrupt collated identity data.
    selected = batch([FIXED_TOKENS[0], 'ordinary'])
    if malformed == 'frames':
        selected['frame_id'] = []
    if malformed == 'metadata_length':
        selected['metadata'].pop()
    if malformed == 'metadata_entry':
        selected['metadata'][1] = None
    if malformed == 'metadata_missing':
        selected.pop('metadata')
    # When capture parses the selected attempt.
    with pytest.raises(ArtifactError, match='Figure 6'):
        stream.run([selected])
    # Then no model execution or record happened.
    assert not any(event[0] == 'optimize' for event in stream.events)
    assert not stream.output.exists()


def test_order_when_capture_writes(stream, monkeypatch):
    # Given an instrumented real writer, preserving its actual persistence.
    from pcdet.utils import figure6_stream
    writer = figure6_stream.write_record

    def observed_write(output, record):
        stream.events.append(('write',))
        return writer(output, record)

    monkeypatch.setattr(figure6_stream, 'write_record', observed_write)
    # When a target is followed by a non-target.
    stream.run([batch(FIXED_TOKENS[:2]), batch(['tail', 'tail2'])])
    # Then capture inserts only writes between optimize and the existing optimizer step.
    assert [event[0] for event in stream.events] == [
        'scheduler', 'zero', 'optimize', 'write', 'write', 'clip', 'step', 'checkpoint',
        'scheduler', 'zero', 'optimize', 'clip', 'step', 'checkpoint']


def test_yaml_when_overrides_require_existing_keys():
    # Given the shipped model config.
    path = Path(__file__).resolve().parents[1] / 'tools/cfgs/nuscenes_models/bevfusion_mos.yaml'
    # When parsing its capture defaults.
    capture = yaml.safe_load(path.read_text())['TTA']['FIGURE6_CAPTURE']
    # Then opt-in and token overrides are supported without adding keys.
    assert capture == {'ENABLED': False, 'OUTPUT_DIR': '', 'TOKENS': list(FIXED_TOKENS)}
