"""Stream lifecycle boundaries, constructor forwarding and CLI provenance."""

import shlex
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from figure6_runtime_harness import definitions, engine, batch as mos_batch
from test_figure6_train_stream import stream, batch
from pcdet.utils.figure6_artifacts import scan_records
from pcdet.utils.figure6_provenance import build_capture_provenance
from pcdet.utils.figure6_schema import ArtifactError, FIXED_TOKENS


def test_disabled_when_collector_construction_is_forbidden(stream, monkeypatch):
    # Given disabled capture and a constructor trap.
    from pcdet.utils import figure6_stream
    stream.tta.FIGURE6_CAPTURE['ENABLED'] = False
    monkeypatch.setattr(figure6_stream, 'Figure6RuntimeCollector',
                        lambda: pytest.fail('disabled capture constructed a collector'))
    # When a complete target batch runs.
    stream.run([batch(FIXED_TOKENS)])
    # Then no capture storage exists.
    assert not stream.output.exists()


def test_no_snapshots_when_only_non_targets(stream, monkeypatch):
    # Given non-target batches and a tensor snapshot trap.
    from pcdet.utils import figure6_runtime
    monkeypatch.setattr(figure6_runtime, 'snapshot', lambda value: pytest.fail('unexpected snapshot'))
    # When ordinary online updates run.
    assert stream.run([batch(['ordinary']), batch(['tail'])]) == 9
    # Then capture remains storage-free.
    assert not stream.output.exists()


def test_occurrences_when_epochs_repeat_and_output_is_explicit(stream, tmp_path):
    # Given a configured destination and repeated epochs.
    output = tmp_path / 'explicit'
    stream.tta.FIGURE6_CAPTURE['OUTPUT_DIR'] = str(output)
    stream.kwargs['total_epochs'] = 2
    # When the same token appears once in each epoch.
    stream.run([batch(FIXED_TOKENS[:1])])
    identities = sorted((entry.record.identity for entry in scan_records(output).records),
                        key=lambda identity: identity.selection_key)
    # Then the epochs have independent identities without changing local samples_seen.
    assert [(identity.epoch, identity.accumulated_iter_before, identity.samples_seen)
            for identity in identities] == [(0, 7, 0), (1, 8, 0)]
    assert not stream.output.exists()


def test_exact_collision_when_run_identity_repeats(stream):
    # Given a committed identity from an earlier run.
    stream.run([batch(FIXED_TOKENS[:1])])
    # When the exact run/epoch/iteration/rank/sample identity repeats.
    with pytest.raises(FileExistsError):
        stream.run([batch(FIXED_TOKENS[:1])])
    # Then the original occurrence remains intact.
    assert len(scan_records(stream.output).records) == 1


def test_optimize_failure_when_transaction_is_active(stream, monkeypatch):
    # Given a model execution failure rather than a normal skip return.
    def fail(self, batch):
        raise ArtifactError('model failure')

    monkeypatch.setattr(stream.namespace['MOS'], 'optimize', fail)
    # When optimize raises.
    with pytest.raises(ArtifactError, match='model failure'):
        stream.run([batch(FIXED_TOKENS[:1])])
    # Then no incomplete result is committed or optimizer stepped.
    assert not stream.output.exists()
    assert not any(event[0] == 'step' for event in stream.events)


def test_cli_provenance_when_paths_need_shell_escaping(monkeypatch):
    # Given authoritative parsed CLI arguments including a path with whitespace.
    argv = ['train.py', '--ckpt', '/checkpoints/source file.pth', '--set', 'TTA.FIGURE6_CAPTURE.ENABLED', 'True']
    args = SimpleNamespace(cfg_file='cfgs/custom.yaml', ckpt=argv[2], fix_random_seed=False)
    monkeypatch.setattr(sys, 'argv', argv)
    # When building provenance at the CLI boundary.
    values = build_capture_provenance(args.cfg_file, args.ckpt, args.fix_random_seed)
    # Then argument boundaries and source identity are preserved exactly.
    assert shlex.split(values['command']) == [sys.executable] + argv
    assert values['source_checkpoint'] == argv[2]
    assert values['config'] == args.cfg_file
    assert values['fixed_seed'] == 'False'


def test_codemerge_constructor_when_collector_supplied():
    # Given the production CodeMerge constructor and a minimal MOS parent.
    class Parent:
        def __init__(self, model, config, logger, dataset=None, figure6_collector=None):
            self.collector = figure6_collector
            self.tta_cfg = config
            self.max_ckpt_cache = 1
            self.rank = 1

    namespace = vars(ModuleType('figure6_codemerge_harness'))
    namespace.update(MOS=Parent, snapshot_floating_state=lambda model: {})
    definitions('pcdet/tta_methods/codemerge.py', namespace)
    sentinel = torch.nn.Identity()
    # When the subclass initializes.
    worker = namespace['CodeMergeTTA'](None, {}, None, figure6_collector=sentinel)
    # Then it forwards the optional collector unchanged.
    assert worker.collector is sentinel


@pytest.mark.parametrize('nonfinite', [False, True])
def test_real_mos_when_capture_runs_through_outer_loop(stream, nonfinite):
    # Given the real MOS optimize body, with only CUDA/detector dependencies replaced.
    instance, namespace, _ = engine()
    namespace['load_data_to_gpu'] = lambda batch: None
    if nonfinite:
        with torch.no_grad():
            instance.model.adapter.shared_refine[-1].bias.fill_(float('nan'))
    stream.kwargs['model'] = instance.model
    stream.kwargs['optimizer'] = None
    results = []
    optimize = instance.optimize

    def observed_optimize(batch):
        result = optimize(batch)
        results.append(result)
        return result

    instance.optimize = observed_optimize

    def worker(model, config, logger, dataset=None, figure6_collector=None):
        instance.figure6_collector = figure6_collector
        return instance

    stream.namespace['MOS'] = worker
    inputs = mos_batch()
    inputs['metadata'] = [{'token': 'ordinary'}, {'token': FIXED_TOKENS[0]}]
    # When the formal outer loop drives optimize, including the existing skip return.
    assert stream.run([inputs]) == 8
    # Then immutable pre-update detections are persisted in either case.
    record, = [entry.record for entry in scan_records(stream.output).records]
    assert record.identity.frame_id == 'selected'
    assert record.stages['final_detection'].arrays['pred_boxes'][0, 0] == 1
    assert bool(results[0][1].get('loss_non_finite_skip', False)) == nonfinite
