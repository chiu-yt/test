"""Execute the real loader/optimizer loop with CPU boundary collaborators."""

import ast
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from figure7_fixtures import observation, provenance
from pcdet.utils.figure7_runtime import Figure7Collector
from pcdet.utils.figure7_stream import Figure7StreamCapture
from test_spcra_k4_runtime import ROOT


class Config(dict):
    def __getattr__(self, name):
        return self[name]


class StepFailure(RuntimeError):
    pass


class TestK4TrainStream(unittest.TestCase):
    def test_updates_when_capture_enabled_match_disabled_stream(self):
        self.run_stream(None)
        self.run_stream(None, False)

    def test_optimizer_failure_records_failed_pre_update_evidence(self):
        self.run_stream('step')

    def test_begun_batch_when_any_operation_fails_is_recorded(self):
        for operation in ('scheduler', 'gpu', 'begin_batch', 'begin_segment', 'zero',
                          'optimize', 'clip', 'end_segment', 'checkpoint_state', 'checkpoint', 'end_batch',
                          'logging', 'progress', 'tensorboard'):
            with self.subTest(operation=operation):
                self.run_stream(operation)

    def run_stream(self, failure_at, profile_enabled=True):
        tree = ast.parse((ROOT / 'tools/train_utils/train_st_utils.py').read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'train_model_st')
        outcomes = []
        with TemporaryDirectory() as directory:
            for enabled in (False, True):
                events, losses = [], []
                failure = StepFailure(str(failure_at))

                def event(name, *values):
                    events.append((name, *values))
                    if name == failure_at:
                        raise failure

                capture = Figure7StreamCapture(Figure7Collector(Path(directory) / str(enabled), provenance()))
                original_observe = capture.collector.observe

                def observe(frame):
                    events.append(('capture', frame.model_step))
                    return original_observe(frame)

                capture.collector.observe = observe

                class Worker:
                    def __init__(self, model, config, logger, dataset=None):
                        self.figure7_capture = None

                    def optimize(self, batch, data_already_on_gpu=False):
                        token = batch['metadata'][0]['token']
                        events.append(('optimize', token))
                        losses.append(.25)
                        if self.figure7_capture is not None:
                            frame = observation(token, stable=token == 'stable')
                            self.figure7_capture.reference(batch['points'], np.eye(4)[None])
                            for _ in range(4):
                                self.figure7_capture.view(batch['points'])
                            self.figure7_capture.evidence(frame.evidence)
                        event('optimize_end')
                        if failure_at == 'optimize':
                            raise failure
                        return .25, {}, {}

                class Loader(list):
                    dataset = None
                    batch_size = 1

                profiler = SimpleNamespace(**{name: lambda *args, name=name: event(name) for name in
                                             ('begin_batch', 'begin_segment', 'end_segment',
                                              'end_batch', 'finalize')})
                class Progress:
                    def __iter__(self):
                        return iter(range(1))

                    def set_postfix(self, values):
                        event('progress')

                    def update(self):
                        event('update')

                    def close(self):
                        pass

                progress = SimpleNamespace(trange=lambda *args, **kwargs: nullcontext(Progress()),
                                           tqdm=lambda **kwargs: Progress())
                namespace = {'tqdm': progress, 'MOS': Worker, 'CodeMergeTTA': Worker,
                             'Figure7StreamCapture': SimpleNamespace(from_config=lambda *args: capture),
                             'EfficiencyProfiler': lambda *args, **kwargs: profiler,
                             'EfficiencyProfileRun': lambda **kwargs: kwargs,
                             'clip_grad_norm_': lambda *args: event('clip'),
                             'load_data_to_gpu': lambda batch: event('gpu'),
                             'torch': SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False)),
                             'checkpoint_state': lambda *args: event('checkpoint_state'),
                             'save_checkpoint': lambda *args, **kwargs: event('checkpoint')}
                exec(compile(ast.Module(body=[function], type_ignores=[]), '<real-stream>', 'exec'), namespace)
                def step():
                    event('step')

                optimizer = SimpleNamespace(param_groups=[{'lr': .1}],
                                            zero_grad=lambda: event('zero'),
                                            step=step)
                tta = Config(ENABLED=True, METHOD='mos', FIGURE7_CAPTURE={'ENABLED': enabled},
                             SAVE_CKPT=[0], SAVE_CKPT_INTERVAL=0,
                             EFFICIENCY_PROFILE={'ENABLED': profile_enabled})
                config = Config(EXP_GROUP_PATH='nuscenes_models', TAG='bevfusion_spcra_k4', TTA=tta)
                batches = Loader([{'batch_size': 1, 'metadata': [{'token': token}], 'frame_id': [token],
                                   'points': np.array([[0., 10., 0., 0., 1.]])}
                                  for token in ('stable', 'variable')])
                arguments = dict(
                    model=SimpleNamespace(parameters=lambda: (), train=lambda: None),
                    optimizer=optimizer, train_loader=batches, model_func=None,
                    lr_scheduler=SimpleNamespace(step=lambda iteration: event('scheduler', iteration)),
                    optim_cfg=Config(GRAD_NORM_CLIP=1), start_epoch=0, total_epochs=1,
                    start_iter=7, rank=0, tb_log=SimpleNamespace(add_scalar=lambda *args: event('tensorboard')),
                    ckpt_save_dir=Path(directory) / 'ckpt', ckpt_save_interval=2,
                    logger=SimpleNamespace(info=lambda text: event('logging') if 'checkpoint_iter_' in text else None),
                    tta_cfg=tta, cfg=config,
                )
                with self.assertRaises(StepFailure) if failure_at else nullcontext() as raised:
                    result = namespace['train_model_st'](**arguments)
                    self.assertEqual(result, 9)
                if failure_at:
                    assert raised is not None
                    self.assertIs(raised.exception, failure)
                if enabled:
                    expected = [('capture', 7)] if failure_at else [('capture', 7), ('capture', 8)]
                    self.assertEqual([event for event in events if event[0] == 'capture'],
                                     expected)
                    self.assertEqual(capture.identities, ())
                    if failure_at:
                        ledger = json.loads((capture.collector.output / 'ledger.jsonl').read_text())
                        self.assertEqual(ledger['status'], 'failed')
                        self.assertEqual(ledger['model_step'], 7)
                        self.assertIn(str(failure), ledger['detail'])
                        self.assertEqual(all(ledger['completion']), failure_at not in
                                         ('scheduler', 'gpu', 'begin_batch', 'begin_segment', 'zero'))
                    else:
                        for index, item in enumerate(events):
                            if item[0] == 'capture':
                                self.assertEqual(events[index - 1], ('tensorboard',))
                                self.assertEqual(events[index + 1][0], 'scheduler' if item[1] == 7 else 'finalize')
                outcomes.append(([event for event in events if event[0] != 'capture'], losses))
        self.assertEqual(outcomes[0], outcomes[1])
