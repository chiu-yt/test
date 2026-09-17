import ast
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from pcdet.utils.efficiency_profiler import (
    EfficiencyProfileConfigurationError,
    EfficiencyProfileRun,
    EfficiencyProfiler,
    parameter_counts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / 'tools' / 'cfgs' / 'nuscenes_models'


class FakeParameter:
    def __init__(self, size, requires_grad=True):
        self.size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self.size


class FakeOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups


class FakeEvent:
    def __init__(self, backend):
        self.backend = backend
        self.timestamp = None

    def record(self):
        self.timestamp = self.backend.timestamps.pop(0)

    def elapsed_time(self, other):
        return other.timestamp - self.timestamp


class FakeCuda:
    def __init__(self, timestamps):
        self.timestamps = list(timestamps)
        self.event_flags = []
        self.synchronizations = 0
        self.peak_resets = []

    def Event(self, enable_timing):
        self.event_flags.append(enable_timing)
        return FakeEvent(self)

    def current_device(self):
        return 0

    def get_device_name(self, device):
        return 'Fake CUDA Device %d' % device

    def synchronize(self, device=None):
        self.synchronizations += 1

    def reset_peak_memory_stats(self, device):
        self.peak_resets.append(device)

    def memory_allocated(self, device):
        return 256 * 1024 ** 2

    def max_memory_allocated(self, device):
        return 512 * 1024 ** 2


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class FakeClock:
    def __init__(self, timestamps):
        self.timestamps = list(timestamps)

    def __call__(self):
        return self.timestamps.pop(0)


def _run(output_dir, optimizer=None, updated_parameters=None):
    parameters = [FakeParameter(5), FakeParameter(3, requires_grad=False)]
    return EfficiencyProfileRun(
        method='source_only',
        entrypoint='test.py',
        config='cfgs/nuscenes_models/bevfusion.yaml',
        boundary='full_batch',
        output_dir=output_dir,
        model_parameters=parameters,
        optimizer=optimizer,
        updated_parameters=updated_parameters,
    )


def _profile_batch(profiler, samples):
    profiler.begin_batch(samples)
    profiler.begin_segment()
    profiler.end_segment()
    profiler.begin_segment()
    profiler.end_segment()
    profiler.end_batch()


def _merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if key == '_BASE_CONFIG_':
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path):
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    base_name = data.get('_BASE_CONFIG_')
    if base_name is None:
        return data
    base = yaml.safe_load((REPO_ROOT / 'tools' / base_name).read_text(encoding='utf-8'))
    return _merge(base, data)


class EfficiencyProfilerTest(unittest.TestCase):
    def test_disabled_profiler_is_noop_and_writes_no_file(self):
        # Given a disabled profile and a CUDA backend that would expose any use.
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            cuda = FakeCuda([])
            profiler = EfficiencyProfiler(
                {'ENABLED': False}, _run(output_dir), cuda_backend=cuda
            )

            # When arbitrary batch and segment calls are made.
            _profile_batch(profiler, samples=2)
            profiler.finalize()

            # Then profiling is inert and creates no output or CUDA work.
            self.assertEqual(list(output_dir.iterdir()), [])
            self.assertEqual(cuda.event_flags, [])
            self.assertEqual(cuda.synchronizations, 0)

    def test_invalid_measurement_windows_raise_specific_error(self):
        # Given invalid warmup and measurement bounds.
        invalid_configs = [
            {'ENABLED': True, 'WARMUP_ITERS': -1, 'MEASURE_ITERS': 1},
            {'ENABLED': True, 'WARMUP_ITERS': 0, 'MEASURE_ITERS': 0},
        ]

        # When each profiler is constructed, then its configuration is rejected.
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(EfficiencyProfileConfigurationError):
                    EfficiencyProfiler(config, _run(Path('.')), cuda_backend=FakeCuda([]))

    def test_warmup_split_segments_and_fixed_measurement_are_accounted(self):
        # Given one warmup batch and two measured batches with two CUDA segments each.
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            cuda = FakeCuda([0.0, 2.0, 4.0, 7.0, 10.0, 15.0, 17.0, 20.0])
            logger = FakeLogger()
            profiler = EfficiencyProfiler(
                {
                    'ENABLED': True,
                    'WARMUP_ITERS': 1,
                    'MEASURE_ITERS': 2,
                    'OUTPUT_NAME': 'profile.json',
                },
                _run(output_dir, updated_parameters=()),
                logger=logger,
                cuda_backend=cuda,
                clock=FakeClock([0.0, 0.002, 0.004, 0.007, 0.010, 0.015, 0.017, 0.020]),
            )

            # When warmup, exactly two measured batches, and an extra batch run.
            _profile_batch(profiler, samples=99)
            _profile_batch(profiler, samples=2)
            _profile_batch(profiler, samples=3)
            _profile_batch(profiler, samples=7)
            profiler.finalize()

            # Then only measured segment sums contribute and continuation is inert.
            report = json.loads((output_dir / 'profile.json').read_text(encoding='utf-8'))
            runtime = report['runtime']
            self.assertEqual(runtime['measured_iters'], 2)
            self.assertEqual(runtime['measured_samples'], 5)
            self.assertAlmostEqual(runtime['runtime_ms'], 13.0)
            self.assertAlmostEqual(runtime['mean_ms_per_batch'], 6.5)
            self.assertAlmostEqual(runtime['std_ms_per_batch'], 1.5)
            self.assertAlmostEqual(runtime['per_sample_ms'], 2.6)
            self.assertAlmostEqual(runtime['fps'], 5000.0 / 13.0)
            self.assertEqual(cuda.peak_resets, [0])
            self.assertEqual(cuda.synchronizations, 9)
            self.assertEqual(cuda.event_flags, [True] * 8)
            self.assertEqual(len(logger.messages), 1)
            self.assertTrue(logger.messages[0].startswith('[Efficiency]'))
            self.assertIn('method=source_only', logger.messages[0])
            self.assertIn('runtime=6.500 ms/batch', logger.messages[0])
            self.assertIn('fps=384.615', logger.messages[0])
            self.assertIn('peak=0.500 GiB', logger.messages[0])
            self.assertIn('trainable=0.000000 M', logger.messages[0])
            self.assertIn('ratio=0.000000%', logger.messages[0])

    def test_json_schema_reports_device_memory_parameters_and_output(self):
        # Given one measured source-only batch and deterministic CUDA memory values.
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            profiler = EfficiencyProfiler(
                {'ENABLED': True, 'WARMUP_ITERS': 0, 'MEASURE_ITERS': 1},
                _run(output_dir, updated_parameters=()),
                logger=FakeLogger(),
                cuda_backend=FakeCuda([1.0, 5.0, 8.0, 14.0]),
                clock=FakeClock([0.001, 0.005, 0.008, 0.014]),
            )

            # When the measurement completes.
            _profile_batch(profiler, samples=2)
            output_path = output_dir / 'efficiency_profile.json'
            report = json.loads(output_path.read_text(encoding='utf-8'))

            # Then the final-table schema and exact unit conversions are present.
            self.assertEqual(report['schema_version'], 1)
            self.assertEqual(report['method'], 'source_only')
            self.assertEqual(report['entrypoint'], 'test.py')
            self.assertEqual(report['config'], 'cfgs/nuscenes_models/bevfusion.yaml')
            self.assertEqual(report['profile'], {
                'enabled': True,
                'warmup_iters': 0,
                'measure_iters': 1,
                'output_name': 'efficiency_profile.json',
            })
            self.assertEqual(report['boundary'], 'full_batch')
            self.assertEqual(report['cuda'], {'device': 0, 'name': 'Fake CUDA Device 0'})
            self.assertEqual(report['memory']['resident_allocated_mb'], 256.0)
            self.assertEqual(report['memory']['resident_allocated_gb'], 0.25)
            self.assertEqual(report['memory']['peak_allocated_mb'], 512.0)
            self.assertEqual(report['memory']['peak_allocated_gb'], 0.5)
            self.assertEqual(report['memory']['peak_allocated_gib'], 0.5)
            self.assertEqual(report['parameters'], {
                'trainable': 0,
                'trainable_m': 0.0,
                'total': 8,
                'ratio': 0.0,
                'ratio_percent': 0.0,
            })
            self.assertEqual(report['output_path'], str(output_path))

    def test_parameter_counts_deduplicate_optimizer_parameters_by_identity(self):
        # Given tied optimizer entries, one frozen entry, and one distinct trainable entry.
        first = FakeParameter(5)
        frozen = FakeParameter(3, requires_grad=False)
        second = FakeParameter(7)
        optimizer = FakeOptimizer([
            {'params': [first, frozen]},
            {'params': [first, second]},
        ])

        # When optimizer-updated and total model parameters are counted.
        trainable, total, ratio = parameter_counts(
            [first, frozen, second, first], optimizer=optimizer
        )
        source_trainable, _, source_ratio = parameter_counts(
            [first, frozen, second], updated_parameters=()
        )

        # Then identity de-duplication and requires_grad are enforced, including explicit zero.
        self.assertEqual((trainable, total), (12, 15))
        self.assertEqual(ratio, 12.0 / 15.0)
        self.assertEqual(source_trainable, 0)
        self.assertEqual(source_ratio, 0.0)

    def test_all_relevant_bevfusion_configs_inherit_disabled_defaults(self):
        # Given every root BEVFusion config used by the efficiency comparison table.
        names = [
            'bevfusion.yaml',
            'bevfusion_mos.yaml',
            'bevfusion_codemerge.yaml',
            'bevfusion_tent.yaml',
            'bevfusion_sar.yaml',
        ]

        # When their actual top-level base chains are resolved.
        profiles = [_load_config(CONFIG_DIR / name)['TTA']['EFFICIENCY_PROFILE'] for name in names]

        # Then command-line override keys exist everywhere with disabled shared defaults.
        expected = {
            'ENABLED': False,
            'WARMUP_ITERS': 20,
            'MEASURE_ITERS': 200,
            'OUTPUT_NAME': 'efficiency_profile.json',
        }
        self.assertTrue(all(profile == expected for profile in profiles))

    def test_profiler_source_is_python_38_compatible(self):
        # Given the profiler source parsed without importing CUDA dependencies.
        source_path = REPO_ROOT / 'pcdet' / 'utils' / 'efficiency_profiler.py'
        tree = ast.parse(source_path.read_text(encoding='utf-8'), feature_version=8)

        # When Python-3.10-only syntax is inspected, then none is present.
        self.assertFalse(any(
            isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
            for node in ast.walk(tree)
        ))
        self.assertFalse(any(
            keyword.arg == 'slots'
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
        ))


if __name__ == '__main__':
    unittest.main()
