import importlib
import json
from pathlib import Path
import statistics
import time
from typing import Iterable, Mapping, NamedTuple, Optional, Protocol


_MB = 1024 ** 2
_GB = 1024 ** 3


class EfficiencyProfileConfigurationError(ValueError):
    """Raised when the efficiency profiling window is invalid."""


class ParameterProtocol(Protocol):
    requires_grad: bool

    def numel(self) -> int:
        ...


class OptimizerProtocol(Protocol):
    param_groups: Iterable[Mapping[str, Iterable[ParameterProtocol]]]


class EfficiencyProfileRun(NamedTuple):
    method: str
    entrypoint: str
    config: str
    boundary: str
    output_dir: Path
    model_parameters: Iterable[ParameterProtocol]
    optimizer: Optional[OptimizerProtocol] = None
    updated_parameters: Optional[Iterable[ParameterProtocol]] = None


def _unique_parameters(parameters):
    unique = []
    seen = set()
    for parameter in parameters:
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(parameter)
    return unique


def parameter_counts(model_parameters, optimizer=None, updated_parameters=None):
    """Return unique optimizer-updated, total, and ratio parameter counts."""
    model_parameters = _unique_parameters(model_parameters)
    total = sum(parameter.numel() for parameter in model_parameters)
    if updated_parameters is not None:
        candidates = updated_parameters
    elif optimizer is not None:
        candidates = (
            parameter
            for group in optimizer.param_groups
            for parameter in group['params']
        )
    else:
        candidates = ()
    trainable = sum(
        parameter.numel()
        for parameter in _unique_parameters(candidates)
        if parameter.requires_grad
    )
    ratio = trainable / total if total else 0.0
    return trainable, total, ratio


class EfficiencyProfiler:
    """Accumulates split CUDA-event segments for a fixed batch window."""

    def __init__(self, profile_cfg, run, logger=None, cuda_backend=None, clock=None):
        self.enabled = bool(profile_cfg.get('ENABLED', False))
        self.warmup_iters = profile_cfg.get('WARMUP_ITERS', 20)
        self.measure_iters = profile_cfg.get('MEASURE_ITERS', 200)
        self.output_name = str(profile_cfg.get('OUTPUT_NAME', 'efficiency_profile.json'))
        if not isinstance(self.warmup_iters, int) or self.warmup_iters < 0:
            raise EfficiencyProfileConfigurationError('WARMUP_ITERS must be an integer >= 0')
        if not isinstance(self.measure_iters, int) or self.measure_iters <= 0:
            raise EfficiencyProfileConfigurationError('MEASURE_ITERS must be an integer > 0')

        self.run = run
        self.logger = logger
        self.cuda = cuda_backend
        self.clock = clock or time.perf_counter
        self.device = None
        self._completed_batches = 0
        self._measurement_ready = False
        self._batch_open = False
        self._batch_measuring = False
        self._segment_start = None
        self._current_segments = []
        self._current_wall_segments = []
        self._measured_batches = []
        self._measured_wall_batches = []
        self._measured_samples = 0
        self._finalized = False
        self.report = None

        if self.enabled:
            if self.cuda is None:
                self.cuda = importlib.import_module('torch').cuda
            self.device = self.cuda.current_device()

    @property
    def complete(self):
        return self._finalized

    @property
    def _active_cuda(self):
        assert self.cuda is not None
        return self.cuda

    def _prepare_measurement(self):
        if self._measurement_ready:
            return
        self._active_cuda.synchronize(self.device)
        self._active_cuda.reset_peak_memory_stats(self.device)
        self._measurement_ready = True

    def begin_batch(self, sample_count):
        if not self.enabled or self._finalized:
            return
        if self._completed_batches >= self.warmup_iters:
            self._prepare_measurement()
            self._batch_measuring = True
        else:
            self._batch_measuring = False
        self._batch_open = True
        self._batch_samples = int(sample_count)
        self._current_segments = []
        self._current_wall_segments = []

    def begin_segment(self):
        if not self.enabled or self._finalized or not self._batch_measuring:
            return
        self._active_cuda.synchronize(self.device)
        self._wall_start = self.clock()
        self._segment_start = self._active_cuda.Event(enable_timing=True)
        self._segment_start.record()

    def end_segment(self):
        if not self.enabled or self._finalized or not self._batch_measuring:
            return
        end_event = self._active_cuda.Event(enable_timing=True)
        end_event.record()
        self._active_cuda.synchronize(self.device)
        self._current_segments.append((self._segment_start, end_event))
        self._current_wall_segments.append((self.clock() - self._wall_start) * 1000.0)
        self._segment_start = None

    def end_batch(self):
        if not self.enabled or self._finalized or not self._batch_open:
            return
        self._batch_open = False
        self._completed_batches += 1
        if not self._batch_measuring:
            if self._completed_batches == self.warmup_iters:
                self._prepare_measurement()
            return
        self._measured_batches.append(self._current_segments)
        self._measured_wall_batches.append(self._current_wall_segments)
        self._measured_samples += self._batch_samples
        if len(self._measured_batches) == self.measure_iters:
            self._finish()

    def finalize(self):
        """Finalize only after the configured fixed measurement window."""
        if len(self._measured_batches) == self.measure_iters and not self._finalized:
            self._finish()
        if self.enabled and not self._finalized:
            raise EfficiencyProfileConfigurationError(
                'Efficiency profiling ended before the measurement window completed'
            )
        return self.report

    def _finish(self):
        cuda_batch_times = [
            sum(start.elapsed_time(end) for start, end in segments)
            for segments in self._measured_batches
        ]
        batch_times = [sum(segments) for segments in self._measured_wall_batches]
        runtime_ms = sum(batch_times)
        mean_ms = statistics.mean(batch_times)
        std_ms = statistics.pstdev(batch_times)
        samples = self._measured_samples
        per_sample_ms = runtime_ms / samples if samples else 0.0
        fps = samples * 1000.0 / runtime_ms if runtime_ms > 0.0 else 0.0
        resident = self._active_cuda.memory_allocated(self.device)
        peak = self._active_cuda.max_memory_allocated(self.device)
        trainable, total, ratio = parameter_counts(
            self.run.model_parameters,
            optimizer=self.run.optimizer,
            updated_parameters=self.run.updated_parameters,
        )
        output_path = Path(self.run.output_dir) / self.output_name
        self.report = {
            'schema_version': 1,
            'method': self.run.method,
            'entrypoint': self.run.entrypoint,
            'config': self.run.config,
            'profile': {
                'enabled': self.enabled,
                'warmup_iters': self.warmup_iters,
                'measure_iters': self.measure_iters,
                'output_name': self.output_name,
            },
            'boundary': self.run.boundary,
            'cuda': {
                'device': self.device,
                'name': self._active_cuda.get_device_name(self.device),
            },
            'runtime': {
                'measured_iters': len(batch_times),
                'measured_samples': samples,
                'runtime_ms': runtime_ms,
                'mean_ms_per_batch': mean_ms,
                'std_ms_per_batch': std_ms,
                'per_sample_ms': per_sample_ms,
                'fps': fps,
                'cuda_runtime_ms': sum(cuda_batch_times),
                'cuda_mean_ms_per_batch': statistics.mean(cuda_batch_times),
            },
            'memory': {
                'resident_allocated_mb': resident / _MB,
                'resident_allocated_gb': resident / _GB,
                'peak_allocated_mb': peak / _MB,
                'peak_allocated_gb': peak / _GB,
                'peak_allocated_gib': peak / _GB,
            },
            'parameters': {
                'trainable': trainable,
                'trainable_m': trainable / 1e6,
                'total': total,
                'ratio': ratio,
                'ratio_percent': ratio * 100.0,
            },
            'output_path': str(output_path),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + '.tmp')
        with temporary_path.open('w', encoding='utf-8') as output_file:
            json.dump(self.report, output_file, indent=2, sort_keys=True)
            output_file.write('\n')
        temporary_path.replace(output_path)
        self._finalized = True
        if self.logger is not None:
            self.logger.info(
                '[Efficiency] method=%s batches=%d samples=%d '
                'runtime=%.3f ms/batch fps=%.3f peak=%.3f GiB '
                'trainable=%.6f M ratio=%.6f%% path=%s'
                % (
                    self.run.method, len(batch_times), samples, mean_ms, fps,
                    peak / _GB, trainable / 1e6, ratio * 100.0, output_path,
                )
            )
