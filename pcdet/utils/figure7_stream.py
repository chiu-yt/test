"""One-batch, pre-update Figure 7 transactions on the existing MOS stream."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np

from .figure6_schema import CaptureArray, Occurrence
from .figure7_runtime import Figure7Collector
from .figure7_schema import CurrentPoints, Figure7Error, Observation, PoolLimits, Provenance

if TYPE_CHECKING:
    from pcdet.tta_methods.spcra_k4_evidence import K4Evidence


class Figure7StreamCapture:
    """Mutable current-batch transaction; the collector owns only bounded summaries."""

    def __init__(self, collector: Figure7Collector) -> None:
        self.collector = collector
        self.identities: tuple[Occurrence, ...] = ()
        self._reference: tuple[CaptureArray, ...] = ()
        self._transforms: tuple[CaptureArray, ...] = ()
        self._views: list[tuple[CaptureArray, ...]] = []
        self._evidence: list[K4Evidence] = []

    @classmethod
    def from_config(cls, run_cfg, provenance, ckpt_dir: Path) -> Figure7StreamCapture:
        from pcdet.tta_methods.spcra_k4_config import validate_k4_config

        if not validate_k4_config(run_cfg):
            raise Figure7Error('Figure 7 capture requires enabled formal K4')
        capture = run_cfg.TTA.FIGURE7_CAPTURE
        limits = PoolLimits(capture.get('STABLE_CAPACITY', 10), capture.get('VARIABLE_CAPACITY', 10))
        output = Path(capture['OUTPUT_DIR']) if capture.get('OUTPUT_DIR') else ckpt_dir.parent / 'figure7_capture'
        return cls(Figure7Collector(output, Provenance(**provenance), limits))

    def begin(self, batch, progress: tuple[int, int, int, int]) -> None:
        if self.identities:
            raise Figure7Error('previous Figure 7 transaction is still open')
        epoch, iteration, samples_seen, batch_size = progress
        metadata, frames = batch['metadata'], batch['frame_id']
        if len(metadata) != batch_size or len(frames) != batch_size:
            raise Figure7Error('Figure 7 metadata must retain batch order')
        self.identities = tuple(Occurrence(
            metadata[index]['token'], frames[index], epoch, iteration,
            samples_seen, 0, 1, index, batch_size,
        ) for index in range(batch_size))

    def _split(self, points: CaptureArray) -> tuple[CaptureArray, ...]:
        return tuple(np.array(points[points[:, 0] == identity.batch_index, 1:], copy=True)
                     for identity in self.identities)

    def reference(self, points: CaptureArray, transforms: CaptureArray) -> None:
        self._reference = self._split(points)
        self._transforms = tuple(np.array(matrix, copy=True) for matrix in transforms)

    def view(self, points: CaptureArray) -> None:
        if len(self._views) >= 4:
            raise Figure7Error('Figure 7 accepts exactly four actual view inputs')
        self._views.append(self._split(points))

    def evidence(self, evidence: K4Evidence) -> None:
        if len(self._evidence) >= len(self.identities):
            raise Figure7Error('K4 evidence exceeds the current batch')
        self._evidence.append(evidence)

    @contextmanager
    def transaction(self):
        try:
            yield
        finally:
            error = sys.exc_info()[1]
            self.finish('' if error is None else f'{type(error).__name__}: {error}')

    def finish(self, failure: str = '') -> None:
        try:
            for index, identity in enumerate(self.identities):
                points = None
                detail = failure
                try:
                    if self._reference and len(self._views) == 4:
                        points = CurrentPoints(self._reference[index],
                                               tuple(view[index] for view in self._views),
                                               self._transforms[index])
                except (ValueError, IndexError, TypeError) as error:
                    point_failure = f'{type(error).__name__}: {error}'
                    detail = '; '.join(value for value in (failure, point_failure) if value)
                evidence = self._evidence[index] if index < len(self._evidence) else None
                self.collector.observe(Observation(
                    identity, identity.accumulated_iter_before, evidence, points, detail,
                ))
        finally:
            self.identities = ()
            self._reference = ()
            self._transforms = ()
            self._views.clear()
            self._evidence.clear()

    def finalize(self) -> None:
        self.collector.finalize()
