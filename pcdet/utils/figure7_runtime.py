"""Standalone synchronous collector; integration supplies only the current occurrence."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np

from .figure7_artifacts import append_ledger, array_digest, atomic_text, stage_record, verify_record
from .figure7_schema import (
    Candidate, Figure7Error, LedgerRecord, Observation, PoolLimits, Provenance,
    SCHEMA_VERSION, Status,
)
from .figure7_selection import candidates, record_id, reference_rows, retain, select_pair


def summarize(frame: Observation) -> LedgerRecord:
    evidence, points = frame.evidence, frame.points
    rows = reference_rows(frame)
    point_arrays = () if points is None else (points.reference, *points.views)
    predictions = () if evidence is None else (evidence.reference_prediction, *evidence.view_predictions)
    masks = () if evidence is None else (evidence.reference_mask, *evidence.view_masks)
    rescues = () if evidence is None else (evidence.reference_rescue_mask, *evidence.view_rescue_masks)
    transforms = () if points is None else (points.reference_transform,)
    if evidence is not None:
        transforms += evidence.transforms
    completion = (evidence is not None, points is not None,
                  *(evidence is not None and len(evidence.view_predictions) > index
                    and points is not None and len(points.views) > index for index in range(4)))
    status: Status = 'ineligible'
    if any(row.stable_eligible or row.variable_eligible for row in rows):
        status = 'complete'
    if not rows:
        status = 'empty'
    if not all(completion):
        status = 'incomplete'
    if frame.failure:
        status = 'failed'
    return LedgerRecord(
        SCHEMA_VERSION, record_id(frame), frame.identity, frame.model_step, status, frame.failure,
        tuple(len(value) for value in point_arrays), tuple(len(value.boxes) for value in predictions),
        tuple(int(np.count_nonzero(value)) for value in masks),
        tuple(int(np.count_nonzero(value)) for value in rescues),
        tuple(tuple(float(value) for value in matrix.ravel()) for matrix in transforms),
        () if evidence is None else evidence.fingerprints,
        tuple(array_digest(value) for value in point_arrays),
        () if evidence is None else evidence.attempts,
        '' if evidence is None else evidence.law_identifier, completion,
        () if evidence is None else tuple(float(value) for value in evidence.coverage),
        () if evidence is None else tuple(float(value) for value in evidence.support), rows,
    )


class Figure7Collector:
    """Single-owner mutable pool summaries; never retain points, evidence or a model.

    Each observe call appends once, including failed or repeated observations.
    Persistent I/O failures are fatal: stop rather than silently drop evidence.
    At most M committed records plus one staging record exist during admission.
    """

    def __init__(self, output: Path, provenance: Provenance, limits: PoolLimits = PoolLimits()) -> None:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'records').mkdir()
        self.output = output
        self.limits = limits
        self.retained: tuple[Candidate, ...] = ()
        self._closed = False
        self._failed = False
        atomic_text(output / 'run.json', json.dumps({
            'schema_version': SCHEMA_VERSION, 'provenance': asdict(provenance),
            'limits': asdict(limits), 'owner': 'current_pre_update',
            'completion_order': ['reference_prediction', 'reference_points', 'view_0', 'view_1', 'view_2', 'view_3'],
            'selection_policy': 'case_a_q85_r30__case_b_q65_r80_all__v2',
        }, sort_keys=True))
        atomic_text(output / 'selection.json', json.dumps({
            'schema_version': SCHEMA_VERSION, 'status': 'incomplete', 'detail': 'stream open',
        }))

    def observe(self, frame: Observation) -> LedgerRecord:
        if self._closed or self._failed:
            raise Figure7Error('collector is closed or failed')
        ledger = LedgerRecord(
            SCHEMA_VERSION, record_id(frame), frame.identity, frame.model_step, 'failed',
            'observation processing failed', (), (), (), (), (), (), (), (), '', (), (), (), (),
        )
        try:
            ledger = summarize(frame)
            if frame.failure or not all(ledger.completion):
                return ledger
            proposed = retain(self.retained + candidates(frame, ledger.references), self.limits)
            admitted = tuple(item.pool for item in proposed if item.record_id == ledger.record_id)
            if not admitted:
                return ledger
            ledger = replace(ledger, status='complete', admitted_pools=admitted)
            previous_ids = {item.record_id for item in self.retained}
            proposed_ids = {item.record_id for item in proposed}
            if ledger.record_id in previous_ids:
                raise Figure7Error('a committed occurrence identity cannot be reused')
            staging = stage_record(self.output, frame, ledger)
            for evicted in sorted(previous_ids - proposed_ids):
                shutil.rmtree(self.output / 'records' / evicted)
            staging.replace(self.output / 'records' / ledger.record_id)
            self.retained = proposed
            return ledger
        except (OSError, ValueError, IndexError, TypeError) as error:
            self._failed = True
            ledger = replace(ledger, status='failed', detail=str(error), admitted_pools=())
            atomic_text(self.output / 'selection.json', json.dumps({
                'schema_version': SCHEMA_VERSION, 'status': 'failed', 'detail': str(error),
            }))
            raise
        finally:
            append_ledger(self.output, ledger)

    def finalize(self) -> tuple[Candidate, Candidate]:
        if self._failed or self._closed:
            raise Figure7Error('collector is closed or failed')
        self._closed = True
        try:
            pair = select_pair(self.retained)
            for item in pair:
                verify_record(self.output / 'records' / item.record_id)
        except (OSError, ValueError) as error:
            atomic_text(self.output / 'selection.json', json.dumps({
                'schema_version': SCHEMA_VERSION, 'status': 'incomplete', 'detail': str(error),
            }))
            raise
        atomic_text(self.output / 'selection.json', json.dumps({
            'schema_version': SCHEMA_VERSION, 'status': 'complete',
            'case_a': asdict(pair[0]), 'case_b': asdict(pair[1]),
            'retained': [asdict(item) for item in self.retained],
        }, sort_keys=True, allow_nan=False))
        return pair
