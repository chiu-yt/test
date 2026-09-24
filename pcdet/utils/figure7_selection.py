"""Deterministic pre-update selection only; no GT or evaluation inputs."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from typing import Final

import numpy as np

from .figure7_schema import (
    CLASS_NAMES, Candidate, Figure7Error, Observation, Pool, PoolLimits, RankKey, ReferenceRow,
)

PRIORITY: Final[tuple[str, ...]] = ('pedestrian', 'bicycle', 'motorcycle', 'traffic_cone')
FAR_RANGE_M: Final[float] = 30.


def record_id(frame: Observation) -> str:
    identity = {'occurrence': asdict(frame.identity), 'model_step': frame.model_step}
    return sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def reference_rows(frame: Observation) -> tuple[ReferenceRow, ...]:
    evidence = frame.evidence
    if evidence is None:
        return ()
    prediction = evidence.reference_prediction
    rows = []
    for index, raw_score in enumerate(prediction.scores):
        label = float(prediction.labels[index])
        class_name = CLASS_NAMES[int(label) - 1] if label.is_integer() and 1 <= label <= 10 else 'invalid'
        score = float(raw_score) if np.isfinite(raw_score) else None
        distance = float(np.linalg.norm(prediction.boxes[index, :2]))
        range_m = distance if math.isfinite(distance) else None
        accepted = bool(evidence.reference_mask[index])
        matches = tuple(int(value) for value in evidence.match_indices[index])
        qualities = tuple(float(value) for value in evidence.view_quality[index])
        reliability = float(evidence.reliability[index])
        priority = PRIORITY.index(class_name) if class_name in PRIORITY else (
            4 if class_name == 'car' and range_m is not None and range_m >= FAR_RANGE_M else 5)
        tie = (frame.identity.token, frame.identity.epoch, frame.model_step,
               frame.identity.global_rank, frame.identity.batch_index, index)
        stable_key = variable_key = None
        if accepted and score is not None and math.isfinite(reliability):
            if score >= .65 and reliability >= .80 and all(value >= 0 for value in matches):
                stable_key = (priority, -score, -reliability, -min(qualities), *tie)
            if score >= .85 and reliability <= .30:
                variable_key = (priority, -(score - reliability), -score, reliability, *tie)
        rows.append(ReferenceRow(
            index, class_name, score, accepted, bool(evidence.reference_rescue_mask[index]),
            matches, qualities, reliability, range_m, stable_key, variable_key,
        ))
    return tuple(rows)


def candidates(frame: Observation, rows: tuple[ReferenceRow, ...]) -> tuple[Candidate, ...]:
    result = []
    for row in rows:
        keys: tuple[tuple[Pool, RankKey | None], ...] = (
            ('stable', row.stable_key), ('variable', row.variable_key))
        for pool, key in keys:
            if key is not None and row.score is not None:
                result.append(Candidate(
                    record_id(frame), frame.identity, frame.model_step, row.index,
                    row.class_name, row.score, row.reliability, pool, key,
                ))
    return tuple(result)


def retain(candidates_: tuple[Candidate, ...], limits: PoolLimits) -> tuple[Candidate, ...]:
    """Top candidate per token per pool; physical storage uses their record-id union."""
    kept = []
    for pool, capacity in (('stable', limits.stable), ('variable', limits.variable)):
        tokens: set[str] = set()
        ranked = sorted((item for item in candidates_ if item.pool == pool),
                        key=lambda item: (item.rank_key, item.record_id))
        for item in ranked:
            if item.identity.token not in tokens and len(tokens) < capacity:
                kept.append(item)
                tokens.add(item.identity.token)
    return tuple(kept)


def select_pair(retained: tuple[Candidate, ...]) -> tuple[Candidate, Candidate]:
    """Case A then Case B: same class, class priority, closest q, then largest r gap.

    Thresholds guarantee a reliability gap of at least 0.50. Far-range car means
    >=30m in the reference input frame; other classes remain fallback candidates.
    """
    pairs = [(variable, stable) for variable in retained for stable in retained
             if variable.pool == 'variable' and stable.pool == 'stable'
             and variable.identity.token != stable.identity.token]
    if not pairs:
        raise Figure7Error('insufficient complete candidates on distinct sample tokens')
    return min(pairs, key=lambda pair: (
        pair[0].class_name != pair[1].class_name,
        max(pair[0].rank_key[0], pair[1].rank_key[0]),
        abs(pair[0].score - pair[1].score),
        -(pair[1].reliability - pair[0].reliability),
        pair[0].rank_key, pair[1].rank_key, pair[0].record_id, pair[1].record_id,
    ))
