from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spcra_k4_core import K4Result, Predictions
from .spcra_k4_seeding import ViewSamples


@dataclass(frozen=True, slots=True)
class PredictionEvidence:
    boxes: NDArray[np.float64]
    labels: NDArray[np.float64]
    scores: NDArray[np.float64]
    camera_support: NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class K4EvidenceInput:
    reference: Predictions
    views: tuple[Predictions, ...]
    samples: ViewSamples


@dataclass(frozen=True, slots=True)
class K4Evidence:
    reference_prediction: PredictionEvidence
    view_predictions: tuple[PredictionEvidence, ...]
    reference_mask: NDArray[np.bool_]
    view_masks: tuple[NDArray[np.bool_], ...]
    reference_rescue_mask: NDArray[np.bool_]
    view_rescue_masks: tuple[NDArray[np.bool_], ...]
    match_indices: NDArray[np.int64]
    view_quality: NDArray[np.float64]
    reliability: NDArray[np.float64]
    retained_indices: tuple[NDArray[np.int64], ...]
    transforms: tuple[NDArray[np.float32], ...]
    fingerprints: tuple[str, ...]
    attempts: tuple[int, ...]
    law_identifier: str
    coverage: NDArray[np.float64]
    support: NDArray[np.float64]


class K4EvidenceSink(Protocol):
    def __call__(self, evidence: K4Evidence) -> None: ...


def _owned_float(value: ArrayLike) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, order='C', copy=True)
    array.setflags(write=False)
    return array


def _owned_float32(value: ArrayLike) -> NDArray[np.float32]:
    array = np.array(value, dtype=np.float32, order='C', copy=True)
    array.setflags(write=False)
    return array


def _owned_bool(value: ArrayLike) -> NDArray[np.bool_]:
    array = np.array(value, dtype=np.bool_, order='C', copy=True)
    array.setflags(write=False)
    return array


def _owned_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.array(value, dtype=np.int64, order='C', copy=True)
    array.setflags(write=False)
    return array


def _prediction_evidence(prediction: Predictions) -> PredictionEvidence:
    support = prediction.camera_support
    return PredictionEvidence(
        boxes=_owned_float(prediction.boxes),
        labels=_owned_float(prediction.labels),
        scores=_owned_float(prediction.scores),
        camera_support=None if support is None else _owned_float(support),
    )


def build_k4_evidence(source: K4EvidenceInput, result: K4Result) -> K4Evidence:
    return K4Evidence(
        reference_prediction=_prediction_evidence(source.reference),
        view_predictions=tuple(_prediction_evidence(view) for view in source.views),
        reference_mask=_owned_bool(result.reference_mask),
        view_masks=tuple(_owned_bool(mask) for mask in result.view_masks),
        reference_rescue_mask=_owned_bool(result.reference_rescue_mask),
        view_rescue_masks=tuple(
            _owned_bool(mask) for mask in result.view_rescue_masks
        ),
        match_indices=_owned_int(result.match_indices),
        view_quality=_owned_float(result.view_quality),
        reliability=_owned_float(result.reliability),
        retained_indices=tuple(_owned_int(indices) for indices in source.samples.indices),
        transforms=tuple(
            _owned_float32(transform) for transform in source.samples.transforms
        ),
        fingerprints=tuple(source.samples.fingerprints),
        attempts=tuple(source.samples.attempts),
        law_identifier=source.samples.law_identifier,
        coverage=_owned_float(result.coverage),
        support=_owned_float(result.support),
    )
