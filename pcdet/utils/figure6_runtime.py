"""Per-process, caller-owned Figure 6 transactions; no persistence or global state."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from pcdet.utils.figure6_schema import (
    ArtifactError, CaptureArray, CaptureRecord, Occurrence,
    StageCapture, StageState, StageStatus,
)

RuntimeArray = Union[torch.Tensor, CaptureArray]


def snapshot(value: RuntimeArray) -> CaptureArray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone().numpy()
    return np.array(value, copy=True)


def stage(values: Mapping[str, RuntimeArray], owner: str, detail: str = '') -> StageCapture:
    arrays = {name: snapshot(value) for name, value in values.items()}
    state = StageState.MISSING
    if arrays:
        state = StageState.COMPLETE if any(value.size for value in arrays.values()) else StageState.OBSERVED_EMPTY
    return StageCapture(StageStatus(state, owner, detail), arrays)


@dataclass(frozen=True)  # noqa: SLOTS_OK - Python 3.8 runtime.
class AdapterCaptureRequest:
    """The response mapping is filled once by the first adapter invocation."""

    indices: Tuple[int, ...]
    responses: Dict[int, StageCapture] = field(default_factory=dict)


class Figure6RuntimeCollector:
    """Mutable transaction accumulator. Begin/finalize belong to the outer loop.

    Pass only selected Occurrences to begin. An empty selection is a no-op.
    Finalize returns immutable schema records and clears this transaction.
    """

    def __init__(self) -> None:
        self._identities: Tuple[Occurrence, ...] = ()
        self._protocol: Dict[str, str] = {}
        self._stages: Dict[int, Dict[str, StageCapture]] = {}
        self._sources: Dict[int, str] = {}

    def begin(self, identities: Sequence[Occurrence], protocol: Mapping[str, str]) -> None:
        if self._identities:
            raise ArtifactError('finalize the active Figure 6 transaction before beginning another')
        indices = [identity.batch_index for identity in identities]
        if len(set(indices)) != len(indices):
            raise ArtifactError('selected batch indices must be unique')
        self._identities = tuple(identities)
        self._protocol = dict(protocol)
        self._sources = {}
        self._stages = {index: {
            name: stage({}, owner, 'not observed') for name, owner in (
                ('input', 'current_pre_update'), ('final_detection', 'current_pre_update'),
                ('sg_dfa', 'current_pre_update'), ('injection', 'training_pseudo'),
            )
        } for index in indices}

    def inputs(self, batch_dict) -> None:
        for identity in self._identities:
            index = identity.batch_index
            points = batch_dict['points']
            self._stages[index]['input'] = stage(
                {'points': points[points[:, 0] == index]}, 'current_pre_update')
            for name, key in (('input_density', 'tta_density_map'),
                              ('input_lidar_transform', 'lidar_aug_matrix')):
                value = batch_dict.get(key)
                self._stages[index][name] = stage(
                    {} if value is None else {key: value[index]},
                    'current_pre_update', 'unavailable' if value is None else '')

    @contextmanager
    def first_forward(self, batch_dict) -> Iterator[None]:
        if not self._identities:
            yield
            return
        request = AdapterCaptureRequest(tuple(self._stages))
        batch_dict['_figure6_adapter_request'] = request
        try:
            yield
        finally:
            batch_dict.pop('_figure6_adapter_request', None)
            for index in request.indices:
                self._stages[index]['sg_dfa'] = request.responses.get(
                    index, stage({}, 'current_pre_update', 'adapter not observed'))
            request.responses.clear()

    def final_detection(self, predictions: Sequence[Mapping[str, RuntimeArray]]) -> None:
        for index, stages in self._stages.items():
            if stages['final_detection'].status.state is StageState.MISSING:
                stages['final_detection'] = stage(
                    {name: predictions[index][name] for name in
                     ('pred_boxes', 'pred_scores', 'pred_labels')}, 'current_pre_update')

    def spcra(self, predictions: Sequence[Mapping[str, RuntimeArray]], owner: str) -> None:
        for index, stages in self._stages.items():
            prediction = predictions[index]
            values = {name: prediction[name] for name in
                      ('pred_boxes', 'pred_scores', 'pred_labels', 'spcra_reliability')}
            stages['spcra.' + owner] = stage(values, owner)

    def pseudo(self, frame_id: str, infos, position: Tuple[str, str]) -> None:
        owner, name = position
        for identity in self._identities:
            if identity.frame_id != frame_id:
                continue
            count = len(infos['gt_boxes'])
            values = {key: value for key, value in infos.items()
                      if isinstance(value, (np.ndarray, torch.Tensor))
                      and value.ndim > 0 and len(value) == count}
            self._stages[identity.batch_index][name + '.' + owner] = stage(values, owner)

    def effective(self, pseudo_labels, owner: str) -> None:
        for identity in self._identities:
            if identity.frame_id in pseudo_labels:
                self.pseudo(identity.frame_id, pseudo_labels[identity.frame_id], (owner, 'effective_pseudo'))
                self._sources[identity.batch_index] = owner

    def injection(self, assigned: Optional[Mapping[str, RuntimeArray]], reason: str = '') -> None:
        for index, stages in self._stages.items():
            values = {} if assigned is None else {name: value[index] for name, value in assigned.items()}
            stages['injection'] = stage(values, 'training_pseudo', reason)

    def finalize(self) -> Tuple[CaptureRecord, ...]:
        records = tuple(CaptureRecord(
            identity, dict(self._protocol, final_pseudo_source=self._sources.get(identity.batch_index, 'unavailable')),
            self._stages[identity.batch_index],
        ) for identity in self._identities)
        self._identities = ()
        self._stages = {}
        self._sources = {}
        self._protocol = {}
        return records
