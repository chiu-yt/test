from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from pcdet.utils.tta_utils import (
    _apply_lidar_aug_matrix_to_boxes_np,
    _normalize_tta_proposal_boxes_np,
    build_tta_density_map,
)


@dataclass(frozen=True)  # noqa: SLOTS_OK - Python 3.8 runtime.
class DensityMapSpec:
    point_cloud_range: tuple[float, float, float, float, float, float]
    grid_size: int = 32


@dataclass(frozen=True)  # noqa: SLOTS_OK - Python 3.8 runtime.
class AdapterInputs:
    density_map: torch.Tensor
    proposal_boxes: torch.Tensor
    proposal_mask: torch.Tensor


class K4ContextError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


def _owned_boxes(value: ArrayLike) -> NDArray[np.float32]:
    normalized = _normalize_tta_proposal_boxes_np(np.asarray(value))
    boxes = np.array(normalized, dtype=np.float32, order='C', copy=True)
    boxes.setflags(write=False)
    return boxes


def _owned_mask(value: ArrayLike, row_count: int) -> NDArray[np.bool_]:
    mask = np.array(value, dtype=np.bool_, order='C', copy=True)
    if mask.shape != (row_count,):
        raise K4ContextError(f'proposal mask must have shape ({row_count},)')
    mask.setflags(write=False)
    return mask


def _frame_values(value, batch_size: int, frame_ndim: int):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if batch_size == 1 and value.ndim == frame_ndim:
            return (value,)
        return tuple(value)
    return tuple(value)


@dataclass(frozen=True)  # noqa: SLOTS_OK - Python 3.8 runtime.
class ReferenceProposalContext:
    """Detached reference proposals whose row identity is fixed across all views."""

    boxes: tuple[NDArray[np.float32], ...]
    masks: tuple[NDArray[np.bool_], ...]

    @classmethod
    def empty(cls, batch_size: int) -> 'ReferenceProposalContext':
        return cls.from_inputs(None, None, batch_size)

    @classmethod
    def from_inputs(
        cls, proposal_boxes, proposal_mask, batch_size: int,
    ) -> 'ReferenceProposalContext':
        if proposal_boxes is None:
            boxes = tuple(_owned_boxes(np.empty((0, 9), dtype=np.float32))
                          for _ in range(batch_size))
            masks = tuple(_owned_mask(np.empty(0, dtype=np.bool_), 0)
                          for _ in range(batch_size))
            return cls(boxes, masks)

        frame_boxes = _frame_values(proposal_boxes, batch_size, frame_ndim=2)
        if len(frame_boxes) != batch_size:
            raise K4ContextError('proposal boxes must match batch size')
        boxes = tuple(_owned_boxes(value) for value in frame_boxes)
        if proposal_mask is None:
            masks = tuple(_owned_mask(np.ones(len(value), dtype=np.bool_), len(value))
                          for value in boxes)
        else:
            frame_masks = _frame_values(proposal_mask, batch_size, frame_ndim=1)
            if len(frame_masks) != batch_size:
                raise K4ContextError('proposal masks must match batch size')
            masks = tuple(_owned_mask(mask, len(box))
                          for box, mask in zip(boxes, frame_masks))
        return cls(boxes, masks)

    @classmethod
    def from_predictions(cls, predictions) -> 'ReferenceProposalContext':
        proposal_boxes = []
        for prediction in predictions:
            values = []
            for key in ('pred_boxes', 'pred_labels', 'pred_scores'):
                value = prediction[key]
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                values.append(np.asarray(value))
            boxes, labels, scores = values
            labels = labels.reshape(-1, 1)
            scores = scores.reshape(-1, 1)
            proposal_boxes.append(np.concatenate((boxes[:, :7], labels, scores), axis=1))
        return cls.from_inputs(tuple(proposal_boxes), None, len(proposal_boxes))

    def for_view(
        self, points: torch.Tensor, view_deltas, density_spec: DensityMapSpec,
    ) -> AdapterInputs:
        deltas = view_deltas.detach().cpu().numpy() if torch.is_tensor(view_deltas) else np.asarray(view_deltas)
        if len(self.boxes) == 1 and deltas.ndim == 2:
            deltas = deltas[None]
        if deltas.shape != (len(self.boxes), 4, 4):
            raise K4ContextError('view deltas must have shape (batch_size, 4, 4)')

        max_rows = max((len(boxes) for boxes in self.boxes), default=0)
        transformed = np.zeros((len(self.boxes), max_rows, 9), dtype=np.float32)
        padded_mask = np.zeros((len(self.boxes), max_rows), dtype=np.bool_)
        for index, (boxes, mask) in enumerate(zip(self.boxes, self.masks)):
            row_count = len(boxes)
            if row_count == 0:
                continue
            active = np.flatnonzero(mask)
            if active.size:
                transformed[index, active] = _apply_lidar_aug_matrix_to_boxes_np(
                    boxes[active], deltas[index], inverse=False,
                )
            padded_mask[index, :row_count] = mask

        density = build_tta_density_map(
            points, len(self.boxes), density_spec.point_cloud_range, density_spec.grid_size,
        )
        proposal_boxes = torch.as_tensor(transformed, device=points.device, dtype=points.dtype)
        proposal_mask = torch.as_tensor(padded_mask, device=points.device, dtype=torch.bool)
        return AdapterInputs(density, proposal_boxes, proposal_mask)


def compose_view_calibration(view_deltas: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    composed = view_deltas.to(torch.float64) @ reference.to(torch.float64)
    return composed.to(view_deltas.dtype)
