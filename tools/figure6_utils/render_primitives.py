from dataclasses import dataclass
from typing import Final, Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

from pcdet.utils.figure6_schema import StageState

from .contracts import Callout, Crop
from .loading import StageEvidence
from .palette import CLASS_COLORS
from .spatial import crop_bev_map


POINT_COLOR: Final[str] = '#707070'
CALLOUT_COLOR: Final[str] = '#CC0000'
TEXT_COLOR: Final[str] = '#263238'
MISSING_COLOR: Final[str] = '#F1F3F4'
EMPTY_COLOR: Final[str] = '#E7EEF2'
FAILED_COLOR: Final[str] = '#F6E8E5'
LABEL_SIZE: Final[float] = 5.0
RESPONSE_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    'figure6_response', ('#315B7D', '#F7F7F5', '#A44A3F'),
)
RELIABILITY_BOUNDARIES: Final[Tuple[float, ...]] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
RELIABILITY_CMAP: Final[ListedColormap] = ListedColormap(
    ('#C62828', '#FDD835', '#2E7D32'), name='figure6_reliability',
)
RELIABILITY_NORM: Final[BoundaryNorm] = BoundaryNorm(
    RELIABILITY_BOUNDARIES, RELIABILITY_CMAP.N, clip=True,
)
STATUS_STYLE = {
    StageState.MISSING: (MISSING_COLOR, '#9AA0A6', 'MISSING'),
    StageState.FAILED: (FAILED_COLOR, '#A04432', 'FAILED'),
    StageState.OBSERVED_EMPTY: (EMPTY_COLOR, '#607D8B', 'OBSERVED EMPTY'),
}


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class RenderDataError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class ReliabilityDisplay:
    callouts: Tuple[Callout, ...]
    crop: Crop


def style_axis(axis: Axes, crop: Crop) -> None:
    x_min, y_min, x_max, y_max = crop
    axis.set_xlim(y_min, y_max)
    axis.set_ylim(x_min, x_max)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_context(axis: Axes, points: Optional[np.ndarray]) -> None:
    if points is None or points.ndim != 2 or points.shape[1] < 3:
        return
    finite = np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
    axis.scatter(
        points[finite, 2], points[finite, 1], s=0.18, c=POINT_COLOR,
        alpha=0.20, linewidths=0.0, rasterized=True, zorder=1,
    )


def draw_callouts(axis: Axes, callouts: Tuple[Callout, ...]) -> None:
    for callout in callouts:
        x_min, y_min, x_max, y_max = callout.roi
        axis.add_patch(Rectangle(
            (y_min, x_min), y_max - y_min, x_max - x_min,
            fill=False, edgecolor=CALLOUT_COLOR, linewidth=1.0,
            alpha=0.72, zorder=4,
        ))


def draw_status(axis: Axes, stage: StageEvidence, crop: Crop) -> None:
    background, edge, label = STATUS_STYLE[stage.runtime_state]
    x_min, y_min, x_max, y_max = crop
    axis.set_facecolor(background)
    detail = stage.detail.strip()
    axis.text(
        (y_min + y_max) / 2.0, (x_min + x_max) / 2.0,
        label if not detail else '%s\n%s' % (label, detail),
        ha='center', va='center', color=edge, fontsize=LABEL_SIZE,
    )


def draw_unavailable(axis: Axes, crop: Crop, detail: str) -> None:
    x_min, y_min, x_max, y_max = crop
    axis.set_facecolor(MISSING_COLOR)
    axis.text(
        (y_min + y_max) / 2.0, (x_min + x_max) / 2.0,
        'SPATIAL DATA UNAVAILABLE\n' + detail,
        ha='center', va='center', color='#9AA0A6', fontsize=LABEL_SIZE,
    )


def box_corners(box: np.ndarray) -> Optional[np.ndarray]:
    if box.size < 7 or not np.isfinite(box[:7]).all() or (box[3:5] <= 0).any():
        return None
    center_x, center_y, length, width, yaw = box[0], box[1], box[3], box[4], box[6]
    local = np.array([
        [length / 2.0, width / 2.0], [length / 2.0, -width / 2.0],
        [-length / 2.0, -width / 2.0], [-length / 2.0, width / 2.0],
    ])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    corners = local @ np.array([[cosine, -sine], [sine, cosine]]).T
    return corners + np.array([center_x, center_y])


def class_color(label: int) -> Tuple[float, float, float]:
    if label < 1 or label > len(CLASS_COLORS):
        raise RenderDataError('class labels must be positive valid nuScenes class ids')
    red, green, blue = CLASS_COLORS[label - 1]
    return red / 255.0, green / 255.0, blue / 255.0


def draw_box(axis: Axes, box: np.ndarray, color: Tuple[float, float, float],
             linewidth: float = 1.1) -> None:
    corners = box_corners(box)
    if corners is None:
        return
    closed = np.concatenate((corners, corners[:1]), axis=0)
    axis.plot(
        closed[:, 1], closed[:, 0], color=color, linewidth=linewidth,
        alpha=0.92, solid_capstyle='round', zorder=3,
    )


def prediction_arrays(stage: StageEvidence) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    boxes = stage.arrays.get('pred_boxes')
    labels = stage.arrays.get('pred_labels')
    scores = stage.arrays.get('pred_scores')
    if boxes is None or labels is None or scores is None:
        return None
    labels = labels.reshape(-1)
    scores = scores.reshape(-1)
    if len(boxes) != len(labels) or len(boxes) != len(scores):
        raise RenderDataError('prediction boxes, labels, and scores must have equal lengths')
    return boxes, labels, scores


def draw_density(axis: Axes, stage: StageEvidence, crop: Crop,
                 density_cmap: LinearSegmentedColormap) -> None:
    points = stage.arrays.get('points')
    if points is not None and points.ndim == 2 and points.shape[1] >= 3:
        x_min, y_min, x_max, y_max = crop
        finite = np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
        counts, _, _ = np.histogram2d(
            points[finite, 2], points[finite, 1], bins=160,
            range=((y_min, y_max), (x_min, x_max)),
        )
        axis.imshow(np.log1p(counts).T, origin='lower', extent=(y_min, y_max, x_min, x_max),
                    cmap=density_cmap, interpolation='nearest', rasterized=True)
        return
    density = stage.arrays.get('tta_density_map')
    calibrated = None if density is None else crop_bev_map(density, stage.spatial_extent, crop)
    if calibrated is None:
        draw_unavailable(axis, crop, 'no captured points or calibrated density map')
        return
    axis.imshow(calibrated[0], origin='lower', extent=calibrated[1], cmap=density_cmap,
                interpolation='nearest', rasterized=True)


def draw_reliability(axis: Axes, stage: StageEvidence,
                     display: ReliabilityDisplay) -> None:
    arrays = prediction_arrays(stage)
    reliability = stage.arrays.get('spcra_reliability')
    if arrays is None or reliability is None:
        raise RenderDataError('prediction and reliability arrays are required')
    values = reliability.reshape(-1)
    if len(arrays[0]) != len(values):
        raise RenderDataError('prediction and reliability arrays must have equal lengths')
    for box, value in zip(arrays[0], values):
        if np.isfinite(value):
            clipped = float(np.clip(value, 0.0, 1.0))
            rgba = RELIABILITY_CMAP(RELIABILITY_NORM(clipped))
            draw_box(axis, box, (rgba[0], rgba[1], rgba[2]))
    used = set()
    for callout in display.callouts:
        center_x = (callout.roi[0] + callout.roi[2]) / 2.0
        center_y = (callout.roi[1] + callout.roi[3]) / 2.0
        candidates = tuple(
            index for index, (box, value) in enumerate(zip(arrays[0], values))
            if index not in used and np.isfinite(value)
            and callout.roi[0] <= box[0] <= callout.roi[2]
            and callout.roi[1] <= box[1] <= callout.roi[3]
        )
        if not candidates:
            continue
        selected = min(candidates, key=lambda index: (
            (arrays[0][index, 0] - center_x) ** 2
            + (arrays[0][index, 1] - center_y) ** 2,
            index,
        ))
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = display.crop
        vertical_margin = (crop_x_max - crop_x_min) * 0.02
        horizontal_margin = (crop_y_max - crop_y_min) * 0.02
        label_x = min(max(callout.roi[1], crop_y_min + horizontal_margin),
                      crop_y_max - horizontal_margin)
        above_space = crop_x_max - callout.roi[2]
        if above_space >= vertical_margin * 2.0:
            label_y = callout.roi[2] + vertical_margin
            vertical_alignment = 'bottom'
        else:
            below_space = callout.roi[0] - crop_x_min
            label_y = callout.roi[0] - min(vertical_margin, below_space * 0.5)
            vertical_alignment = 'top'
        axis.text(
            label_x, label_y, 'r=%.2f' % values[selected], fontsize=LABEL_SIZE,
            color=TEXT_COLOR, ha='left', va=vertical_alignment, clip_on=True, zorder=5,
            bbox={
                'boxstyle': 'round,pad=0.15', 'facecolor': '#FFFFFF',
                'edgecolor': 'none', 'alpha': 0.82,
            },
        )
        used.add(selected)


def draw_rgplm(axis: Axes, stage: StageEvidence) -> None:
    pseudo = stage.arrays.get('gt_boxes')
    class_column = 9 if stage.selected_stage == 'injection' else 7
    if pseudo is None or pseudo.ndim != 2 or pseudo.shape[1] <= class_column:
        raise RenderDataError('effective/injection gt_boxes do not match captured format')
    labels = pseudo[:, class_column]
    valid = np.isfinite(labels) & (labels > 0) & (labels <= len(CLASS_COLORS))
    valid &= np.equal(labels, np.floor(labels))
    for box, label in zip(pseudo[valid, :7], labels[valid].astype(np.int64)):
        draw_box(axis, box, class_color(int(label)), linewidth=1.6)


def draw_sgdfa(axis: Axes, stage: StageEvidence, crop: Crop, limit: float) -> None:
    delta = stage.arrays.get('delta')
    calibrated = None if delta is None else crop_bev_map(delta, stage.spatial_extent, crop)
    if calibrated is None or not np.isfinite(calibrated[0]).any():
        draw_unavailable(axis, crop, 'captured delta map lacks finite spatial calibration')
        return
    axis.imshow(calibrated[0], origin='lower', extent=calibrated[1], cmap=RESPONSE_CMAP,
                vmin=-limit, vmax=limit, interpolation='nearest', rasterized=True)


def draw_final(axis: Axes, stage: StageEvidence) -> None:
    arrays = prediction_arrays(stage)
    if arrays is None:
        raise RenderDataError('first-forward final_detection arrays are required')
    for box, label in zip(arrays[0], arrays[1]):
        draw_box(axis, box, class_color(int(label)), linewidth=1.1)
