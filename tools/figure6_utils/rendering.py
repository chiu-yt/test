"""Publication rendering primitives for the five aligned Figure 6 stages."""

from pathlib import Path
from typing import Callable, Final, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import numpy as np

from pcdet.utils.figure6_schema import StageState
from .contracts import Crop
from .loading import StageEvidence, TokenEvidence
from .palette import CLASS_COLORS, CLASS_NAMES
from .spatial import crop_bev_map


PANEL_SIZE: Final[Tuple[float, float]] = (3.45, 1.95)
PLATE_WIDTH: Final[float] = 7.2
TITLE_SIZE: Final[float] = 6.5
LABEL_SIZE: Final[float] = 5.0
ROW_SIZE: Final[float] = 5.5
POINT_SIZE: Final[float] = 0.45
POINT_COLOR: Final[str] = '#59636B'
TEXT_COLOR: Final[str] = '#263238'
MISSING_COLOR: Final[str] = '#F1F3F4'
EMPTY_COLOR: Final[str] = '#E7EEF2'
FAILED_COLOR: Final[str] = '#F6E8E5'
MISSING_EDGE: Final[str] = '#9AA0A6'
EMPTY_EDGE: Final[str] = '#607D8B'
FAILED_EDGE: Final[str] = '#A04432'
BOX_LINE_WIDTH: Final[float] = 0.8
DENSITY_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    'figure6_density', ('#FFFFFF', '#D9E4EA', '#78909C', '#263238'),
)
RESPONSE_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    'figure6_response', ('#315B7D', '#F7F7F5', '#A44A3F'),
)
STATUS_STYLE: Final[Mapping[StageState, Tuple[str, str, str]]] = {
    StageState.MISSING: (MISSING_COLOR, MISSING_EDGE, 'MISSING'),
    StageState.FAILED: (FAILED_COLOR, FAILED_EDGE, 'FAILED'),
    StageState.OBSERVED_EMPTY: (EMPTY_COLOR, EMPTY_EDGE, 'OBSERVED EMPTY'),
}


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 7,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def _style_axis(axis: Axes, crop: Crop) -> None:
    x_min, y_min, x_max, y_max = crop
    axis.set_xlim(y_min, y_max)
    axis.set_ylim(x_min, x_max)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_visible(False)


def _status(axis: Axes, stage: StageEvidence, crop: Crop) -> None:
    background, edge, label = STATUS_STYLE[stage.runtime_state]
    x_min, y_min, x_max, y_max = crop
    axis.set_facecolor(background)
    axis.plot(
        [y_min, y_max, y_max, y_min, y_min], [x_min, x_min, x_max, x_max, x_min],
        color=edge, linewidth=0.7, linestyle=(0, (3, 2)),
    )
    detail = stage.detail.strip()
    text = label if not detail else '%s\n%s' % (label, detail)
    axis.text(
        (y_min + y_max) / 2.0, (x_min + x_max) / 2.0, text,
        ha='center', va='center', color=edge, fontsize=LABEL_SIZE, linespacing=1.35,
    )


def _unavailable(axis: Axes, crop: Crop, detail: str) -> None:
    x_min, y_min, x_max, y_max = crop
    axis.set_facecolor(MISSING_COLOR)
    axis.text(
        (y_min + y_max) / 2.0, (x_min + x_max) / 2.0,
        'SPATIAL DATA UNAVAILABLE\n' + detail, ha='center', va='center',
        color=MISSING_EDGE, fontsize=LABEL_SIZE, linespacing=1.35,
    )


def _draw_density(axis: Axes, stage: StageEvidence, crop: Crop) -> None:
    points = stage.arrays.get('points')
    if points is not None and points.ndim == 2 and points.shape[1] >= 3:
        x_min, y_min, x_max, y_max = crop
        finite = np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
        counts, _, _ = np.histogram2d(
            points[finite, 2], points[finite, 1], bins=160,
            range=((y_min, y_max), (x_min, x_max)),
        )
        axis.imshow(
            np.log1p(counts).T, origin='lower', extent=(y_min, y_max, x_min, x_max),
            cmap=DENSITY_CMAP, interpolation='nearest', rasterized=True,
        )
        return
    density = stage.arrays.get('tta_density_map')
    calibrated = None if density is None else crop_bev_map(
        density, stage.spatial_extent, crop,
    )
    if calibrated is None:
        _unavailable(axis, crop, 'no captured points or calibrated density map')
        return
    axis.imshow(
        calibrated[0], origin='lower', extent=calibrated[1],
        cmap=DENSITY_CMAP, interpolation='nearest', rasterized=True,
    )


def _class_color(label: int) -> Tuple[float, float, float]:
    index = min(max(label - 1, 0), len(CLASS_NAMES) - 1)
    rgb = CLASS_COLORS[index]
    return rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0


def _draw_boxes(axis: Axes, boxes: np.ndarray, labels: np.ndarray,
                scores: Optional[np.ndarray], reliability: Optional[np.ndarray] = None) -> None:
    count = min(len(boxes), len(labels))
    if scores is not None:
        count = min(count, len(scores))
    for index in range(count):
        box = boxes[index]
        if box.size < 7 or not np.isfinite(box[:7]).all() or (box[3:5] <= 0).any():
            continue
        center_x, center_y, length, width, yaw = box[0], box[1], box[3], box[4], box[6]
        local = np.array([
            [length / 2.0, width / 2.0], [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0], [-length / 2.0, width / 2.0],
        ])
        cosine, sine = np.cos(yaw), np.sin(yaw)
        corners = local @ np.array([[cosine, -sine], [sine, cosine]]).T
        corners += np.array([center_x, center_y])
        closed = np.concatenate((corners, corners[:1]), axis=0)
        label = int(abs(labels[index]))
        color = _class_color(label)
        alpha = 0.95
        suffix = ''
        if reliability is not None and index < len(reliability):
            value = float(reliability[index])
            alpha = min(max(value, 0.2), 1.0) if np.isfinite(value) else 0.2
            suffix = ' r%.2f' % value
        axis.plot(
            closed[:, 1], closed[:, 0], color=color, linewidth=BOX_LINE_WIDTH,
            alpha=alpha, solid_capstyle='round', zorder=2,
        )
        class_index = min(max(label - 1, 0), len(CLASS_NAMES) - 1)
        class_name = CLASS_NAMES[class_index]
        score_text = '' if scores is None else ' %.2f' % scores[index]
        axis.text(
            center_y, center_x, '%s%s%s' % (class_name, score_text, suffix),
            fontsize=LABEL_SIZE, color=color, ha='left', va='bottom', clip_on=True,
            zorder=3,
        )


def _prediction_arrays(stage: StageEvidence) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    boxes = stage.arrays.get('pred_boxes')
    labels = stage.arrays.get('pred_labels')
    scores = stage.arrays.get('pred_scores')
    if boxes is None or labels is None or scores is None:
        return None
    return boxes, labels.reshape(-1), scores.reshape(-1)


def _draw_reliability(axis: Axes, stage: StageEvidence, crop: Crop) -> None:
    arrays = _prediction_arrays(stage)
    reliability = stage.arrays.get('spcra_reliability')
    if arrays is None or reliability is None:
        _unavailable(axis, crop, 'prediction or reliability arrays absent')
        return
    _draw_boxes(axis, arrays[0], arrays[1], arrays[2], reliability.reshape(-1))


def _draw_rgplm(axis: Axes, stage: StageEvidence, crop: Crop) -> None:
    pseudo = stage.arrays.get('gt_boxes')
    if pseudo is None or pseudo.ndim != 2 or pseudo.shape[1] < 8:
        _unavailable(axis, crop, 'effective/injection gt_boxes absent')
        return
    labels = np.abs(pseudo[:, 7]).astype(np.int64)
    scores = pseudo[:, 8] if pseudo.shape[1] > 8 else None
    _draw_boxes(axis, pseudo[:, :7], labels, scores)


def _draw_sgdfa(axis: Axes, stage: StageEvidence, crop: Crop) -> None:
    delta = stage.arrays.get('delta')
    calibrated = None if delta is None else crop_bev_map(
        delta, stage.spatial_extent, crop,
    )
    if calibrated is None or not np.isfinite(calibrated[0]).any():
        _unavailable(axis, crop, 'captured delta map lacks finite spatial calibration')
        return
    spatial, extent = calibrated
    limit = float(np.nanpercentile(np.abs(spatial), 98.0))
    limit = max(limit, float(np.finfo(np.float32).eps))
    axis.imshow(
        spatial, origin='lower', extent=extent,
        cmap=RESPONSE_CMAP, vmin=-limit, vmax=limit, interpolation='bilinear',
        rasterized=True,
    )


def _draw_final(axis: Axes, stage: StageEvidence, crop: Crop) -> None:
    arrays = _prediction_arrays(stage)
    if arrays is None:
        _unavailable(axis, crop, 'first-forward final_detection arrays absent')
        return
    _draw_boxes(axis, arrays[0], arrays[1], arrays[2])


DRAWERS: Final[Mapping[str, Callable[[Axes, StageEvidence, Crop], None]]] = {
    'density': _draw_density,
    'reliability': _draw_reliability,
    'rgplm': _draw_rgplm,
    'sgdfa': _draw_sgdfa,
    'finaldet': _draw_final,
}


def draw_stage(axis: Axes, stage: StageEvidence, crop: Crop, title: bool) -> None:
    if stage.runtime_state is StageState.COMPLETE:
        DRAWERS[stage.slug](axis, stage, crop)
    else:
        _status(axis, stage, crop)
    _style_axis(axis, crop)
    axis.set_title(stage.title if title else '', fontsize=TITLE_SIZE, pad=2.0)


def save_stage_png(stage: StageEvidence, crop: Crop, path: Path) -> None:
    figure, axis = plt.subplots(figsize=PANEL_SIZE)
    draw_stage(axis, stage, crop, True)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90)
    figure.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(figure)


def render_plate(rows: Sequence[TokenEvidence]) -> Figure:
    height = 1.18 * len(rows) + 0.35
    figure, axes = plt.subplots(len(rows), 5, figsize=(PLATE_WIDTH, height), squeeze=False)
    for row_index, row in enumerate(rows):
        for column_index, stage in enumerate(row.stages):
            draw_stage(axes[row_index, column_index], stage, row.crop, row_index == 0)
        axes[row_index, 0].text(
            -0.035, 0.5, chr(ord('a') + row_index), transform=axes[row_index, 0].transAxes,
            fontsize=ROW_SIZE, fontweight='bold', color=TEXT_COLOR,
            ha='right', va='center', clip_on=False,
        )
    figure.subplots_adjust(left=0.025, right=0.995, bottom=0.02, top=0.91,
                           hspace=0.06, wspace=0.025)
    return figure


def save_plate(rows: Sequence[TokenEvidence], png: Path, pdf: Path) -> None:
    figure = render_plate(rows)
    figure.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.02)
    figure.savefig(pdf, bbox_inches='tight', pad_inches=0.02)
    plt.close(figure)
