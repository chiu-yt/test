from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from tools.figure6_utils.palette import CLASS_COLORS

from .contracts import CaptureBundle, Crop, SelectedRecord


PLATE_SIZE: Final[Tuple[float, float]] = (11.6, 4.4)
PREVIEW_SIZE: Final[Tuple[float, float]] = (11.6, 2.3)
PANEL_TITLES: Final[Tuple[str, ...]] = (
    'Reference', 'View 1', 'View 2', 'View 3', 'View 4', 'Summary / Reliability',
)
ROW_LABELS: Final[Tuple[str, str]] = (
    '(a) Case A\nhigh q / low r',
    '(b) Case B\nhigh q / high r',
)
POINT_COLOR: Final[str] = '#707070'
CONTEXT_COLOR: Final[str] = '#B0BEC5'
TEXT_COLOR: Final[str] = '#263238'
UNMATCHED_COLOR: Final[str] = '#C62828'
PANEL_EDGE_COLOR: Final[str] = '#CFD8DC'
RELIABILITY_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    'figure7_reliability', ('#C62828', '#FDD835', '#2E7D32'),
)
RELIABILITY_NORM: Final[Normalize] = Normalize(0.0, 1.0, clip=True)


@dataclass(frozen=True, slots=True)
class RenderLayout:
    labels: Tuple[str, ...]
    size: Tuple[float, float]
    point_range: Crop


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'pdf.fonttype': 42,
    'font.size': 7,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def _bounded(center: float, half_span: float, lower: float, upper: float) -> Tuple[float, float]:
    width = min(2.0 * half_span, upper - lower)
    start = min(max(center - width / 2.0, lower), upper - width)
    return start, start + width


def crop_for_case(record: SelectedRecord, point_range: Crop) -> Crop:
    box = record.arrays['reference_boxes'][record.candidate.reference_index]
    half_span = max(12.0, 2.0 * float(max(box[3], box[4])))
    x_min, x_max = _bounded(float(box[0]), half_span, point_range[0], point_range[2])
    y_min, y_max = _bounded(float(box[1]), half_span, point_range[1], point_range[3])
    return x_min, y_min, x_max, y_max


def _corners(box: np.ndarray) -> np.ndarray:
    half_x, half_y = box[3] / 2.0, box[4] / 2.0
    local = np.array(((half_x, half_y), (half_x, -half_y),
                      (-half_x, -half_y), (-half_x, half_y)))
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    xy = local @ rotation.T + box[:2]
    return xy[:, (1, 0)]


def _style_axis(axis: Axes, crop: Crop) -> None:
    axis.set_xlim(crop[1], crop[3])
    axis.set_ylim(crop[0], crop[2])
    axis.set_aspect('equal', adjustable='box')
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_box(axis: Axes, box: np.ndarray, color: str | Tuple[float, float, float],
              width: float, alpha: float, layer: int) -> None:
    axis.add_patch(Polygon(
        _corners(box), closed=True, fill=False, edgecolor=color,
        linewidth=width, alpha=alpha, zorder=layer,
    ))


def _panel_data(record: SelectedRecord, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = 'reference' if index == 0 else 'view_%d' % (index - 1)
    points = record.arrays[prefix + '_points']
    boxes = record.arrays[prefix + '_boxes']
    if index > 0:
        matrix = record.arrays[prefix + '_transform']
        points = record.inverse_points(points, matrix)
        finite = np.isfinite(boxes).all(axis=1)
        boxes = np.array(boxes, copy=True)
        boxes[finite] = record.inverse_boxes(boxes[finite], matrix)
    labels = record.arrays[prefix + '_labels']
    return points, boxes, labels


def _point_style(points: np.ndarray, crop: Crop) -> Tuple[float, float]:
    visible = (np.isfinite(points[:, :3]).all(axis=1)
               & (points[:, 0] >= crop[0]) & (points[:, 0] <= crop[2])
               & (points[:, 1] >= crop[1]) & (points[:, 1] <= crop[3]))
    count = max(int(np.count_nonzero(visible)), 1)
    size = float(np.clip(100.0 / np.sqrt(count), .25, 4.0))
    alpha = float(np.clip(1.5 / max(np.log10(count + 1), 1.0), .28, .65))
    return size, alpha


def _draw_bev(axis: Axes, record: SelectedRecord, index: int, crop: Crop) -> None:
    points, boxes, labels = _panel_data(record, index)
    finite = np.isfinite(points[:, :3]).all(axis=1)
    point_size, point_alpha = _point_style(points, crop)
    axis.scatter(points[finite, 1], points[finite, 0], s=point_size, c=POINT_COLOR,
                 alpha=point_alpha, linewidths=0, rasterized=True, zorder=1)
    for box in boxes[np.isfinite(boxes).all(axis=1)]:
        _draw_box(axis, box, CONTEXT_COLOR, .45, .35, 2)
    reference_index = record.candidate.reference_index
    match = reference_index if index == 0 else int(record.arrays['match_indices'][reference_index, index - 1])
    if match >= 0:
        label = int(labels[match])
        raw_rgb = CLASS_COLORS[label - 1]
        rgb = (raw_rgb[0] / 255.0, raw_rgb[1] / 255.0, raw_rgb[2] / 255.0)
        _draw_box(axis, boxes[match], rgb, 1.6, .95, 4)
    quality = 1.0 if index == 0 else float(record.arrays['view_quality'][reference_index, index - 1])
    state = 'selected' if index == 0 else ('matched' if match >= 0 else 'unmatched')
    mask_key = 'reference_mask' if index == 0 else 'view_%d_mask' % (index - 1)
    rescue_key = 'reference_rescue_mask' if index == 0 else 'view_%d_rescue_mask' % (index - 1)
    accepted = match >= 0 and bool(record.arrays[mask_key][match])
    rescued = match >= 0 and bool(record.arrays[rescue_key][match])
    axis.text(.03, .97, '%s  quality=%.3f\naccepted=%s  rescued=%s' % (
        state, quality, str(accepted).lower(), str(rescued).lower()),
        transform=axis.transAxes, ha='left', va='top', fontsize=6.0,
        color=UNMATCHED_COLOR if match < 0 else TEXT_COLOR,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .78, 'pad': 1.2}, zorder=6)
    _style_axis(axis, crop)


def _draw_summary(axis: Axes, record: SelectedRecord) -> None:
    reliability = record.candidate.reliability
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis('off')
    axis.add_patch(Rectangle((.01, .01), .98, .98, fill=False, edgecolor=PANEL_EDGE_COLOR,
                             linewidth=.6, zorder=0))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    axis.imshow(gradient, extent=(.10, .90, .15, .23), aspect='auto',
                cmap=RELIABILITY_CMAP, norm=RELIABILITY_NORM, zorder=1)
    axis.add_patch(Rectangle((.10, .15), .80, .08, fill=False, edgecolor=TEXT_COLOR,
                             linewidth=.5, zorder=2))
    marker = .10 + .80 * reliability
    axis.plot((marker, marker), (.12, .26), color=TEXT_COLOR, linewidth=1.0, zorder=3)
    axis.text(.5, .90, '%s  proposal %d' % (
        record.candidate.class_name, record.candidate.reference_index),
        ha='center', va='top', fontsize=6.8, color=TEXT_COLOR, weight='bold')
    axis.text(.5, .72, 'confidence q = %.3f\nreliability r = %.3f' % (
        record.candidate.score, reliability), ha='center', va='top', fontsize=6.6,
        color=TEXT_COLOR, linespacing=1.5)
    axis.text(.5, .48, 'qualities  ' + ' / '.join('%.2f' % value
              for value in record.arrays['view_quality'][record.candidate.reference_index]),
              ha='center', va='top', fontsize=5.8,
              color=TEXT_COLOR)
    axis.text(.10, .10, '0', ha='center', va='top', fontsize=4.8, color=TEXT_COLOR)
    axis.text(.90, .10, '1', ha='center', va='top', fontsize=4.8, color=TEXT_COLOR)


def _render_rows(rows: Sequence[SelectedRecord], layout: RenderLayout) -> Figure:
    figure, axes = plt.subplots(len(rows), 6, figsize=layout.size, squeeze=False)
    for row_index, record in enumerate(rows):
        crop = crop_for_case(record, layout.point_range)
        for panel_index in range(5):
            _draw_bev(axes[row_index, panel_index], record, panel_index, crop)
        _draw_summary(axes[row_index, 5], record)
        axes[row_index, 0].set_ylabel('forward x', fontsize=5.0)
        axes[row_index, 0].set_xlabel('lateral y', fontsize=5.0)
        position = axes[row_index, 0].get_position()
        figure.text(.008, (position.y0 + position.y1) / 2.0, layout.labels[row_index],
                    ha='left', va='center', fontsize=6.0, color=TEXT_COLOR, linespacing=1.4)
    for index, title in enumerate(PANEL_TITLES):
        axes[0, index].set_title(title, fontsize=7.0, pad=2.0)
    figure.subplots_adjust(left=.105, right=.995, bottom=.07, top=.91,
                           hspace=.10, wspace=.025)
    return figure


def render_plate(bundle: CaptureBundle, point_range: Crop) -> Figure:
    return _render_rows(
        (bundle.case_a, bundle.case_b), RenderLayout(ROW_LABELS, PLATE_SIZE, point_range),
    )


def render_preview(record: SelectedRecord, point_range: Crop) -> Figure:
    label = 'Case A candidate' if record.candidate.pool == 'variable' else 'Case B candidate'
    return _render_rows((record,), RenderLayout((label,), PREVIEW_SIZE, point_range))


def save_figure(figure: Figure, png: Path, pdf: Path | None = None) -> None:
    try:
        figure.savefig(png, dpi=600, facecolor='white')
        if pdf is not None:
            figure.savefig(pdf, facecolor='white')
    finally:
        plt.close(figure)
