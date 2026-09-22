from dataclasses import dataclass
from typing import Final, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import numpy as np

from .domain import FigureColumn, ImageRoi
from .final_selection import FinalRowSelection
from .rendering import POINT_COLOR, _box_bev_corners, _draw_callout, _draw_detection


FINAL_DPI: Final[int] = 600
FINAL_PLATE_SIZE: Final[Tuple[float, float]] = (7.2, 5.45)
HORIZONTAL_PLATE_SIZE: Final[Tuple[float, float]] = (7.2, 3.6)
FINAL_ROW_SIZE: Final[Tuple[float, float]] = (7.2, 1.95)
FIGURE6_SIZE: Final[Tuple[float, float]] = (3.45, 3.35)
TITLE_SIZE: Final[float] = 7.0
FIGURE6_TITLE_SIZE: Final[float] = 7.0
POINT_SIZE: Final[float] = 0.25
POINT_ALPHA: Final[float] = 0.58
HORIZONTAL_POINT_SIZE: Final[float] = 0.65
HORIZONTAL_POINT_ALPHA: Final[float] = 0.85
HORIZONTAL_POINT_COLOR: Final[str] = '#707070'
HORIZONTAL_DATA_RATIO: Final[float] = 1.8
HORIZONTAL_CONTEXT_M: Final[float] = 4.0
HORIZONTAL_TITLE_PAD: Final[float] = 1.5
DENSITY_BINS: Final[int] = 160
DENSITY_COLORS: Final[Tuple[str, ...]] = ('#FFFFFF', '#D9E4EA', '#78909C', '#263238')
PLATE_LAYOUT: Final[Tuple[float, float, float, float, float, float]] = (
    0.012, 0.995, 0.012, 0.94, 0.035, 0.025,
)
HORIZONTAL_PLATE_LAYOUT: Final[Tuple[float, float, float, float, float, float]] = (
    0.012, 0.995, 0.015, 0.955, 0.03, 0.02,
)
ROW_LAYOUT: Final[Tuple[float, float, float, float, float]] = (
    0.012, 0.995, 0.025, 0.84, 0.025,
)
FIGURE6_LAYOUT: Final[Tuple[float, float, float, float]] = (0.02, 0.98, 0.02, 0.92)
DENSITY_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    'sparse_density', DENSITY_COLORS,
)


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif',
]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class FinalRenderRow:
    selection: FinalRowSelection
    lidar_points: np.ndarray


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class _PointStyle:
    size: float
    alpha: float
    color: str


_LEGACY_POINT_STYLE: Final[_PointStyle] = _PointStyle(
    POINT_SIZE, POINT_ALPHA, POINT_COLOR,
)
_HORIZONTAL_POINT_STYLE: Final[_PointStyle] = _PointStyle(
    HORIZONTAL_POINT_SIZE, HORIZONTAL_POINT_ALPHA, HORIZONTAL_POINT_COLOR,
)


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class RenderRowCountError(ValueError):
    actual: int

    def __str__(self) -> str:
        return 'final Figure 5 requires exactly three rows; received %d' % self.actual


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class _RowRender:
    row: FinalRenderRow
    crop: ImageRoi
    titles: bool
    show_callouts: bool
    title_pad: float
    point_style: _PointStyle = _LEGACY_POINT_STYLE


def _style_axis(axis: Axes, crop: ImageRoi) -> None:
    x_min, y_min, x_max, y_max = crop
    axis.set_xlim(y_min, y_max)
    axis.set_ylim(x_min, x_max)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_points(
        axis: Axes, points: np.ndarray,
        style: _PointStyle = _LEGACY_POINT_STYLE,
) -> None:
    axis.scatter(
        points[:, 1], points[:, 0], s=style.size, c=style.color,
        alpha=style.alpha, linewidths=0.0, rasterized=True, zorder=1,
    )


def _draw_row(axes: Sequence[Axes], render: _RowRender) -> None:
    row = render.row
    for axis, column in zip(axes, tuple(FigureColumn)):
        _draw_points(axis, row.lidar_points, render.point_style)
        for detection in row.selection.frame.detections(column):
            _draw_detection(axis, detection)
        if render.show_callouts:
            for callout in row.selection.callouts:
                _draw_callout(axis, callout)
        _style_axis(axis, render.crop)
        axis.set_title(column.value if render.titles else '', fontsize=TITLE_SIZE,
                       fontweight='normal', pad=render.title_pad)


def _expand_to_horizontal_ratio(crop: ImageRoi) -> ImageRoi:
    x_min, y_min, x_max, y_max = crop
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    x_span = x_max - x_min
    y_span = y_max - y_min
    if y_span / x_span < HORIZONTAL_DATA_RATIO:
        y_span = x_span * HORIZONTAL_DATA_RATIO
    else:
        x_span = y_span / HORIZONTAL_DATA_RATIO
    return (
        x_center - x_span / 2.0,
        y_center - y_span / 2.0,
        x_center + x_span / 2.0,
        y_center + y_span / 2.0,
    )


def _corners_intersect_crop(corners: np.ndarray, crop: ImageRoi) -> bool:
    x_min, y_min, x_max, y_max = crop
    crop_corners = np.array([
        [x_min, y_min], [x_min, y_max],
        [x_max, y_max], [x_max, y_min],
    ])
    edges = np.roll(corners, -1, axis=0) - corners
    axes = np.concatenate((
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.column_stack((-edges[:, 1], edges[:, 0])),
    ))
    for axis in axes:
        box_projection = corners @ axis
        crop_projection = crop_corners @ axis
        if (box_projection.max() < crop_projection.min() or
                crop_projection.max() < box_projection.min()):
            return False
    return True


def horizontal_crop(row: FinalRenderRow) -> ImageRoi:
    extents = []
    for column in tuple(FigureColumn):
        for detection in row.selection.frame.detections(column):
            corners = _box_bev_corners(detection)
            if _corners_intersect_crop(corners, row.selection.crop):
                extents.append(corners)
    extents.extend(np.array([
        [callout.roi[0], callout.roi[1]],
        [callout.roi[2], callout.roi[3]],
    ]) for callout in row.selection.callouts)
    if not extents:
        return _expand_to_horizontal_ratio(row.selection.crop)
    points = np.concatenate(tuple(extents), axis=0)
    return _expand_to_horizontal_ratio((
        float(points[:, 0].min()) - HORIZONTAL_CONTEXT_M,
        float(points[:, 1].min()) - HORIZONTAL_CONTEXT_M,
        float(points[:, 0].max()) + HORIZONTAL_CONTEXT_M,
        float(points[:, 1].max()) + HORIZONTAL_CONTEXT_M,
    ))


def render_final_plate(rows: Sequence[FinalRenderRow], show_callouts: bool) -> Figure:
    if len(rows) != 3:
        raise RenderRowCountError(len(rows))
    figure, axes = plt.subplots(3, 4, figsize=FINAL_PLATE_SIZE, squeeze=False)
    left, right, bottom, top, height_space, width_space = PLATE_LAYOUT
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                           hspace=height_space, wspace=width_space)
    for row_index, row in enumerate(rows):
        _draw_row(tuple(axes[row_index]), _RowRender(
            row, row.selection.crop, row_index == 0, show_callouts, 3.0,
        ))
    return figure


def render_horizontal_plate(rows: Sequence[FinalRenderRow], show_callouts: bool) -> Figure:
    if len(rows) != 3:
        raise RenderRowCountError(len(rows))
    figure, axes = plt.subplots(3, 4, figsize=HORIZONTAL_PLATE_SIZE, squeeze=False)
    left, right, bottom, top, height_space, width_space = HORIZONTAL_PLATE_LAYOUT
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                           hspace=height_space, wspace=width_space)
    for row_index, row in enumerate(rows):
        _draw_row(tuple(axes[row_index]), _RowRender(
            row, horizontal_crop(row), row_index == 0, show_callouts,
            HORIZONTAL_TITLE_PAD, _HORIZONTAL_POINT_STYLE,
        ))
    return figure


def render_final_row(row: FinalRenderRow) -> Figure:
    figure, axes = plt.subplots(1, 4, figsize=FINAL_ROW_SIZE)
    left, right, bottom, top, width_space = ROW_LAYOUT
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=width_space)
    _draw_row(tuple(axes), _RowRender(row, row.selection.crop, True, True, 3.0))
    return figure


def render_density(row: FinalRenderRow) -> Figure:
    figure, axis = plt.subplots(figsize=FIGURE6_SIZE)
    x_min, y_min, x_max, y_max = row.selection.crop
    horizontal = row.lidar_points[:, 1]
    vertical = row.lidar_points[:, 0]
    counts, _, _ = np.histogram2d(
        horizontal, vertical, bins=DENSITY_BINS,
        range=((y_min, y_max), (x_min, x_max)),
    )
    axis.imshow(
        np.log1p(counts).T, origin='lower', extent=(y_min, y_max, x_min, x_max),
        cmap=DENSITY_CMAP, interpolation='nearest', rasterized=True,
    )
    _style_axis(axis, row.selection.crop)
    axis.set_title('Offline sparse-point density', fontsize=FIGURE6_TITLE_SIZE, pad=4.0)
    left, right, bottom, top = FIGURE6_LAYOUT
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return figure


def render_final_detections(row: FinalRenderRow) -> Figure:
    figure, axis = plt.subplots(figsize=FIGURE6_SIZE)
    _draw_points(axis, row.lidar_points)
    for detection in row.selection.frame.refuse_tta:
        _draw_detection(axis, detection)
    _style_axis(axis, row.selection.crop)
    axis.set_title('ReFuse final detections', fontsize=FIGURE6_TITLE_SIZE, pad=4.0)
    left, right, bottom, top = FIGURE6_LAYOUT
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return figure
