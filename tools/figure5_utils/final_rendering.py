from dataclasses import dataclass
from typing import Final, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import numpy as np

from .domain import FigureColumn
from .final_selection import FinalRowSelection
from .rendering import POINT_COLOR, _draw_callout, _draw_detection


FINAL_DPI: Final[int] = 600
FINAL_PLATE_SIZE: Final[Tuple[float, float]] = (7.2, 5.45)
FINAL_ROW_SIZE: Final[Tuple[float, float]] = (7.2, 1.95)
FIGURE6_SIZE: Final[Tuple[float, float]] = (3.45, 3.35)
TITLE_SIZE: Final[float] = 7.0
FIGURE6_TITLE_SIZE: Final[float] = 7.0
POINT_SIZE: Final[float] = 0.25
POINT_ALPHA: Final[float] = 0.58
DENSITY_BINS: Final[int] = 160
DENSITY_COLORS: Final[Tuple[str, ...]] = ('#FFFFFF', '#D9E4EA', '#78909C', '#263238')
PLATE_LAYOUT: Final[Tuple[float, float, float, float, float, float]] = (
    0.012, 0.995, 0.012, 0.94, 0.035, 0.025,
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
class RenderRowCountError(ValueError):
    actual: int

    def __str__(self) -> str:
        return 'final Figure 5 requires exactly three rows; received %d' % self.actual


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class _RowRender:
    row: FinalRenderRow
    titles: bool
    show_callouts: bool


def _style_axis(axis: Axes, row: FinalRowSelection) -> None:
    x_min, y_min, x_max, y_max = row.crop
    axis.set_xlim(y_min, y_max)
    axis.set_ylim(x_min, x_max)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_points(axis: Axes, points: np.ndarray) -> None:
    axis.scatter(
        points[:, 1], points[:, 0], s=POINT_SIZE, c=POINT_COLOR,
        alpha=POINT_ALPHA, linewidths=0.0, rasterized=True, zorder=1,
    )


def _draw_row(axes: Sequence[Axes], render: _RowRender) -> None:
    row = render.row
    for axis, column in zip(axes, tuple(FigureColumn)):
        _draw_points(axis, row.lidar_points)
        for detection in row.selection.frame.detections(column):
            _draw_detection(axis, detection)
        if render.show_callouts:
            for callout in row.selection.callouts:
                _draw_callout(axis, callout)
        _style_axis(axis, row.selection)
        axis.set_title(column.value if render.titles else '', fontsize=TITLE_SIZE,
                       fontweight='normal', pad=3.0)


def render_final_plate(rows: Sequence[FinalRenderRow], show_callouts: bool) -> Figure:
    if len(rows) != 3:
        raise RenderRowCountError(len(rows))
    figure, axes = plt.subplots(3, 4, figsize=FINAL_PLATE_SIZE, squeeze=False)
    left, right, bottom, top, height_space, width_space = PLATE_LAYOUT
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                           hspace=height_space, wspace=width_space)
    for row_index, row in enumerate(rows):
        _draw_row(tuple(axes[row_index]), _RowRender(row, row_index == 0, show_callouts))
    return figure


def render_final_row(row: FinalRenderRow) -> Figure:
    figure, axes = plt.subplots(1, 4, figsize=FINAL_ROW_SIZE)
    left, right, bottom, top, width_space = ROW_LAYOUT
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=width_space)
    _draw_row(tuple(axes), _RowRender(row, True, True))
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
    _style_axis(axis, row.selection)
    axis.set_title('Offline sparse-point density', fontsize=FIGURE6_TITLE_SIZE, pad=4.0)
    left, right, bottom, top = FIGURE6_LAYOUT
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return figure


def render_final_detections(row: FinalRenderRow) -> Figure:
    figure, axis = plt.subplots(figsize=FIGURE6_SIZE)
    _draw_points(axis, row.lidar_points)
    for detection in row.selection.frame.refuse_tta:
        _draw_detection(axis, detection)
    _style_axis(axis, row.selection)
    axis.set_title('ReFuse final detections', fontsize=FIGURE6_TITLE_SIZE, pad=4.0)
    left, right, bottom, top = FIGURE6_LAYOUT
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return figure
