from pathlib import Path
from typing import Final, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
import numpy as np

from pcdet.utils.figure6_schema import StageState

from .contracts import Crop
from .loading import StageEvidence, TokenEvidence
from .render_primitives import (
    RELIABILITY_CMAP, RenderDataError, draw_callouts, draw_context, draw_density, draw_final,
    draw_reliability, draw_rgplm, draw_sgdfa, draw_status, style_axis,
)
from .spatial import crop_bev_map


PLATE_SIZE: Final[Tuple[float, float]] = (7.2, 2.75)
ROW_SIZE: Final[Tuple[float, float]] = (7.2, 1.55)
TITLE_SIZE: Final[float] = 6.5
ROW_LABELS: Final[Tuple[str, str]] = (
    '(a) Far-range recovery', '(b) Small-object recovery',
)
PLATE_LEFT: Final[float] = 0.17
DENSITY_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    'figure6_density', ('#FFFFFF', '#D9E4EA', '#78909C', '#263238'),
)


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'pdf.fonttype': 42,
    'font.size': 7,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def _points(row: TokenEvidence) -> Optional[np.ndarray]:
    return row.stages[0].arrays.get('points')


def pooled_sgdfa_limit(rows: Sequence[TokenEvidence]) -> float:
    values = []
    for row in rows:
        stage = row.stages[3]
        delta = stage.arrays.get('delta')
        calibrated = None if delta is None else crop_bev_map(delta, stage.spatial_extent, row.crop)
        if calibrated is not None:
            finite = calibrated[0][np.isfinite(calibrated[0])]
            if finite.size:
                values.append(np.abs(finite))
    if not values:
        return float(np.finfo(np.float32).eps)
    return max(float(np.percentile(np.concatenate(values), 98.0)),
               float(np.finfo(np.float32).eps))


def draw_stage(axis: Axes, row: TokenEvidence, stage: StageEvidence,
               title: bool, sgdfa_limit: float, annotate_reliability: bool = True) -> None:
    displayed_callouts = row.callouts[:2]
    if stage.runtime_state is not StageState.COMPLETE:
        draw_status(axis, stage, row.crop)
    if stage.runtime_state is StageState.COMPLETE and stage.slug == 'density':
        draw_density(axis, stage, row.crop, DENSITY_CMAP)
    if stage.runtime_state is StageState.COMPLETE and stage.slug == 'reliability':
        draw_reliability(axis, stage, displayed_callouts if annotate_reliability else ())
    if stage.runtime_state is StageState.COMPLETE and stage.slug == 'rgplm':
        draw_rgplm(axis, stage)
    if stage.runtime_state is StageState.COMPLETE and stage.slug == 'sgdfa':
        draw_sgdfa(axis, stage, row.crop, sgdfa_limit)
    if stage.runtime_state is StageState.COMPLETE and stage.slug == 'finaldet':
        draw_final(axis, stage)
    draw_context(axis, _points(row))
    draw_callouts(axis, displayed_callouts)
    style_axis(axis, row.crop)
    axis.set_title(stage.title if title else '', fontsize=TITLE_SIZE, pad=2.0)


def _add_reliability_key(figure: Figure, reliability_axis: Axes) -> None:
    position = reliability_axis.get_position()
    color_axis = figure.add_axes((
        position.x0 + position.width * 0.25, 0.012,
        position.width * 0.5, 0.012,
    ))
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=RELIABILITY_CMAP),
        cax=color_axis, orientation='horizontal', ticks=(0.0, 1.0),
    )
    colorbar.ax.tick_params(
        labelsize=4.5, length=1.5, pad=1.0,
        labeltop=True, labelbottom=False,
    )


def render_plate(rows: Sequence[TokenEvidence], alternate: bool = False,
                 sgdfa_limit: Optional[float] = None) -> Figure:
    size = PLATE_SIZE if len(rows) == 2 else (PLATE_SIZE[0], 1.2 * len(rows) + 0.35)
    figure, axes = plt.subplots(len(rows), 5, figsize=size, squeeze=False)
    limit = pooled_sgdfa_limit(rows) if sgdfa_limit is None else sgdfa_limit
    for row_index, row in enumerate(rows):
        for column_index, stage in enumerate(row.stages):
            draw_stage(
                axes[row_index, column_index], row, stage, row_index == 0, limit,
                annotate_reliability=not alternate,
            )
    figure.subplots_adjust(left=PLATE_LEFT, right=0.995, bottom=0.045, top=0.91,
                           hspace=0.08, wspace=0.025)
    for row_index, row_axes in enumerate(axes):
        position = row_axes[0].get_position()
        label = ROW_LABELS[row_index] if row_index < len(ROW_LABELS) else ''
        figure.text(0.012, (position.y0 + position.y1) / 2.0, label, fontsize=6.0,
                    color='#263238', ha='left', va='center')
    _add_reliability_key(figure, axes[0, 1])
    return figure


def render_row(row: TokenEvidence, sgdfa_limit: float) -> Figure:
    figure, axes = plt.subplots(1, 5, figsize=ROW_SIZE, squeeze=False)
    for column_index, stage in enumerate(row.stages):
        draw_stage(axes[0, column_index], row, stage, True, sgdfa_limit)
    figure.subplots_adjust(left=0.02, right=0.995, bottom=0.04, top=0.86, wspace=0.025)
    return figure


def save_figure(figure: Figure, png: Path, pdf: Optional[Path] = None) -> None:
    figure.savefig(png, dpi=600, facecolor='white')
    if pdf is not None:
        figure.savefig(pdf, facecolor='white')
    plt.close(figure)


def save_plate(rows: Sequence[TokenEvidence], png: Path, pdf: Path,
               alternate: bool = False, sgdfa_limit: Optional[float] = None) -> None:
    save_figure(render_plate(
        rows, alternate=alternate, sgdfa_limit=sgdfa_limit,
    ), png, pdf)


def save_row(row: TokenEvidence, png: Path, sgdfa_limit: float) -> None:
    save_figure(render_row(row, sgdfa_limit), png)


__all__ = [
    'RenderDataError', 'pooled_sgdfa_limit', 'render_plate', 'render_row',
    'save_plate', 'save_row',
]
