from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

from .domain import Callout, Detection, FigureColumn, FrameRecord
from .palette import NUSCENES_DETECTION_PALETTE
from .policies import RenderPolicy


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif',
]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

STANDARD_DPI: Final[int] = 150
HD_DPI: Final[int] = 300
PANEL_SIZE_INCHES: Final[Tuple[float, float]] = (12.0, 3.2)
CONTACT_SHEET_PAGE_SIZE: Final[Tuple[float, float]] = (12.0, 24.0)
CONTACT_SHEET_CANDIDATES: Final[int] = 12
BEV_RANGE_METERS: Final[Tuple[float, float]] = (-50.0, 50.0)
POINT_COLOR: Final[str] = '#B8B8B8'
CALLOUT_COLOR: Final[str] = '#CC0000'


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class PanelSpec:
    frame: FrameRecord
    lidar_points: np.ndarray
    rank: int
    callouts: Tuple[Callout, ...] = ()


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class PanelOutputs:
    standard: Path
    hd_clean: Path
    hd_callout: Path


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class _PanelRender:
    panel: PanelSpec
    policy: RenderPolicy
    show_callouts: bool


def _box_bev_corners(detection: Detection) -> np.ndarray:
    center_x, center_y, _ = detection.box.center
    length, width, _ = detection.box.size
    local = np.array([
        [length / 2.0, width / 2.0],
        [length / 2.0, -width / 2.0],
        [-length / 2.0, -width / 2.0],
        [-length / 2.0, width / 2.0],
    ])
    cosine = np.cos(detection.box.yaw)
    sine = np.sin(detection.box.yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.array([center_x, center_y])


def _draw_detection(axis: Axes, detection: Detection) -> None:
    corners = _box_bev_corners(detection)
    closed = np.concatenate((corners, corners[:1]), axis=0)
    red, green, blue = NUSCENES_DETECTION_PALETTE[detection.class_name]
    color = (red / 255.0, green / 255.0, blue / 255.0)
    axis.plot(
        closed[:, 1],
        closed[:, 0],
        color=color,
        linewidth=1.1,
        solid_capstyle='round',
        zorder=2,
    )


def _draw_callout(axis: Axes, callout: Callout) -> None:
    forward_x, lateral_y, forward_max, lateral_max = callout.roi
    axis.add_patch(Rectangle(
        (lateral_y, forward_x),
        lateral_max - lateral_y,
        forward_max - forward_x,
        fill=False,
        edgecolor=CALLOUT_COLOR,
        linewidth=1.2,
        zorder=3,
    ))


def _style_axis(axis: Axes, column: FigureColumn) -> None:
    axis.set_title(column.value, fontsize=7, fontweight='normal', pad=3)
    axis.set_xlim(BEV_RANGE_METERS)
    axis.set_ylim(BEV_RANGE_METERS)
    axis.set_aspect('equal', adjustable='box')
    axis.set_facecolor('white')
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_panel(
    axes: Sequence[Axes],
    render: _PanelRender,
) -> None:
    panel = render.panel
    point_x = panel.lidar_points[:, 1]
    point_y = panel.lidar_points[:, 0]
    for axis, column in zip(axes, render.policy.columns):
        axis.scatter(
            point_x,
            point_y,
            s=0.35,
            c=POINT_COLOR,
            alpha=0.55,
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )
        for detection in panel.frame.detections(column):
            _draw_detection(axis, detection)
        if render.show_callouts:
            for callout in panel.callouts:
                _draw_callout(axis, callout)
        _style_axis(axis, column)


def render_comparison_panel(
    panel: PanelSpec,
    show_callouts: bool = False,
    policy: RenderPolicy = RenderPolicy(),
) -> Figure:
    figure, axes = plt.subplots(1, 4, figsize=PANEL_SIZE_INCHES)
    figure.patch.set_facecolor('white')
    figure.subplots_adjust(left=0.015, right=0.995, bottom=0.02, top=0.86, wspace=0.03)
    _draw_panel(tuple(axes), _PanelRender(panel, policy, show_callouts))
    figure.text(
        0.015,
        0.965,
        'Rank %02d | %s' % (panel.rank, panel.frame.sample_token),
        ha='left',
        va='top',
        fontsize=6.5,
        color='#333333',
    )
    return figure


def _save_and_close(figure: Figure, path: Path, dpi: int) -> Path:
    try:
        figure.savefig(path, dpi=dpi, facecolor='white')
    finally:
        plt.close(figure)
    return path


def save_panel_pngs(panel: PanelSpec, outputs: PanelOutputs) -> Tuple[Path, Path, Path]:
    standard = _save_and_close(
        render_comparison_panel(panel, show_callouts=False),
        outputs.standard,
        STANDARD_DPI,
    )
    hd_clean = _save_and_close(
        render_comparison_panel(panel, show_callouts=False),
        outputs.hd_clean,
        HD_DPI,
    )
    hd_callout = _save_and_close(
        render_comparison_panel(panel, show_callouts=True),
        outputs.hd_callout,
        HD_DPI,
    )
    return standard, hd_clean, hd_callout


def _render_contact_page(panels: Sequence[PanelSpec], policy: RenderPolicy) -> Figure:
    figure, axes = plt.subplots(
        CONTACT_SHEET_CANDIDATES,
        4,
        figsize=CONTACT_SHEET_PAGE_SIZE,
        squeeze=False,
    )
    figure.patch.set_facecolor('white')
    figure.subplots_adjust(left=0.015, right=0.995, bottom=0.01, top=0.975,
                           hspace=0.36, wspace=0.03)
    for row, panel in enumerate(panels):
        row_axes = tuple(axes[row])
        _draw_panel(row_axes, _PanelRender(panel, policy, False))
        row_axes[0].text(
            0.0,
            1.08,
            'Rank %02d | %s' % (panel.rank, panel.frame.sample_token),
            transform=row_axes[0].transAxes,
            ha='left',
            va='bottom',
            fontsize=5.5,
            color='#333333',
        )
    for row in range(len(panels), CONTACT_SHEET_CANDIDATES):
        for axis in axes[row]:
            axis.set_axis_off()
    return figure


def render_contact_sheets(
    panels: Sequence[PanelSpec],
    policy: RenderPolicy = RenderPolicy(),
) -> Tuple[Figure, ...]:
    return tuple(
        _render_contact_page(
            panels[start:start + CONTACT_SHEET_CANDIDATES],
            policy,
        )
        for start in range(0, len(panels), CONTACT_SHEET_CANDIDATES)
    )


def save_contact_sheets(
    panels: Sequence[PanelSpec],
    output_stem: Path,
    policy: RenderPolicy = RenderPolicy(),
) -> Tuple[Path, ...]:
    saved = []
    for page_number, figure in enumerate(render_contact_sheets(panels, policy), start=1):
        output_path = output_stem.with_name(
            '%s-page-%02d.png' % (output_stem.name, page_number)
        )
        saved.append(_save_and_close(figure, output_path, STANDARD_DPI))
    return tuple(saved)
