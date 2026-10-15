#!/usr/bin/env python3
# noqa: SIZE_OK - the requested deliverable is one self-contained generator file.
"""Generate Figure 8 from KITTI Tracking ground truth.

Run from ``tools/`` with ``python generate_figure8_gt.py --kitti-root ROOT
--output-dir OUTPUT``. ROOT must contain ``label_2/*.txt``, ``calib/*.txt``,
``image_2/<sequence>/*.png``, and ``velodyne/<sequence>/*.bin``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from PIL import Image

from visual_utils.kitti_figure8_geometry import transform_3dbox_to_image, transform_3dbox_to_pointcloud


VEHICLE_CLASSES = frozenset(("Car", "Van", "Truck"))
PEDESTRIAN_CLASSES = frozenset(("Pedestrian", "Person_sitting"))
CYCLIST_CLASSES = frozenset(("Cyclist",))
VEHICLE_RGB = (37, 99, 184)
PEDESTRIAN_RGB = (205, 48, 54)
CYCLIST_RGB = (238, 183, 32)
OTHER_RGB = (112, 118, 124)
POINT_RGB = (92, 100, 108)
TEXT_RGB = (35, 40, 45)
GRID_RGB = (215, 219, 223)
FIGURE_SIZE = (15.0, 11.25)
FIGURE_DPI = 180
CAMERA_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7), (0, 5), (1, 4))
BEV_FOOTPRINT = (4, 5, 6, 7, 4)
THEME_TITLES = ("Dense vehicles", "Pedestrian / Cyclist", "Far-range sparse observations")
ROW_LABELS = ("Camera t", "BEV t", "Camera t+delta", "BEV t+delta")
ARTIFACT_NAMES = ("figure8_kitti_gt.png", "figure8_kitti_gt.pdf", "figure8_kitti_gt_selection.json")


class Figure8Error(Exception):
    pass


@dataclass(frozen=True)  # noqa: SLOTS_OK - slots=True requires Python 3.10.
class MissingThemeError(Figure8Error):
    theme: str

    def __str__(self) -> str:
        display = "VRU" if self.theme == "vru" else self.theme
        return f"no eligible {display} frame pair found"


@dataclass(frozen=True)  # noqa: SLOTS_OK - slots=True requires Python 3.10.
class DatasetFormatError(Figure8Error):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"invalid KITTI data at {self.path}: {self.reason}"


@dataclass(frozen=True)  # noqa: SLOTS_OK - slots=True requires Python 3.10.
class OutputDirectoryError(Figure8Error):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"output directory {self.path}: {self.reason}"


@dataclass(frozen=True)  # noqa: SLOTS_OK - slots=True requires Python 3.10.
class GroundTruth:
    track_id: int
    object_type: str
    dimensions: Tuple[float, float, float]
    location: Tuple[float, float, float]
    rotation: float


@dataclass(frozen=True)  # noqa: SLOTS_OK - slots=True requires Python 3.10.
class FrameData:
    sequence: str
    frame: int
    labels: Tuple[GroundTruth, ...]
    image_path: Path
    cloud_path: Path


@dataclass(frozen=True)  # noqa: SLOTS_OK - slots=True requires Python 3.10.
class SelectionMetrics:
    preferred_class: Optional[str]
    frame_t_relevant_count: int
    frame_t_plus_delta_relevant_count: int
    shared_relevant_count: int
    dense_min_vehicle_count: int
    shared_min_range_m: float


@dataclass(frozen=True)  # noqa: SLOTS_OK - slots=True requires Python 3.10.
class PairSelection:
    theme: str
    sequence: str
    frame_t: int
    frame_t_plus_delta: int
    delta: int
    shared_track_ids: Tuple[int, ...]
    metrics: SelectionMetrics

    @property
    def identity(self) -> Tuple[str, int, int]:
        return self.sequence, self.frame_t, self.frame_t_plus_delta


def parse_tracking_labels(path: Path) -> Dict[int, Tuple[GroundTruth, ...]]:
    grouped: Dict[int, List[GroundTruth]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 17:
            raise DatasetFormatError(path, f"line {line_number} has {len(fields)} fields, expected 17")
        try:
            frame, track_id = int(fields[0]), int(fields[1])
            numeric = tuple(float(value) for value in fields[3:])
        except ValueError as error:
            raise DatasetFormatError(path, f"line {line_number} contains a non-numeric field") from error
        if fields[2] == "DontCare":
            continue
        grouped.setdefault(frame, []).append(GroundTruth(
            track_id, fields[2], (numeric[7], numeric[8], numeric[9]),
            (numeric[10], numeric[11], numeric[12]), numeric[13]))
    return {frame: tuple(labels) for frame, labels in grouped.items()}


def parse_calibration(path: Path) -> Dict[str, np.ndarray]:
    calibration: Dict[str, np.ndarray] = {}
    matrix_specs = {"P2": ("P2", 3, 4), "Rect": ("Rect", 3, 3), "R_rect": ("Rect", 3, 3),
                    "R0_rect": ("Rect", 3, 3), "Tr_velo_cam": ("Tr_velo_cam", 3, 4),
                    "Tr_velo_to_cam": ("Tr_velo_cam", 3, 4)}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        key = fields[0].rstrip(":")
        if key not in matrix_specs:
            continue
        canonical_key, rows, columns = matrix_specs[key]
        try:
            values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        except ValueError as error:
            raise DatasetFormatError(path, f"line {line_number} contains a non-numeric value") from error
        expected_size = rows * columns
        if values.size != expected_size:
            raise DatasetFormatError(path, f"{key} has {values.size} values, expected {expected_size}")
        calibration[canonical_key] = values.reshape(rows, columns)
    for key, reason in (("P2", "missing P2 calibration"), ("Rect", "missing rectification calibration"),
                        ("Tr_velo_cam", "missing Velodyne-to-camera calibration")):
        if key not in calibration:
            raise DatasetFormatError(path, reason)
    return calibration


def load_frames(kitti_root: Path) -> Tuple[FrameData, ...]:
    labels_root = kitti_root / "label_2"
    if not labels_root.is_dir():
        raise DatasetFormatError(labels_root, "missing label_2 directory")
    frames: List[FrameData] = []
    for label_path in sorted(labels_root.glob("*.txt")):
        sequence = label_path.stem
        image_root, cloud_root = kitti_root / "image_2" / sequence, kitti_root / "velodyne" / sequence
        calibration_path = kitti_root / "calib" / f"{sequence}.txt"
        if not image_root.is_dir() or not cloud_root.is_dir():
            continue
        parse_calibration(calibration_path)
        labels = parse_tracking_labels(label_path)
        images = {path.stem: path for path in image_root.glob("*.png")}
        clouds = {path.stem: path for path in cloud_root.glob("*.bin")}
        for stem in sorted(images.keys() & clouds.keys()):
            try:
                frame = int(stem)
            except ValueError as error:
                raise DatasetFormatError(images[stem], "frame stem is not an integer") from error
            frames.append(FrameData(sequence, frame, labels.get(frame, ()), images[stem], clouds[stem]))
    return tuple(frames)


def _objects_by_id(labels: Iterable[GroundTruth], classes: Optional[frozenset]) -> Dict[int, GroundTruth]:
    if classes is None:
        return {item.track_id: item for item in labels}
    return {item.track_id: item for item in labels if item.object_type in classes}


def _frame_pairs(frames: Sequence[FrameData], max_delta: int) -> Iterable[Tuple[FrameData, FrameData]]:
    grouped: Dict[str, List[FrameData]] = {}
    for frame in frames:
        grouped.setdefault(frame.sequence, []).append(frame)
    for sequence in sorted(grouped):
        ordered = sorted(grouped[sequence], key=lambda item: item.frame)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]:
                delta = second.frame - first.frame
                if delta > max_delta:
                    break
                if delta >= 1:
                    yield first, second


def _candidate_sets(frames: Sequence[FrameData], max_delta: int, far_range_m: float):
    dense, cyclists, pedestrians, far = [], [], [], []
    for first, second in _frame_pairs(frames, max_delta):
        delta = second.frame - first.frame
        first_vehicles, second_vehicles = _objects_by_id(first.labels, VEHICLE_CLASSES), _objects_by_id(second.labels, VEHICLE_CLASSES)
        shared_vehicles = tuple(sorted(first_vehicles.keys() & second_vehicles.keys()))
        if len(first_vehicles) >= 3 and len(second_vehicles) >= 3 and shared_vehicles:
            dense.append(PairSelection("dense", first.sequence, first.frame, second.frame, delta, shared_vehicles,
                SelectionMetrics(None, len(first_vehicles), len(second_vehicles), len(shared_vehicles), min(len(first_vehicles), len(second_vehicles)), 0.0)))
        for classes, preferred, destination in ((CYCLIST_CLASSES, "Cyclist", cyclists), (PEDESTRIAN_CLASSES, "Pedestrian/Person_sitting", pedestrians)):
            first_vru, second_vru = _objects_by_id(first.labels, classes), _objects_by_id(second.labels, classes)
            shared_vru = tuple(sorted(first_vru.keys() & second_vru.keys()))
            if shared_vru:
                destination.append(PairSelection("vru", first.sequence, first.frame, second.frame, delta, shared_vru,
                    SelectionMetrics(preferred, len(first_vru), len(second_vru), len(shared_vru), 0, 0.0)))
        first_all, second_all = _objects_by_id(first.labels, None), _objects_by_id(second.labels, None)
        shared_far = []
        for track_id in sorted(first_all.keys() & second_all.keys()):
            first_location, second_location = first_all[track_id].location, second_all[track_id].location
            first_range, second_range = math.hypot(first_location[0], first_location[2]), math.hypot(second_location[0], second_location[2])
            if first_range >= far_range_m and second_range >= far_range_m:
                shared_far.append((track_id, min(first_range, second_range)))
        if shared_far:
            shared_ids = tuple(item[0] for item in shared_far)
            far.append(PairSelection("far", first.sequence, first.frame, second.frame, delta, shared_ids,
                SelectionMetrics(None, len(shared_far), len(shared_far), len(shared_far), 0, max(item[1] for item in shared_far))))
    dense.sort(key=lambda item: (-item.metrics.dense_min_vehicle_count, -item.metrics.shared_relevant_count, item.delta, item.sequence, item.frame_t, item.frame_t_plus_delta, item.shared_track_ids))
    vru_key = lambda item: (-item.metrics.shared_relevant_count, -min(item.metrics.frame_t_relevant_count, item.metrics.frame_t_plus_delta_relevant_count), -max(item.metrics.frame_t_relevant_count, item.metrics.frame_t_plus_delta_relevant_count), item.sequence, item.frame_t, item.frame_t_plus_delta, item.delta, item.shared_track_ids)
    cyclists.sort(key=vru_key); pedestrians.sort(key=vru_key)
    far.sort(key=lambda item: (-item.metrics.shared_min_range_m, -item.metrics.shared_relevant_count, item.sequence, item.frame_t, item.frame_t_plus_delta, item.delta, item.shared_track_ids))
    return dense, cyclists, pedestrians, far


def _choose_distinct(candidates):
    dense, vru, far = candidates
    for dense_item in dense:
        for vru_item in vru:
            if vru_item.identity == dense_item.identity:
                continue
            far_item = next((item for item in far if item.identity not in (dense_item.identity, vru_item.identity)), None)
            if far_item is not None:
                return dense_item, vru_item, far_item
    selected = [dense[0], vru[0], far[0]]
    if len({item.identity for item in selected}) == 1:
        for theme_index in (2, 1, 0):
            alternative = next((item for item in candidates[theme_index] if item.identity != selected[theme_index].identity), None)
            if alternative is not None:
                selected[theme_index] = alternative
                break
    return tuple(selected)


def select_theme_pairs(frames: Sequence[FrameData], max_delta: int = 5, far_range_m: float = 40.0) -> Tuple[PairSelection, ...]:
    if max_delta < 1:
        raise DatasetFormatError(Path("--max-delta"), "must be at least 1")
    if far_range_m <= 0:
        raise DatasetFormatError(Path("--far-range-m"), "must be positive")
    dense, cyclists, pedestrians, far = _candidate_sets(frames, max_delta, far_range_m)
    if not dense:
        raise MissingThemeError("dense")
    vru = cyclists if cyclists else pedestrians
    if not vru:
        raise MissingThemeError("vru")
    if not far:
        raise MissingThemeError("far")
    return _choose_distinct((dense, vru, far))


def _mpl_color(rgb):
    return rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0


def _class_color(object_type):
    if object_type in VEHICLE_CLASSES: return _mpl_color(VEHICLE_RGB)
    if object_type in PEDESTRIAN_CLASSES: return _mpl_color(PEDESTRIAN_RGB)
    if object_type in CYCLIST_CLASSES: return _mpl_color(CYCLIST_RGB)
    return _mpl_color(OTHER_RGB)


def _draw_camera(ax: Axes, frame: FrameData, calibration: Mapping[str, np.ndarray]) -> None:
    with Image.open(frame.image_path) as source:
        image = np.asarray(source.convert("RGB"))
    ax.imshow(image)
    image_xlim, image_ylim = (-0.5, image.shape[1] - 0.5), (image.shape[0] - 0.5, -0.5)
    for label in frame.labels:
        corners = transform_3dbox_to_image(label.dimensions, label.location, label.rotation, calibration)
        if corners is None or not np.all(np.isfinite(corners)): continue
        for start, end in CAMERA_EDGES:
            ax.plot(corners[[start, end], 0], corners[[start, end], 1], color=_class_color(label.object_type), linewidth=1.05, solid_capstyle="round", scalex=False, scaley=False)
    ax.set_xlim(image_xlim); ax.set_ylim(image_ylim)
    ax.text(0.018, 0.07, f"seq {frame.sequence} | frame {frame.frame:06d}", transform=ax.transAxes, color="white", fontsize=7.5, fontweight="normal", bbox={"facecolor": (0.05, 0.06, 0.07, 0.72), "edgecolor": "none", "pad": 2.5})
    ax.set_axis_off()


def _calibrated_velodyne_corners(label: GroundTruth, calibration: Mapping[str, np.ndarray]) -> np.ndarray:
    axis_swapped = transform_3dbox_to_pointcloud(label.dimensions, label.location, label.rotation)
    rectified_camera = np.column_stack((-axis_swapped[:, 1], -axis_swapped[:, 2], axis_swapped[:, 0]))
    rectification, velodyne_to_camera = np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)
    rectification[:3, :3], velodyne_to_camera[:3, :4] = calibration["Rect"], calibration["Tr_velo_cam"]
    rectified_homogeneous = np.column_stack((rectified_camera, np.ones(rectified_camera.shape[0])))
    velodyne_homogeneous = rectified_homogeneous @ np.linalg.inv(rectification @ velodyne_to_camera).T
    return velodyne_homogeneous[:, :3] / velodyne_homogeneous[:, 3, None]


def _draw_bev(ax: Axes, frame: FrameData, limits, calibration: Mapping[str, np.ndarray]) -> None:
    cloud = np.fromfile(str(frame.cloud_path), dtype=np.float32)
    if cloud.size % 4: raise DatasetFormatError(frame.cloud_path, "point count is not divisible into N x 4")
    points = cloud.reshape(-1, 4)
    ax.scatter(points[:, 0], points[:, 1], s=0.18, c=[_mpl_color(POINT_RGB)], alpha=0.56, linewidths=0, rasterized=True)
    for label in frame.labels:
        footprint = _calibrated_velodyne_corners(label, calibration)[list(BEV_FOOTPRINT)]
        ax.plot(footprint[:, 0], footprint[:, 1], color=_class_color(label.object_type), linewidth=1.35)
    ax.set_xlim(limits[0], limits[1]); ax.set_ylim(limits[2], limits[3]); ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=_mpl_color(GRID_RGB), linewidth=0.45, alpha=0.58)
    ax.tick_params(labelsize=6.5, colors=_mpl_color(TEXT_RGB), length=2.5, width=0.5)
    ax.set_xlabel("forward x (m)", fontsize=7, color=_mpl_color(TEXT_RGB), labelpad=2)
    ax.set_ylabel("lateral y (m)", fontsize=7, color=_mpl_color(TEXT_RGB), labelpad=2)
    ax.text(0.015, 0.955, f"seq {frame.sequence} | frame {frame.frame:06d}", transform=ax.transAxes, va="top", color=_mpl_color(TEXT_RGB), fontsize=7.5, fontweight="normal")
    for spine in ax.spines.values():
        spine.set_color(_mpl_color(GRID_RGB)); spine.set_linewidth(0.65)


def _bev_limits(frames: Iterable[FrameData], calibrations: Mapping[str, Mapping[str, np.ndarray]]):
    forward_min, forward_max, lateral_min, lateral_max = 0.0, 80.0, -30.0, 30.0
    for frame in frames:
        for label in frame.labels:
            corners = _calibrated_velodyne_corners(label, calibrations[frame.sequence])
            forward_min, forward_max = min(forward_min, float(np.min(corners[:, 0])) - 2.0), max(forward_max, float(np.max(corners[:, 0])) + 2.0)
            lateral_min, lateral_max = min(lateral_min, float(np.min(corners[:, 1])) - 2.0), max(lateral_max, float(np.max(corners[:, 1])) + 2.0)
    return forward_min, forward_max, lateral_min, lateral_max


def _render(kitti_root: Path, output_dir: Path, selections: Sequence[PairSelection], frames: Sequence[FrameData]) -> None:
    frame_index = {(frame.sequence, frame.frame): frame for frame in frames}
    selected_frames = tuple(frame_index[(selection.sequence, number)] for selection in selections for number in (selection.frame_t, selection.frame_t_plus_delta))
    calibrations = {selection.sequence: parse_calibration(kitti_root / "calib" / f"{selection.sequence}.txt") for selection in selections}
    limits = _bev_limits(selected_frames, calibrations)
    figure, axes = plt.subplots(4, 3, figsize=FIGURE_SIZE, dpi=FIGURE_DPI, gridspec_kw={"height_ratios": (1, 2, 1, 2)})
    figure.patch.set_facecolor("white")
    for column, selection in enumerate(selections):
        first, second = frame_index[(selection.sequence, selection.frame_t)], frame_index[(selection.sequence, selection.frame_t_plus_delta)]
        axes[0, column].set_title(THEME_TITLES[column], fontsize=11, fontweight="bold", color=_mpl_color(TEXT_RGB), pad=9)
        _draw_camera(axes[0, column], first, calibrations[selection.sequence]); _draw_bev(axes[1, column], first, limits, calibrations[selection.sequence])
        _draw_camera(axes[2, column], second, calibrations[selection.sequence]); _draw_bev(axes[3, column], second, limits, calibrations[selection.sequence])
    for row, label in enumerate(ROW_LABELS):
        position = axes[row, 0].get_position()
        figure.text(0.03, (position.y0 + position.y1) / 2.0, label, rotation=90, va="center", ha="center", fontsize=8.5, fontweight="bold", color=_mpl_color(TEXT_RGB))
    legend = (Line2D([0], [0], color=_mpl_color(VEHICLE_RGB), lw=2, label="Car / Van / Truck"), Line2D([0], [0], color=_mpl_color(PEDESTRIAN_RGB), lw=2, label="Pedestrian / Person_sitting"), Line2D([0], [0], color=_mpl_color(CYCLIST_RGB), lw=2, label="Cyclist"), Line2D([0], [0], color=_mpl_color(OTHER_RGB), lw=2, label="Other GT"))
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.976), ncol=4, frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.8)
    figure.subplots_adjust(left=0.085, right=0.992, bottom=0.065, top=0.92, wspace=0.16, hspace=0.24)
    try:
        figure.savefig(output_dir / ARTIFACT_NAMES[0], dpi=FIGURE_DPI, facecolor="white")
        figure.savefig(output_dir / ARTIFACT_NAMES[1], facecolor="white")
    finally:
        plt.close(figure)


def _write_manifest(kitti_root: Path, output_dir: Path, selections: Sequence[PairSelection], frames: Sequence[FrameData]) -> None:
    frame_index = {(frame.sequence, frame.frame): frame for frame in frames}; records = []
    for selection in selections:
        first, second = frame_index[(selection.sequence, selection.frame_t)], frame_index[(selection.sequence, selection.frame_t_plus_delta)]
        records.append({"theme": selection.theme, "sequence": selection.sequence, "frame_t": f"{selection.frame_t:06d}", "frame_t_plus_delta": f"{selection.frame_t_plus_delta:06d}", "delta": selection.delta, "shared_relevant_track_ids": list(selection.shared_track_ids), "selection_metrics": asdict(selection.metrics), "source_files": {"calibration": f"calib/{selection.sequence}.txt", "labels": f"label_2/{selection.sequence}.txt", "frame_t": {"image": first.image_path.relative_to(kitti_root).as_posix(), "velodyne": first.cloud_path.relative_to(kitti_root).as_posix()}, "frame_t_plus_delta": {"image": second.image_path.relative_to(kitti_root).as_posix(), "velodyne": second.cloud_path.relative_to(kitti_root).as_posix()}}})
    manifest = {"kitti_root": str(kitti_root.resolve()), "colors_rgb": {"Car/Van/Truck": list(VEHICLE_RGB), "Pedestrian/Person_sitting": list(PEDESTRIAN_RGB), "Cyclist": list(CYCLIST_RGB), "Other GT": list(OTHER_RGB)}, "selections": records}
    (output_dir / ARTIFACT_NAMES[2]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir(): raise OutputDirectoryError(output_dir, "path exists and is not a directory")
    if output_dir.exists() and not overwrite and any(output_dir.iterdir()): raise OutputDirectoryError(output_dir, "must be fresh unless --overwrite is passed")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ARTIFACT_NAMES:
            artifact = output_dir / name
            if artifact.is_file(): artifact.unlink()


def generate_figure8(kitti_root: Path, output_dir: Path, max_delta: int = 5, far_range_m: float = 40.0, overwrite: bool = False) -> Tuple[PairSelection, ...]:
    frames = load_frames(kitti_root)
    selections = select_theme_pairs(frames, max_delta, far_range_m)
    _prepare_output(output_dir, overwrite)
    _render(kitti_root, output_dir, selections, frames)
    _write_manifest(kitti_root, output_dir, selections, frames)
    return selections


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate KITTI Tracking real-GT Figure 8")
    parser.add_argument("--kitti-root", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-delta", default=5, type=int); parser.add_argument("--far-range-m", default=40.0, type=float); parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        generate_figure8(args.kitti_root, args.output_dir, args.max_delta, args.far_range_m, args.overwrite)
    except Figure8Error as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
