from __future__ import annotations

import json
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

figure8 = importlib.import_module("generate_figure8_gt")
geometry = importlib.import_module("visual_utils.kitti_figure8_geometry")
DatasetFormatError = figure8.DatasetFormatError
FrameData = figure8.FrameData
GroundTruth = figure8.GroundTruth
MissingThemeError = figure8.MissingThemeError
_bev_limits = figure8._bev_limits
_calibrated_velodyne_corners = figure8._calibrated_velodyne_corners
_draw_bev = figure8._draw_bev
_draw_camera = figure8._draw_camera
generate_figure8 = figure8.generate_figure8
parse_calibration = figure8.parse_calibration
parse_tracking_labels = figure8.parse_tracking_labels
select_theme_pairs = figure8.select_theme_pairs
transform_3dbox_to_image = geometry.transform_3dbox_to_image
transform_3dbox_to_pointcloud = geometry.transform_3dbox_to_pointcloud


def _gt(track_id: int, object_type: str, z: float) -> GroundTruth:
    return GroundTruth(track_id, object_type, (1.5, 1.6, 4.0), (0.0, 1.5, z), 0.0)


def _frame(sequence: str, frame: int, labels: tuple[GroundTruth, ...]) -> FrameData:
    stem = f"{frame:06d}"
    return FrameData(sequence, frame, labels, Path(stem + ".png"), Path(stem + ".bin"))


def _write_calibration(path: Path) -> None:
    projection = "100 0 60 0 0 100 30 0 0 0 1 0"
    calibration = "\n".join((
        f"P0: {projection}", f"P1 {projection}", f"P2: {projection}", f"P3 {projection}",
        "R_rect 1 0 0 0 1 0 0 0 1", "Tr_velo_cam 1 0 0 0 0 1 0 0 0 0 1 0",
        "Tr_imu_velo 1 0 0 0 0 1 0 0 0 0 1 0",
    ))
    path.write_text(calibration + "\n", encoding="utf-8")


def _write_dataset(root: Path, include_vru: bool) -> None:
    for folder in ("image_2/0000", "velodyne/0000", "calib", "label_2"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    label_lines = []
    for frame in (0, 1):
        Image.new("RGB", (120, 60), (205, 210, 214)).save(root / "image_2" / "0000" / f"{frame:06d}.png")
        np.array([[5, 0, 0, 1], [42, 4, 0, 0.6], [76, -8, 0, 0.2]], dtype=np.float32).tofile(
            root / "velodyne" / "0000" / f"{frame:06d}.bin")
        objects = [(1, "Car", -3, 15), (2, "Van", -1, 20), (3, "Truck", 1, 25), (9, "Car", 3, 45)]
        if include_vru:
            objects.append((10, "Cyclist", 0, 12))
        for track_id, object_type, x, z in objects:
            label_lines.append(f"{frame} {track_id} {object_type} 0 0 0 10 10 20 20 1.5 1.6 4.0 {x} 1.5 {z + frame} 0")
    (root / "label_2" / "0000.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    _write_calibration(root / "calib" / "0000.txt")


class ParsingTests(unittest.TestCase):
    def test_parses_tracking_labels_and_colon_optional_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels_path = root / "labels.txt"
            labels_path.write_text(
                "0 7 Car 0 1 0 1 2 3 4 1.5 1.6 4.2 2 1 41 0.2\n"
                "0 -1 DontCare -1 -1 -10 0 0 1 1 -1 -1 -1 -1 -1 -1 -1\n"
                "1 8 Cyclist 0 0 0 1 2 3 4 1.7 0.6 1.8 0 1 12 -0.1\n", encoding="utf-8")
            calibration_path = root / "calib.txt"
            calibration_path.write_text(
                "P0: 1 0 0 0 0 1 0 0 0 0 1 0\nP2 2 0 3 0 0 2 4 0 0 0 1 0\n"
                "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam 1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
            labels, calibration = parse_tracking_labels(labels_path), parse_calibration(calibration_path)
            self.assertEqual([item.track_id for item in labels[0]], [7])
            self.assertEqual(labels[1][0].object_type, "Cyclist")
            np.testing.assert_array_equal(calibration["P2"], np.array([[2, 0, 3, 0], [0, 2, 4, 0], [0, 0, 1, 0]]))
            np.testing.assert_array_equal(calibration["Rect"], np.eye(3))
            np.testing.assert_array_equal(calibration["Tr_velo_cam"], np.column_stack((np.eye(3), np.zeros(3))))
            self.assertEqual(set(calibration), {"P2", "Rect", "Tr_velo_cam"})

    def test_calibration_requires_rectification_and_velodyne_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calib.txt"
            projection = "1 0 0 0 0 1 0 0 0 0 1 0"
            cases = ((f"P2: {projection}\nTr_velo_cam: {projection}\n", "rectification"),
                     (f"P2: {projection}\nR_rect: 1 0 0 0 1 0 0 0 1\n", "Velodyne-to-camera"))
            for contents, missing_name in cases:
                with self.subTest(missing_name=missing_name):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(DatasetFormatError) as caught:
                        parse_calibration(path)
                    self.assertIn(missing_name, caught.exception.reason)


class GeometryTests(unittest.TestCase):
    def test_geometry_preserves_yaw_and_rejects_near_plane(self) -> None:
        dimensions, location = (2.0, 2.0, 4.0), (1.0, 2.0, 10.0)
        actual = transform_3dbox_to_pointcloud(dimensions, location, np.pi / 2)
        expected_camera_corner = np.array([2.0, 2.0, 8.0])
        np.testing.assert_allclose(actual[0], [expected_camera_corner[2], -expected_camera_corner[0], -expected_camera_corner[1]])
        projection = {"P2": np.array([[100, 0, 60, 0], [0, 100, 30, 0], [0, 0, 1, 0]], dtype=float)}
        self.assertIsNone(transform_3dbox_to_image(dimensions, (0.0, 2.0, 0.5), 0.0, projection))

    def test_off_image_projected_box_preserves_full_camera_raster_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "frame.png"
            Image.new("RGB", (120, 60), (0, 0, 0)).save(image_path)
            frame = FrameData("0000", 0, (_gt(1, "Car", 10),), image_path, Path("unused.bin"))
            calibration = {"P2": np.array([[100, 0, 60, 100000], [0, 100, 30, 0], [0, 0, 1, 0]])}
            figure, ax = plt.subplots()
            try:
                _draw_camera(ax, frame, calibration)
                self.assertEqual(ax.get_xlim(), (-0.5, 119.5)); self.assertEqual(ax.get_ylim(), (59.5, -0.5))
            finally:
                plt.close(figure)

    def test_non_identity_calibration_places_bev_footprint_in_raw_velodyne(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cloud_path = Path(temporary) / "frame.bin"
            np.array([[0, 0, 0, 1]], dtype=np.float32).tofile(cloud_path)
            label = GroundTruth(1, "Car", (2.0, 2.0, 4.0), (10.0, 2.0, 30.0), 0.0)
            frame = FrameData("0000", 0, (label,), Path("unused.png"), cloud_path)
            rect = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
            transform = np.array([[1, 0, 0, -100], [0, 1, 0, 2], [0, 0, 1, 3]], dtype=float)
            calibration = {"P2": np.zeros((3, 4)), "Rect": rect, "Tr_velo_cam": transform}
            camera_corners = np.array([[12, 2, 31], [12, 2, 29], [8, 2, 29], [8, 2, 31], [12, 0, 31], [12, 0, 29], [8, 0, 29], [8, 0, 31]], dtype=float)
            expected = camera_corners @ rect - np.array([-100, 2, 3], dtype=float)
            actual = _calibrated_velodyne_corners(label, calibration)
            limits = _bev_limits((frame,), {"0000": calibration})
            figure, ax = plt.subplots()
            try:
                _draw_bev(ax, frame, limits, calibration)
                footprint = expected[[4, 5, 6, 7, 4]]
                np.testing.assert_allclose(actual, expected)
                np.testing.assert_allclose(np.asarray(ax.lines[0].get_xdata(), dtype=float), footprint[:, 0])
                np.testing.assert_allclose(np.asarray(ax.lines[0].get_ydata(), dtype=float), footprint[:, 1])
                self.assertEqual(limits[1], 104.0)
            finally:
                plt.close(figure)


class SelectionTests(unittest.TestCase):
    def test_prefers_cyclist_globally_and_enforces_shared_far_tracks(self) -> None:
        dense = (_gt(1, "Car", 15), _gt(2, "Van", 20), _gt(3, "Truck", 25), _gt(20, "Car", 35))
        cyclist, pedestrian, far = (_gt(10, "Cyclist", 12),), (_gt(30, "Pedestrian", 10), _gt(31, "Pedestrian", 11)), (_gt(40, "Tram", 61),)
        frames = (_frame("0000", 0, dense + cyclist), _frame("0000", 1, dense + cyclist), _frame("0001", 0, dense + pedestrian), _frame("0001", 1, dense + pedestrian), _frame("0002", 0, dense + far), _frame("0002", 1, dense + far))
        selected = {item.theme: item for item in select_theme_pairs(frames, 5, 40.0)}
        self.assertEqual(selected["vru"].metrics.preferred_class, "Cyclist"); self.assertEqual(selected["vru"].shared_track_ids, (10,))
        self.assertEqual(selected["far"].sequence, "0002"); self.assertEqual(selected["far"].shared_track_ids, (40,)); self.assertGreaterEqual(selected["far"].metrics.shared_min_range_m, 60.0)

    def test_falls_back_to_pedestrian_when_no_cyclist_pair_exists(self) -> None:
        labels = (_gt(1, "Car", 15), _gt(2, "Van", 20), _gt(3, "Truck", 25), _gt(9, "Car", 45), _gt(12, "Person_sitting", 10))
        selected = {item.theme: item for item in select_theme_pairs((_frame("0000", 0, labels), _frame("0000", 1, labels)), 5, 40.0)}
        self.assertEqual(selected["vru"].metrics.preferred_class, "Pedestrian/Person_sitting"); self.assertEqual(selected["vru"].shared_track_ids, (12,))

    def test_raises_typed_error_when_vru_theme_is_missing(self) -> None:
        labels = (_gt(1, "Car", 15), _gt(2, "Van", 20), _gt(3, "Truck", 25), _gt(9, "Car", 45))
        with self.assertRaises(MissingThemeError) as caught:
            select_theme_pairs((_frame("0000", 0, labels), _frame("0000", 1, labels)), 5, 40.0)
        self.assertEqual(caught.exception.theme, "vru")


class ExportTests(unittest.TestCase):
    def test_exports_png_pdf_and_manifest_from_real_fixture_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "KITTI", Path(temporary) / "output"
            _write_dataset(root, include_vru=True)
            with patch("generate_figure8_gt.plt.close"):
                generate_figure8(root, output, max_delta=5, far_range_m=40.0)
                figure = plt.gcf(); figure.canvas.draw()
            try:
                self.assertGreater(figure.axes[3].get_position().height, figure.axes[0].get_position().height * 1.5)
                row_labels = {text.get_text(): text for text in figure.texts}
                for label in ("Camera t", "BEV t", "Camera t+delta", "BEV t+delta"):
                    bounds = row_labels[label].get_window_extent()
                    self.assertGreaterEqual(bounds.x0, figure.bbox.x0); self.assertLessEqual(bounds.x1, figure.bbox.x1)
            finally:
                plt.close(figure)
            manifest = json.loads((output / "figure8_kitti_gt_selection.json").read_text(encoding="utf-8"))
            with Image.open(output / "figure8_kitti_gt.png") as image:
                self.assertGreater(image.width, image.height)
            self.assertTrue((output / "figure8_kitti_gt.pdf").read_bytes().startswith(b"%PDF"))
            self.assertEqual([item["theme"] for item in manifest["selections"]], ["dense", "vru", "far"])
            self.assertEqual(manifest["selections"][0]["source_files"]["frame_t"]["image"], "image_2/0000/000000.png")

    def test_missing_vru_writes_no_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "KITTI", Path(temporary) / "output"
            _write_dataset(root, include_vru=False)
            with self.assertRaises(MissingThemeError):
                generate_figure8(root, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
