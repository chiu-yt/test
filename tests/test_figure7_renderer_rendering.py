from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
import numpy as np

from test_figure7_renderer_loading import completed_capture
from tools.figure7_utils.loading import load_capture
from tools.figure7_utils.rendering import _panel_data, crop_for_case, render_plate, render_preview


class TestFigure7RendererRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_crop_is_row_local_square_and_clipped_to_configured_range(self) -> None:
        """Given a selected proposal, when cropped, then the required bounded half-span is used."""
        selected = load_capture(completed_capture(self.root)).case_a
        crop = crop_for_case(selected, (-15., -20., 15., 20.))
        self.assertEqual(crop, (-9., -12., 15., 12.))

    def test_rendering_uses_array_correspondence_not_ledger(self) -> None:
        """Given stale ledger display data, when rendered, then validated arrays drive panels."""
        bundle = load_capture(completed_capture(self.root))
        record = replace(bundle.case_b, reference=replace(
            bundle.case_b.reference, matches=(-1,) * 4, qualities=(.123,) * 4))
        figure = render_plate(replace(bundle, case_b=record), (-20., -20., 20., 20.))
        try:
            self.assertEqual(len(figure.axes[6].patches), 2)
            self.assertEqual(figure.axes[6].patches[-1].get_edgecolor(), to_rgba('#2E7D32', 1.0))
        finally:
            plt.close(figure)

    def test_nonfinite_context_is_not_transformed_or_drawn(self) -> None:
        """Given rejected invalid context before a match, when drawn, then indices stay intact."""
        bundle = load_capture(completed_capture(self.root))
        record = bundle.case_b
        arrays = dict(record.arrays)
        for prefix in ('reference', 'view_0', 'view_1', 'view_2', 'view_3'):
            arrays[prefix + '_boxes'] = np.concatenate((
                np.full((1, 7), np.inf), arrays[prefix + '_boxes']))
            for suffix in ('labels', 'mask', 'rescue_mask'):
                arrays[prefix + '_' + suffix] = np.concatenate((
                    arrays[prefix + '_' + suffix][:1], arrays[prefix + '_' + suffix]))
        arrays['match_indices'] = np.array([[-1] * 4, [1] * 4])
        arrays['view_quality'] = np.array([[0.] * 4, [1.] * 4])
        record = replace(record, arrays=arrays, candidate=replace(record.candidate, reference_index=1))
        with np.errstate(invalid='raise'):
            _, boxes, _ = _panel_data(record, 1)
            figure = render_plate(replace(bundle, case_b=record), (-20., -20., 20., 20.))
        try:
            self.assertTrue(np.isinf(boxes[0]).all())
            self.assertEqual(len(figure.axes[6].patches), 2)
            self.assertTrue(all(np.isfinite(patch.get_path().vertices).all() for patch in figure.axes[6].patches))
            figure.canvas.draw()
        finally:
            plt.close(figure)

    def test_plate_and_preview_have_five_clean_bev_columns(self) -> None:
        bundle = load_capture(completed_capture(self.root))
        figure = render_plate(bundle, (-20., -20., 20., 20.))
        preview = render_preview(bundle.case_a, (-20., -20., 20., 20.))
        try:
            self.assertEqual(len(figure.axes), 10)
            self.assertEqual(len(preview.axes), 5)
            self.assertEqual([axis.get_title() for axis in figure.axes[:5]], [
                'Reference', 'View 1', 'View 2', 'View 3', 'View 4'])
            self.assertEqual(figure.axes[0].get_xlim(), figure.axes[4].get_xlim())
            self.assertEqual(figure.axes[0].get_ylim(), figure.axes[4].get_ylim())
            forbidden = ('selected quality', 'matched quality', 'accepted', 'rescued',
                         'forward x', 'lateral y')
            panel_text = ' '.join(
                text.get_text().lower()
                for rendered in (figure, preview)
                for axis in rendered.axes
                for text in axis.texts
            )
            panel_text += ' ' + ' '.join(
                '%s %s' % (axis.get_xlabel().lower(), axis.get_ylabel().lower())
                for rendered in (figure, preview) for axis in rendered.axes
            )
            self.assertTrue(all(fragment not in panel_text for fragment in forbidden))
        finally:
            plt.close(figure)
            plt.close(preview)

    def test_selected_and_matched_outlines_encode_quality_without_unmatched_geometry(self) -> None:
        record = load_capture(completed_capture(self.root)).case_b
        arrays = dict(record.arrays)
        arrays['match_indices'] = np.array([[0, 0, 0, -1]])
        arrays['view_quality'] = np.array([[.85, .70, .699, 0.]])

        figure = render_preview(replace(record, arrays=arrays), (-20., -20., 20., 20.))
        try:
            expected = ('#2E7D32', '#2E7D32', '#EF6C00', '#C62828')
            for axis, color in zip(figure.axes[:4], expected):
                selected = axis.patches[-1]
                self.assertEqual(selected.get_edgecolor(), to_rgba(color, 1.0))
                self.assertEqual(selected.get_linewidth(), 2.4)
            unmatched = figure.axes[4]
            self.assertEqual(len(unmatched.patches), 1)
            self.assertTrue(all(spine.get_visible() for spine in unmatched.spines.values()))
            self.assertTrue(all(spine.get_edgecolor() == to_rgba('#C62828')
                                for spine in unmatched.spines.values()))
            self.assertTrue(all(spine.get_linewidth() == 1.2
                                for spine in unmatched.spines.values()))
        finally:
            plt.close(figure)

    def test_reference_frame_is_preserved_while_views_are_inverse_aligned(self) -> None:
        """Given nonidentity transforms, when panel data loads, then only views are aligned."""
        bundle = load_capture(completed_capture(self.root))
        selected = bundle.case_a
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [5., -2., 1.]
        selected.arrays['reference_transform'][:] = transform
        selected.arrays['view_0_transform'][:] = transform

        reference_points, reference_boxes, _ = _panel_data(selected, 0)
        view_points, view_boxes, _ = _panel_data(selected, 1)

        np.testing.assert_array_equal(reference_points, selected.arrays['reference_points'])
        np.testing.assert_array_equal(reference_boxes, selected.arrays['reference_boxes'])
        np.testing.assert_allclose(
            view_points[:, :3], selected.arrays['view_0_points'][:, :3] - transform[:3, 3])
        np.testing.assert_allclose(
            view_boxes[:, :3], selected.arrays['view_0_boxes'][:, :3] - transform[:3, 3])
        self.assertEqual(crop_for_case(selected, (-20., -20., 20., 20.)),
                         (-4., -12., 20., 12.))

        figure = render_plate(bundle, (-20., -20., 20., 20.))
        try:
            for axis in figure.axes[:5]:
                self.assertEqual(axis.get_xlim(), (-12., 12.))
                self.assertEqual(axis.get_ylim(), (-4., 20.))
        finally:
            plt.close(figure)

    def test_sparse_captured_points_remain_visibly_inspectable(self) -> None:
        """Given sparse real points, when rendered, then context is not sub-pixel or transparent."""
        bundle = load_capture(completed_capture(self.root))
        figure = render_plate(bundle, (-20., -20., 20., 20.))
        try:
            context = next(item for item in figure.axes[0].collections
                           if isinstance(item, PathCollection))
            alpha = context.get_alpha()
            self.assertGreaterEqual(float(context.get_sizes()[0]), 3.9)
            self.assertIsNotNone(alpha)
            assert alpha is not None
            self.assertGreaterEqual(float(alpha), .6)
        finally:
            plt.close(figure)
