import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

from tools.figure5_utils import final_rendering
from tools.figure5_utils.domain import FrameRecord, SampleToken
from tools.figure5_utils.final_selection import FinalRowSelection


def _render_rows():
    token = SampleToken('horizontal-style-regression')
    frame = FrameRecord(token, (), (), (), ())
    selection = FinalRowSelection(1, frame, (), (0.0, 0.0, 4.0, 8.0))
    points = np.array([[1.0, 2.0, 0.0], [3.0, 6.0, 0.0]])
    return tuple(final_rendering.FinalRenderRow(selection, points) for _ in range(3))


def test_horizontal_scatter_strengthens_only_point_style():
    # Given identical render rows for legacy and horizontal plates.
    rows = _render_rows()

    # When both plate variants render through the shared row path.
    legacy = final_rendering.render_final_plate(rows, show_callouts=False)
    horizontal = final_rendering.render_horizontal_plate(rows, show_callouts=False)

    # Then coordinates and rasterization match while only horizontal marker weight increases.
    expected_offsets = np.array([[2.0, 1.0], [6.0, 3.0]])
    for legacy_axis, horizontal_axis in zip(legacy.axes, horizontal.axes):
        legacy_scatter = legacy_axis.collections[0]
        horizontal_scatter = horizontal_axis.collections[0]
        assert isinstance(legacy_scatter, PathCollection)
        assert isinstance(horizontal_scatter, PathCollection)
        np.testing.assert_array_equal(legacy_scatter.get_offsets(), expected_offsets)
        np.testing.assert_array_equal(horizontal_scatter.get_offsets(), expected_offsets)
        assert legacy_scatter.get_sizes()[0] == final_rendering.POINT_SIZE
        assert legacy_scatter.get_alpha() == final_rendering.POINT_ALPHA
        assert horizontal_scatter.get_sizes()[0] == 0.65
        assert horizontal_scatter.get_alpha() == 0.85
        assert final_rendering.HORIZONTAL_POINT_COLOR == '#707070'
        assert final_rendering._HORIZONTAL_POINT_STYLE.color == '#707070'
        assert legacy_scatter.get_rasterized()
        assert horizontal_scatter.get_rasterized()
    plt.close(legacy)
    plt.close(horizontal)
