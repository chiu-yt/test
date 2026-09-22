from pathlib import Path
from unittest.mock import patch

from matplotlib.figure import Figure

from tools.figure5_utils import final_artifacts
from tools.figure5_utils.final_rendering import FINAL_DPI


def test_pdf_save_uses_final_dpi_for_rasterized_artists():
    # Given a figure whose save call is observed without writing a file.
    figure = Figure()
    output = Path('figure5.pdf')

    # When the PDF export helper saves the figure.
    with patch.object(figure, 'savefig') as savefig:
        final_artifacts._save_pdf(figure, output)

    # Then the rasterized artist resolution uses the final publication DPI.
    savefig.assert_called_once()
    assert savefig.call_args.kwargs['dpi'] == FINAL_DPI
