from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from test_figure7_renderer_loading import completed_capture
from tools.figure7_utils.publication import OUTPUT_NAMES


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'tools' / 'generate_figure7.py'


class TestFigure7RendererCli(unittest.TestCase):
    def test_synthetic_cli_e2e_publishes_exact_bundle(self) -> None:
        """Given synthetic complete evidence, when CLI runs, then the real surface succeeds."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            capture = completed_capture(root)
            output = root / 'published'
            result = subprocess.run([
                sys.executable, str(CLI), '--capture_dir', str(capture),
                '--output_dir', str(output), '--point_cloud_range',
                '-20', '-20', '20', '20',
            ], cwd=ROOT / 'tools', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({path.name for path in output.iterdir()}, set(OUTPUT_NAMES))
            self.assertIn('Published 6 Figure 7 artifacts', result.stdout)
