import csv
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from test_figure7_renderer_loading import completed_capture
from tools.figure7_utils.publication import OUTPUT_NAMES, PublicationError, publish_figure7


def snapshot(directory: Path) -> tuple[tuple[str, bytes | str], ...]:
    entries = []
    for path in sorted(directory.rglob('*')):
        value = str(path.readlink()) if path.is_symlink() else (
            path.read_bytes() if path.is_file() else 'directory')
        entries.append((str(path.relative_to(directory)), value))
    return tuple(entries)


class TestFigure7RendererPublication(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_publication_emits_exact_atomic_bundle(self) -> None:
        """Given complete evidence, when published, then only six declared artifacts exist."""
        output = self.root / 'output'
        result = publish_figure7(completed_capture(self.root), output)
        self.assertEqual(result.artifacts, OUTPUT_NAMES)
        self.assertEqual({path.name for path in output.iterdir()}, set(OUTPUT_NAMES))
        with (output / 'figure7_candidates.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row['selected_case'] for row in rows}, {'A', 'B'})
        self.assertEqual(len(tuple((output / 'figure7_caseA_candidates').glob('*.png'))), 1)
        self.assertEqual(len(tuple((output / 'figure7_caseB_candidates').glob('*.png'))), 1)
        summary = (output / 'figure7_summary.md').read_text()
        for required in ('positive-query-only weighting', 'K4 protocol', 'ledger records', 'crop'):
            self.assertIn(required, summary)

    def test_publication_preserves_existing_bundle_when_staging_fails(self) -> None:
        """Given an existing bundle, when rendering fails, then old bytes remain untouched."""
        output = self.root / 'output'
        output.mkdir()
        sentinel = output / 'figure7_summary.md'
        sentinel.write_text('old bundle')
        with patch('tools.figure7_utils.publication.write_bundle', side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):
                publish_figure7(completed_capture(self.root), output, overwrite=True)
        self.assertEqual(sentinel.read_text(), 'old bundle')

    def test_unsafe_paths_are_rejected_before_loading_or_mutation(self) -> None:
        """Given overlapping paths or leaf links, when published, then evidence is untouched."""
        capture = completed_capture(self.root / 'evidence')
        unrelated = self.root / 'unrelated'
        unrelated.mkdir()
        (unrelated / 'sentinel').write_bytes(b'unrelated evidence')
        (capture / 'sentinel').write_bytes(b'capture evidence')
        alias = self.root / 'alias'
        alias.symlink_to(capture, target_is_directory=True)
        parent_alias = self.root / 'parent-alias'
        parent_alias.symlink_to(capture.parent, target_is_directory=True)
        leaf = self.root / 'leaf'
        leaf.symlink_to(unrelated, target_is_directory=True)
        dangling = self.root / 'dangling'
        dangling.symlink_to(self.root / 'absent', target_is_directory=True)
        lexical_parent = self.root / 'lexical-parent'
        lexical_parent.mkdir()
        capture_link = lexical_parent / 'capture-link'
        capture_link.symlink_to(capture, target_is_directory=True)
        escape = capture / 'escape'
        escape.symlink_to(unrelated, target_is_directory=True)
        cases = (
            (capture, capture),
            (capture, capture.parent),
            (capture, capture / 'nested' / 'output'),
            (capture, capture / '..' / 'capture'),
            (capture / '..' / 'capture', capture.parent),
            (capture, alias),
            (alias, capture),
            (alias, capture.parent),
            (capture, alias / 'missing' / 'output'),
            (capture, parent_alias / 'capture' / 'missing' / 'output'),
            (capture, parent_alias / 'capture'),
            (capture_link, lexical_parent),
            (capture, escape / 'output'),
            (capture, leaf),
            (capture, dangling),
        )
        before = snapshot(self.root)
        for source, output in cases:
            for overwrite in (False, True):
                with self.subTest(source=source, output=output, overwrite=overwrite):
                    with ExitStack() as stack:
                        for name in ('load_capture', 'mkdtemp', 'write_bundle', '_publish'):
                            stack.enter_context(patch(
                                'tools.figure7_utils.publication.' + name,
                                side_effect=AssertionError('preflight ran too late: ' + name),
                            ))
                        try:
                            with self.assertRaises(PublicationError):
                                publish_figure7(source, output, overwrite=overwrite)
                        finally:
                            self.assertEqual(snapshot(self.root), before)
                            self.assertEqual(list(self.root.rglob('.*.staging.*')), [])
                            self.assertEqual(list(self.root.rglob('.*.backup.*')), [])

    def test_publication_error_propagates_through_context_manager(self) -> None:
        """Given a publication error, when context cleanup runs, then traceback stays writable."""
        error = PublicationError('rejected')
        with self.assertRaises(PublicationError) as raised:
            with ExitStack():
                error.__traceback__ = None
                raise error
        self.assertIs(raised.exception, error)

    def test_sibling_overwrite_preserves_capture_and_unrelated_files(self) -> None:
        """Given a sibling output, when explicitly overwritten, then only that bundle changes."""
        capture = completed_capture(self.root)
        output = self.root / 'capture-output'
        output.mkdir()
        (output / 'old').write_bytes(b'old bundle')
        sentinel = self.root / 'sentinel'
        sentinel.write_bytes(b'unrelated')
        before = snapshot(capture)
        result = publish_figure7(capture, output, overwrite=True)
        self.assertEqual(result.output_dir, output)
        self.assertEqual({path.name for path in output.iterdir()}, set(OUTPUT_NAMES))
        self.assertEqual(snapshot(capture), before)
        self.assertEqual(sentinel.read_bytes(), b'unrelated')
        self.assertEqual(list(self.root.glob('.*.staging.*')), [])
        self.assertEqual(list(self.root.glob('.*.backup.*')), [])

    def test_existing_sibling_requires_explicit_overwrite(self) -> None:
        """Given an unrelated existing output, when overwrite is off, then no files change."""
        capture = completed_capture(self.root)
        output = self.root / 'output'
        output.mkdir()
        (output / 'sentinel').write_bytes(b'preserve me')
        before = snapshot(self.root)
        with self.assertRaises(FileExistsError):
            publish_figure7(capture, output)
        self.assertEqual(snapshot(self.root), before)

    def test_existing_bundle_is_restored_when_final_rename_fails(self) -> None:
        """Given a staged replacement, when its rename fails, then backup restores old bytes."""
        capture = completed_capture(self.root)
        output = self.root / 'output'
        output.mkdir()
        (output / 'sentinel').write_bytes(b'old bundle')
        (self.root / 'unrelated').write_bytes(b'untouched')
        before = snapshot(self.root)
        replace = Path.replace

        def fail_staging(source: Path, target: Path) -> Path:
            if source.name.startswith('.output.staging.'):
                raise OSError('injected final rename failure')
            return replace(source, target)

        with patch.object(Path, 'replace', fail_staging):
            with self.assertRaises(OSError):
                publish_figure7(capture, output, overwrite=True)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(list(self.root.glob('.*.staging.*')), [])
        self.assertEqual(list(self.root.glob('.*.backup.*')), [])
