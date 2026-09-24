from collections import Counter
from dataclasses import dataclass
from os.path import abspath
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Final, Tuple
from uuid import uuid4

from .candidates import write_candidates
from .contracts import CaptureBundle, Crop, SelectedRecord
from .loading import load_capture
from .rendering import crop_for_case, render_plate, save_figure


OUTPUT_NAMES: Final[Tuple[str, ...]] = (
    'figure7_candidates.csv', 'figure7_caseA_candidates', 'figure7_caseB_candidates',
    'figure7_confidence_vs_reliability.png', 'figure7_confidence_vs_reliability.pdf',
    'figure7_summary.md',
)
DEFAULT_RANGE: Final[Crop] = (-54.0, -54.0, 54.0, 54.0)


@dataclass(frozen=True, slots=True)
class PublicationResult:
    output_dir: Path
    artifacts: Tuple[str, ...]


class PublicationError(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


def _selection_line(label: str, record: SelectedRecord, point_range: Crop) -> str:
    candidate, reference = record.candidate, record.reference
    crop = crop_for_case(record, point_range)
    occurrence = candidate.identity
    return (
        '- **%s** occurrence=`%s` frame=`%s` epoch=%d model_step=%d rank=%d batch=%d; '
        'proposal=%d class=`%s`; q=`%.17g`; exact r=`%.17g`; accepted=`%s`; rescued=`%s`; '
        'matches=`%s`; qualities=`%s`; ranking_key=`%s`; crop=`%s`.'
        % (label, occurrence.token, occurrence.frame_id, occurrence.epoch,
           candidate.model_step, occurrence.global_rank, occurrence.batch_index,
           candidate.reference_index, candidate.class_name, candidate.score,
           candidate.reliability, str(reference.accepted).lower(),
           str(reference.rescued).lower(), reference.matches, reference.qualities,
           candidate.rank_key, crop)
    )


def summary_text(bundle: CaptureBundle, point_range: Crop) -> str:
    counts = Counter(entry.status for entry in bundle.ledger)
    retained_a = sum(item.candidate.pool == 'variable' for item in bundle.retained)
    retained_b = sum(item.candidate.pool == 'stable' for item in bundle.retained)
    provenance = bundle.provenance
    lines = [
        '# Figure 7 confidence versus reliability summary', '',
        '## Selected evidence', '',
        _selection_line('Case A', bundle.case_a, point_range),
        _selection_line('Case B', bundle.case_b, point_range), '',
        '## Capture and provenance', '',
        '- config: `%s`' % provenance.config,
        '- command: `%s`' % provenance.command,
        '- source checkpoint: `%s`' % provenance.source_checkpoint,
        '- fixed seed: `%s`' % provenance.fixed_seed,
        '- ledger records: `%d` (%s)' % (
            len(bundle.ledger), ', '.join('%s=%d' % item for item in sorted(counts.items()))),
        '- retained candidates: Case A=`%d`, Case B=`%d`, physical records=`%d`' % (
            retained_a, retained_b, len({item.candidate.record_id for item in bundle.retained})),
        '- configured reference range: `%s`' % (point_range,), '',
        '## K4 protocol', '',
        '- selection policy: `%s`' % bundle.selection_policy,
        '- Case A: `q >= 0.85` and `r <= 0.30`; Case B: `q >= 0.65`, `r >= 0.80`, '
        'and all four views matched.',
        '- Pairing prefers same class, nearest confidence, largest reliability gap, and distinct tokens.',
        '- Four captured view predictions and actual view points are inverse-aligned with the K4 '
        '`(value - translation) @ inverse(linear).T` convention; no view is reconstructed.',
        '- Reliability is the exact captured fixed-four-view mean quality and is shown on one shared '
        'fixed `[0,1]` red-yellow-green scale.', '',
        '- The continuous reliability scale is intentional: unlike Figure 6 discrete bins, Figure 7 '
        'preserves the selected proposals\' exact `r` positions.', '',
        '## Rendering contract', '',
        '- final plate: exactly `2 rows x 6 columns`.',
        '- horizontal y / vertical x orientation: `true`.',
        '- row-local crop: half-span `max(12 m, 2 * max(dx,dy))`, clipped to configured range.',
        '- identical limits across Reference and all four views in each row: `true`.',
        '- PNG: `600 DPI`; PDF: vector text and boxes with rasterized point context.', '',
        '## Limitation', '',
        '- **positive-query-only weighting**: reliability is attached only to accepted positive '
        'reference queries in the current method; this figure does not establish calibration for '
        'negative, suppressed, or post-update predictions.',
        '- This bounded plate is a two-selected-case mechanism illustration, not a population-level '
        'calibration curve or distributional analysis.', '',
    ]
    return '\n'.join(lines)


def write_bundle(directory: Path, bundle: CaptureBundle, point_range: Crop) -> None:
    write_candidates(bundle, directory, point_range)
    save_figure(
        render_plate(bundle, point_range),
        directory / 'figure7_confidence_vs_reliability.png',
        directory / 'figure7_confidence_vs_reliability.pdf',
    )
    (directory / 'figure7_summary.md').write_text(summary_text(bundle, point_range), encoding='utf-8')


def _publish(staging: Path, output: Path, overwrite: bool) -> None:
    if not output.exists():
        staging.replace(output)
        return
    if not overwrite:
        raise FileExistsError('Figure 7 output already exists: %s' % output)
    backup = output.parent / ('.%s.backup.%s' % (output.name, uuid4().hex))
    output.replace(backup)
    try:
        staging.replace(output)
    except OSError:
        backup.replace(output)
        raise
    shutil.rmtree(backup)


def publish_figure7(capture_dir: Path, output_dir: Path, overwrite: bool = False,
                    point_range: Crop = DEFAULT_RANGE) -> PublicationResult:
    if output_dir.is_symlink():
        raise PublicationError('Figure 7 output must not be a symlink: %s' % output_dir)
    for capture, output in (
        (Path(abspath(capture_dir)), Path(abspath(output_dir))),
        (capture_dir.resolve(), output_dir.resolve()),
    ):
        if capture == output or capture in output.parents or output in capture.parents:
            raise PublicationError('Figure 7 capture and output directories must not overlap')
    if point_range[0] >= point_range[2] or point_range[1] >= point_range[3]:
        raise PublicationError('point-cloud range must be nondegenerate')
    if output_dir.exists() and not overwrite:
        raise FileExistsError('Figure 7 output already exists: %s' % output_dir)
    bundle = load_capture(capture_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix='.%s.staging.' % output_dir.name, dir=str(output_dir.parent)))
    published = False
    try:
        write_bundle(staging, bundle, point_range)
        actual = tuple(sorted(path.name for path in staging.iterdir()))
        if actual != tuple(sorted(OUTPUT_NAMES)):
            raise PublicationError('staged bundle differs from the six-artifact contract')
        _publish(staging, output_dir, overwrite)
        published = True
        return PublicationResult(output_dir, OUTPUT_NAMES)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
