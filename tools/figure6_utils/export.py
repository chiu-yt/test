from dataclasses import dataclass
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Final, Tuple
from uuid import uuid4

from .contracts import load_crop_manifest
from .loading import EvidenceBundle, TokenEvidence, load_evidence
from .rendering import save_plate, save_row


MAIN_NAMES: Final[Tuple[str, ...]] = (
    'figure6_refined_2row.png', 'figure6_refined_2row.pdf',
    'figure6_row_a.png', 'figure6_row_b.png', 'figure6_refine_summary.md',
)
ALT_NAMES: Final[Tuple[str, ...]] = (
    'figure6_refined_2row_alt.png', 'figure6_refined_2row_alt.pdf',
)


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class ExportResult:
    output_dir: Path
    artifacts: Tuple[str, ...]


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class ExportBundleError(RuntimeError):
    expected: Tuple[str, ...]
    actual: Tuple[str, ...]

    def __str__(self) -> str:
        return 'staged Figure 6 bundle differs from the declared artifact contract'


def artifact_names(include_alt: bool = False) -> Tuple[str, ...]:
    return MAIN_NAMES + (ALT_NAMES if include_alt else ())


def _stage_states(row: TokenEvidence) -> str:
    return ', '.join('%s=`%s`' % (stage.title, stage.runtime_state.value)
                     for stage in row.stages)


def _protocol(row: TokenEvidence) -> str:
    if not row.protocol:
        return 'unavailable'
    return '; '.join('%s=%s' % item for item in sorted(row.protocol.items()))


def _summary(bundle: EvidenceBundle, include_alt: bool) -> str:
    rows = bundle.rows[:2]
    lines = [
        '# Figure 6 refinement summary', '',
        '## Fixed rows', '',
        '- `(a) Far-range recovery`: `%s`' % rows[0].token,
        '- `(b) Small-object recovery`: `%s`' % rows[1].token,
        '', '## Visualization methods', '',
        '1. LiDAR Density: log-density with faint captured point and one enlarged clipped ROI.',
        '2. SPCRA Reliability: discrete low/mid/high red-yellow-green box coloring.',
        '3. RG-PLM Retained: light-grey SPCRA-evaluated candidate context beneath '
        'positive valid retained pseudo labels at linewidth 1.6.',
        '4. SG-DFA Response: signed channel-mean delta with nearest interpolation.',
        '5. Final Detection: captured pre-update arrays with the Figure 5 class palette.',
        '', '## Refinement contract', '',
        '- exact Figure 5 horizontal crops: `true`',
        '- horizontal-y / vertical-x orientation: `true`',
        '- identical crop across each row: `true`',
        '- discrete reliability thresholds `[0,1/3,2/3,1]`: `true`',
        '- constant reliability opacity: `true`',
        '- class and score labels removed: `true`',
        '- one enlarged clipped canonical ROI per row: `true`',
        '- at most one ROI-based reliability annotation per row: `true`',
        '- array length mismatches rejected: `true`',
        '- positive valid RG-PLM labels only: `true`',
        '- grey boxes denote SPCRA-evaluated candidate context, not proven rejected labels: `true`',
        '- effective pseudo class column 7: `true`',
        '- injection pseudo class column 9: `true`',
        '- RG-PLM retained linewidth `1.6`: `true`',
        '- Final Detection linewidth `1.4`: `true`',
        '- signed SG-DFA channel mean: `true`',
        '- nearest SG-DFA interpolation: `true`',
        '- row-local 98th-percentile SG-DFA scale: `true`',
        '- faint captured context and one enlarged clipped Figure 5 ROI overlay: `true`',
        '- pre-update final detection: `true`',
        '- 7.2-inch white plate: `true`',
        '- 600 DPI PNG and vector PDF text/boxes: `true`',
        '- alternate plate included: `%s`' % str(include_alt).lower(),
        '- alternate difference: `%s`' % (
            'compact reliability r= annotations suppressed; all other plate content retained.'
            if include_alt else 'not exported'
        ),
        '', '## Runtime evidence', '',
    ]
    for row in rows:
        lines.extend([
            '- `%s` protocol: %s' % (row.token, _protocol(row)),
            '- `%s` stage states: %s' % (row.token, _stage_states(row)),
        ])
    lines.extend([
        '- corrupt or incomplete runtime records: `%d`' % len(bundle.issues),
        '',
    ])
    return '\n'.join(lines)


def _write_bundle(directory: Path, bundle: EvidenceBundle, include_alt: bool) -> None:
    rows = bundle.rows[:2]
    save_plate(rows, directory / MAIN_NAMES[0], directory / MAIN_NAMES[1])
    save_row(rows[0], directory / MAIN_NAMES[2])
    save_row(rows[1], directory / MAIN_NAMES[3])
    (directory / MAIN_NAMES[4]).write_text(_summary(bundle, include_alt), encoding='utf-8')
    if include_alt:
        save_plate(rows, directory / ALT_NAMES[0], directory / ALT_NAMES[1], alternate=True)


def _publish(staging: Path, output: Path, overwrite: bool) -> None:
    if not output.exists():
        staging.replace(output)
        return
    if not overwrite:
        raise FileExistsError('Figure 6 output already exists: %s' % output)
    backup = output.parent / ('.%s.backup.%s' % (output.name, uuid4().hex))
    output.replace(backup)
    try:
        staging.replace(output)
    except OSError:
        backup.replace(output)
        raise
    shutil.rmtree(backup)


def export_figure6(capture_dir: Path, crop_manifest: Path, output_dir: Path,
                   overwrite: bool = False, include_alt: bool = False) -> ExportResult:
    if output_dir.exists() and not overwrite:
        raise FileExistsError('Figure 6 output already exists: %s' % output_dir)
    bundle = load_evidence(capture_dir, load_crop_manifest(crop_manifest))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix='.%s.staging.' % output_dir.name, dir=str(output_dir.parent)))
    published = False
    names = artifact_names(include_alt)
    try:
        _write_bundle(staging, bundle, include_alt)
        actual = tuple(sorted(path.name for path in staging.iterdir()))
        if actual != tuple(sorted(names)):
            raise ExportBundleError(tuple(sorted(names)), actual)
        _publish(staging, output_dir, overwrite)
        published = True
        return ExportResult(output_dir, names)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
