"""Atomic Figure 6 bundle publication and machine-readable provenance."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Dict, Final, Mapping, Sequence, Tuple
from uuid import uuid4

from pcdet.utils.figure6_schema import FIXED_TOKENS

from .contracts import load_crop_manifest
from .loading import EvidenceBundle, TokenEvidence, load_evidence
from .rendering import save_plate, save_stage_png


STATUS_NAME: Final[str] = 'figure6_status_manifest.json'
SUMMARY_NAME: Final[str] = 'figure6_summary.md'
DRAFT_NAMES: Final[Tuple[str, ...]] = (
    'figure6_draft_2row.png', 'figure6_draft_2row.pdf',
    'figure6_draft_3row.png', 'figure6_draft_3row.pdf',
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


def artifact_names(tokens: Sequence[str]) -> Tuple[str, ...]:
    panels = tuple(
        '%s_%s.png' % (slug, token)
        for token in tokens
        for slug in ('density', 'reliability', 'rgplm', 'sgdfa', 'finaldet')
    )
    return panels + DRAFT_NAMES + (STATUS_NAME, SUMMARY_NAME)


def _stage_payload(row: TokenEvidence) -> Tuple[Mapping[str, str], ...]:
    return tuple({
        'column': stage.title,
        'artifact': '%s_%s.png' % (stage.slug, row.token),
        'runtime_state': stage.runtime_state.value,
        'owner': stage.owner,
        'detail': stage.detail,
        'selected_stage': stage.selected_stage,
        'selected_source': stage.selected_source,
        'array_names': sorted(stage.arrays),
        'spatial_extent': (None if stage.spatial_extent is None else list(stage.spatial_extent)),
    } for stage in row.stages)


def _status(bundle: EvidenceBundle) -> Dict:
    rows = []
    for row in bundle.rows:
        rows.append({
            'token': row.token,
            'row_number': row.row_number,
            'purpose': row.purpose,
            'crop': list(row.crop),
            'occurrence_path': None if row.occurrence_path is None else str(row.occurrence_path),
            'selection_key': None if row.selection_key is None else list(row.selection_key),
            'alternative_count': row.alternative_count,
            'protocol': dict(row.protocol),
            'stages': list(_stage_payload(row)),
        })
    return {
        'schema_version': 1,
        'figure': 6,
        'columns': [stage.title for stage in bundle.rows[0].stages],
        'canonical_tokens': list(FIXED_TOKENS),
        'rows': rows,
        'record_issues': [
            {'path': str(issue.path), 'detail': issue.detail} for issue in bundle.issues
        ],
        'drafts': {
            '2row': {'tokens': list(FIXED_TOKENS[:2]),
                     'artifacts': list(DRAFT_NAMES[:2])},
            '3row': {'tokens': list(FIXED_TOKENS),
                     'artifacts': list(DRAFT_NAMES[2:])},
        },
    }


def _summary(bundle: EvidenceBundle) -> str:
    lines = [
        '# Figure 6 export summary', '',
        'Five aligned runtime stages are rendered over the exact Figure 5 horizontal crops.', '',
        '| Row | Token | Runtime stage states |',
        '|---:|---|---|',
    ]
    for row in bundle.rows:
        states = ', '.join('%s: `%s`' % (stage.title, stage.runtime_state.value)
                           for stage in row.stages)
        lines.append('| %d | `%s` | %s |' % (row.row_number, row.token, states))
    lines.extend(['', 'Corrupt or incomplete runtime records reported: **%d**.' % len(bundle.issues), ''])
    return '\n'.join(lines)


def _write_bundle(directory: Path, bundle: EvidenceBundle) -> None:
    for row in bundle.rows:
        for stage in row.stages:
            save_stage_png(stage, row.crop, directory / ('%s_%s.png' % (stage.slug, row.token)))
    save_plate(
        bundle.rows[:2], directory / DRAFT_NAMES[0], directory / DRAFT_NAMES[1],
    )
    save_plate(
        bundle.rows, directory / DRAFT_NAMES[2], directory / DRAFT_NAMES[3],
    )
    with (directory / STATUS_NAME).open('w', encoding='utf-8') as stream:
        json.dump(_status(bundle), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    (directory / SUMMARY_NAME).write_text(_summary(bundle), encoding='utf-8')


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
                   overwrite: bool = False) -> ExportResult:
    if output_dir.exists() and not overwrite:
        raise FileExistsError('Figure 6 output already exists: %s' % output_dir)
    crops = load_crop_manifest(crop_manifest)
    bundle = load_evidence(capture_dir, crops)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix='.%s.staging.' % output_dir.name, dir=str(output_dir.parent)))
    published = False
    try:
        _write_bundle(staging, bundle)
        names = artifact_names(FIXED_TOKENS)
        actual = tuple(sorted(path.name for path in staging.iterdir()))
        if actual != tuple(sorted(names)):
            raise ExportBundleError(tuple(sorted(names)), actual)
        _publish(staging, output_dir, overwrite)
        published = True
        return ExportResult(output_dir, names)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
