import json
from pathlib import Path
from typing import Dict, Final, Mapping, Sequence, Tuple, TypedDict

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .domain import SampleToken
from .final_rendering import (
    FINAL_DPI, FinalRenderRow, render_density, render_final_detections,
    render_final_plate, render_final_row,
)
from .final_selection import FINAL_SAMPLE_TOKENS


FIGURE5_FILES: Final[Tuple[str, ...]] = (
    'figure5_final_clean.png',
    'figure5_final_clean.pdf',
    'figure5_final_callout.png',
    'figure5_final_callout.pdf',
    'figure5_row1.png',
    'figure5_row2.png',
    'figure5_row3.png',
    'figure5_refine_summary.md',
)
MANIFEST_NAME: Final[str] = 'figure6_status_manifest.json'
RUNTIME_STEMS: Final[Tuple[str, ...]] = ('reliability', 'rgplm', 'sgdfa')
ROW_PURPOSES: Final[Tuple[str, ...]] = (
    'far-range recovery',
    'small-object recovery',
    'localization / false-positive improvement',
)


class ArtifactStatus(TypedDict):
    status: str
    content: str


class PointProvenance(TypedDict):
    mode: str
    exact_dataloader_replay: bool
    disclosure: str


class StatusManifest(TypedDict):
    schema_version: int
    point_provenance: PointProvenance
    artifacts: Mapping[str, ArtifactStatus]


def artifact_names(tokens: Sequence[SampleToken]) -> Tuple[str, ...]:
    generated = tuple(
        name
        for token in tokens
        for name in ('density_%s.png' % token, 'finaldet_%s.png' % token)
    )
    runtime = tuple(
        '%s_%s.png' % (stem, token)
        for token in tokens
        for stem in RUNTIME_STEMS
    )
    return FIGURE5_FILES + (MANIFEST_NAME,) + generated + runtime


def _manifest(tokens: Sequence[SampleToken], supplied_points: bool) -> StatusManifest:
    artifacts: Dict[str, ArtifactStatus] = {}
    for token in tokens:
        artifacts['density_%s.png' % token] = {
            'status': 'generated',
            'content': 'offline_sparse_point_density',
        }
        artifacts['finaldet_%s.png' % token] = {
            'status': 'generated',
            'content': 'refuse_final_boxes_from_result_pkl',
        }
        for stem, content in zip(
                RUNTIME_STEMS, ('spcra_reliability', 'rg_plm', 'sg_dfa')):
            artifacts['%s_%s.png' % (stem, token)] = {
                'status': 'requires_runtime_capture',
                'content': content,
            }
    provenance = PointProvenance(
        mode=('supplied_sparse_npy_unverified' if supplied_points else
              'deterministic_random_keep_s5_reconstruction'),
        exact_dataloader_replay=False,
        disclosure=(
            'Token-named sparse NPY supplied; correspondence to the evaluation run is unverified.'
            if supplied_points else
            'Deterministic random_keep S5 reconstruction; not an exact DataLoader replay.'
        ),
    )
    return StatusManifest(schema_version=1, point_provenance=provenance, artifacts=artifacts)


def write_status_manifest(output: Path, tokens: Sequence[SampleToken],
                          supplied_points: bool) -> Path:
    path = output / MANIFEST_NAME
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(_manifest(tokens, supplied_points), stream, indent=2, sort_keys=True)
        stream.write('\n')
    temporary.replace(path)
    return path


def _save_png(figure: Figure, path: Path) -> None:
    figure.savefig(
        str(path), dpi=FINAL_DPI, facecolor='white',
        metadata={'Software': 'OpenPCDet Figure 5 exporter'},
    )


def _save_pdf(figure: Figure, path: Path) -> None:
    figure.savefig(
        str(path), facecolor='white',
        metadata={
            'Creator': 'OpenPCDet Figure 5 exporter',
            'CreationDate': None,
            'ModDate': None,
        },
    )


def _save_single_png(figure: Figure, path: Path) -> None:
    try:
        _save_png(figure, path)
    finally:
        plt.close(figure)


def _write_summary(rows: Sequence[FinalRenderRow], output: Path,
                   supplied_points: bool) -> None:
    point_text = (
        'Token-named sparse NPY arrays were supplied, but correspondence to the evaluation '
        'DataLoader realization is unverified.' if supplied_points else
        'Points use deterministic project-defined random_keep S5 reconstruction. '
        'This is not an exact DataLoader replay.'
    )
    lines = [
        '# Figure 5 final refinement summary',
        '',
        '- The final plate uses a fixed 3 x 4 GT / Source-only / CodeMerge / ReFuse-TTA layout.',
        '- `figure5_row1.png`, `figure5_row2.png`, and `figure5_row3.png` are callout versions.',
        '- Clean and callout plates reuse the same sparse points and one square crop per row.',
        '- ' + point_text,
        '',
        '## Fixed rows',
        '',
    ]
    for row in rows:
        selection = row.selection
        lines.append('- Row %d `%s`: %s; crop=%s.' % (
            selection.row_number,
            selection.frame.sample_token,
            ROW_PURPOSES[selection.row_number - 1],
            tuple(round(value, 3) for value in selection.crop),
        ))
        for callout_index, callout in enumerate(selection.callouts, 1):
            lines.append('  - Callout %d: type=%s; class=%s; distance_m=%.2f; roi=%s.' % (
                callout_index, callout.kind.value, callout.class_name.value,
                callout.distance_m, tuple(round(value, 3) for value in callout.roi),
            ))
    lines.extend([
        '',
        '## Figure 6 intermediate files',
        '',
    ])
    for row in rows:
        token = row.selection.frame.sample_token
        lines.extend([
            '- `%s`:' % token,
            '  - `density_%s.png`: generated offline sparse-point density.' % token,
            '  - `reliability_%s.png`: requires SPCRA runtime capture; not generated.' % token,
            '  - `rgplm_%s.png`: requires RG-PLM runtime capture; not generated.' % token,
            '  - `sgdfa_%s.png`: requires SG-DFA runtime capture; not generated.' % token,
            '  - `finaldet_%s.png`: generated from ReFuse boxes in `result.pkl`.' % token,
        ])
    lines.extend([
        '',
        'Runtime-only SPCRA, RG-PLM, and SG-DFA images are not reconstructed from aggregate logs.',
        '',
    ])
    (output / FIGURE5_FILES[-1]).write_text('\n'.join(lines), encoding='utf-8')


def export_final_artifacts(rows: Sequence[FinalRenderRow], output: Path,
                           supplied_points: bool) -> None:
    clean = render_final_plate(rows, show_callouts=False)
    try:
        _save_png(clean, output / FIGURE5_FILES[0])
        _save_pdf(clean, output / FIGURE5_FILES[1])
    finally:
        plt.close(clean)
    callout = render_final_plate(rows, show_callouts=True)
    try:
        _save_png(callout, output / FIGURE5_FILES[2])
        _save_pdf(callout, output / FIGURE5_FILES[3])
    finally:
        plt.close(callout)
    for row in rows:
        token = row.selection.frame.sample_token
        _save_single_png(render_final_row(row), output / ('figure5_row%d.png' % row.selection.row_number))
        _save_single_png(render_density(row), output / ('density_%s.png' % token))
        _save_single_png(render_final_detections(row), output / ('finaldet_%s.png' % token))
    _write_summary(rows, output, supplied_points)
    write_status_manifest(
        output, tuple(row.selection.frame.sample_token for row in rows), supplied_points,
    )


def publish_final_artifacts(staging: Path, output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    runtime_names = tuple(
        '%s_%s.png' % (stem, token)
        for token in FINAL_SAMPLE_TOKENS
        for stem in RUNTIME_STEMS
    )
    runtime_collisions = tuple(output / name for name in runtime_names if (output / name).exists())
    if runtime_collisions:
        raise FileExistsError(
            'runtime capture exists; publish to a fresh output directory: %s' %
            runtime_collisions[0].name
        )
    collisions = tuple(path for path in staging.iterdir() if (output / path.name).exists())
    if collisions and not overwrite:
        raise FileExistsError('final Figure 5 output exists; use --overwrite: %s' % collisions[0].name)
    for path in staging.iterdir():
        path.replace(output / path.name)
