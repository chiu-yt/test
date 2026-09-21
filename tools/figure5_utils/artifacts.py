"""Exact Figure 5 artifact names, bounded rendering, and package publication."""
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Final, Sequence, Tuple

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .candidates import CandidateCsvRecord, CandidateEvaluation, bucket_candidates
from .input_types import ArtifactDataset, PointSource
from .io import load_frame
from .rendering import (CONTACT_SHEET_CANDIDATES, HD_DPI, STANDARD_DPI, PanelSpec,
                        render_comparison_panel, render_contact_sheets)


TABLE_FILES: Final = ('figure5_candidates_top100.csv', 'figure5_candidates_by_type.csv',
                     'figure5_callouts.json', 'figure5_candidate_summary.md')
PANEL_DIRECTORIES: Final = ('qual_panels_top60', 'qual_panels_top30_hd')
CONTACT_SHEET_NAME: Final = re.compile(r'contact_sheet_page[0-9]+\.png')


def export_tables(ranked: Sequence[CandidateEvaluation], output: Path) -> None:
    fields = ('rank',) + tuple(CandidateCsvRecord.__annotations__)
    ranks = {candidate.sample_token: rank for rank, candidate in enumerate(ranked, 1)}
    with (output / TABLE_FILES[0]).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, candidate in enumerate(ranked[:100], 1):
            writer.writerow(dict(candidate.to_csv_record(), rank=rank))
    buckets = bucket_candidates(ranked)
    with (output / TABLE_FILES[1]).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ('type_rank', 'theme'))
        writer.writeheader()
        for kind, candidates in buckets.items():
            for type_rank, candidate in enumerate(candidates[:20], 1):
                writer.writerow(dict(candidate.to_csv_record(), rank=ranks[candidate.sample_token],
                                     type_rank=type_rank, theme=kind.value))
    included = {candidate.sample_token: candidate for candidate in ranked[:100]}
    for candidates in buckets.values():
        included.update((candidate.sample_token, candidate) for candidate in candidates[:20])
    payload = {token: [{'type': item.kind.value, 'class': item.class_name.value,
                        'distance': item.distance_m, 'roi': item.roi}
                       for item in candidate.callouts] for token, candidate in included.items()}
    with (output / TABLE_FILES[2]).open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_figure(figure: Figure, path: Path, dpi: int) -> None:
    try:
        figure.savefig(str(path), dpi=dpi, facecolor='white')
    finally:
        plt.close(figure)


def render_package(candidates: Sequence[CandidateEvaluation],
                   inputs: Tuple[ArtifactDataset, PointSource], output: Path) -> None:
    """Hold at most twelve point arrays and one contact-sheet figure at a time."""
    standard, hd = (output / name for name in PANEL_DIRECTORIES)
    standard.mkdir()
    hd.mkdir()
    dataset, source = inputs
    for start in range(0, len(candidates), CONTACT_SHEET_CANDIDATES):
        panels = []
        for rank, candidate in enumerate(candidates[start:start + CONTACT_SHEET_CANDIDATES], start + 1):
            loaded = load_frame(dataset, candidate.sample_token, source)
            panel = PanelSpec(candidate.frame, loaded.sparse_points, rank, candidate.callouts)
            panels.append(panel)
            stem = '%d_%s' % (rank, candidate.sample_token)
            save_figure(render_comparison_panel(panel), standard / (stem + '.png'), STANDARD_DPI)
            if rank <= 30:
                save_figure(render_comparison_panel(panel), hd / (stem + '.png'), HD_DPI)
                save_figure(render_comparison_panel(panel, True), hd / (stem + '_callout.png'), HD_DPI)
            print('Rendered %d/%d: %s' % (rank, len(candidates), candidate.sample_token), flush=True)
        page = start // CONTACT_SHEET_CANDIDATES + 1
        for figure in render_contact_sheets(panels):
            save_figure(figure, output / ('contact_sheet_page%d.png' % page), STANDARD_DPI)


def publish_package(staging: Path, output: Path, overwrite: bool) -> None:
    """Replace owned artifacts only, after the entire new package succeeds."""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not overwrite:
        raise FileExistsError('output is occupied; use --overwrite: %s' % output)
    owned = [output / name for name in TABLE_FILES + PANEL_DIRECTORIES]
    owned.extend(path for path in output.iterdir()
                 if CONTACT_SHEET_NAME.fullmatch(path.name) is not None)
    for path in owned:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    for path in staging.iterdir():
        path.replace(output / path.name)
