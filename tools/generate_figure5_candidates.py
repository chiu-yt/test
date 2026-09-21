#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy>=1.24", "scipy>=1.10", "matplotlib>=3.7"]
# ///
# How to run (from repository root):
# python3 tools/generate_figure5_candidates.py --help
# uv run tools/generate_figure5_candidates.py --infos "$INFOS" --data-root "$DATA_ROOT" \
#   --source-result "$SOURCE_RESULT" --codemerge-result "$CODEMERGE_RESULT" \
#   --refuse-result "$REFUSE_RESULT" --output-dir "$FIGURE5_OUTPUT"
"""Build a reviewable Figure 5 candidate package from trusted saved predictions."""
import argparse
from dataclasses import replace
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Optional, Sequence

from figure5_utils.artifacts import export_tables, publish_package, render_package
from figure5_utils.candidates import evaluate_frame, rank_candidates
from figure5_utils.domain import DistanceMeters, FrameRecord
from figure5_utils.input_types import ArtifactFormatError, PointSource, ResultPaths
from figure5_utils.io import load_artifacts
from figure5_utils.matching import match_frame
from figure5_utils.policies import MatchingPolicy
from figure5_utils.summary import write_summary


class Arguments(argparse.Namespace):
    infos: Path
    data_root: Path
    source_result: Path
    codemerge_result: Path
    refuse_result: Path
    output_dir: Path
    nuscenes_tables: Optional[Path]
    sparse_points_dir: Optional[Path]
    score_threshold: float
    max_center_distance: float
    min_bev_iou: float
    sparsity_mode: str
    sparsity_severity: int
    limit: Optional[int]
    overwrite: bool


def filter_predictions(frame: FrameRecord, threshold: float) -> FrameRecord:
    """Apply one inclusive score cut to predictions; GT is preserved by identity."""
    return replace(frame,
                   source_only=tuple(item for item in frame.source_only if item.score >= threshold),
                   codemerge=tuple(item for item in frame.codemerge if item.score >= threshold),
                   refuse_tta=tuple(item for item in frame.refuse_tta if item.score >= threshold))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('infos', 'data-root', 'source-result', 'codemerge-result', 'refuse-result', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--nuscenes-tables', type=Path)
    parser.add_argument('--sparse-points-dir', type=Path)
    parser.add_argument('--score-threshold', type=float, default=0.1)
    parser.add_argument('--max-center-distance', type=float, default=2.0)
    parser.add_argument('--min-bev-iou', type=float, default=0.0)
    parser.add_argument('--sparsity-mode', choices=('density_dec_global', 'random_keep'),
                        default='density_dec_global')
    parser.add_argument('--sparsity-severity', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--limit', type=int, help='Smoke only: first N tokens in lexicographic order')
    parser.add_argument('--overwrite', action='store_true', help='Replace only existing Figure 5 package artifacts')
    args = parser.parse_args(argv, namespace=Arguments())
    if not math.isfinite(args.score_threshold) or not 0 <= args.score_threshold <= 1:
        parser.error('--score-threshold must be finite and in [0, 1]')
    if not math.isfinite(args.max_center_distance) or args.max_center_distance < 0:
        parser.error('--max-center-distance must be finite and nonnegative')
    if not math.isfinite(args.min_bev_iou) or not 0 <= args.min_bev_iou <= 1:
        parser.error('--min-bev-iou must be finite and in [0, 1]')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not args.overwrite:
        parser.error('output is occupied; use a fresh directory or --overwrite')
    policy = MatchingPolicy(DistanceMeters(args.max_center_distance), args.min_bev_iou)
    paths = ResultPaths(args.source_result, args.codemerge_result, args.refuse_result)
    print('Loading and validating token-aligned artifacts', flush=True)
    dataset = load_artifacts(paths, args.infos, args.nuscenes_tables)
    tokens = sorted(dataset.frames)[:args.limit]
    candidates = []
    for index, token in enumerate(tokens, 1):
        if re.fullmatch(r'[A-Za-z0-9_-]+', token) is None:
            raise ArtifactFormatError(token, 'sample token must be filename-safe')
        frame = filter_predictions(dataset.frames[token], args.score_threshold)
        candidates.append(evaluate_frame(frame, match_frame(frame, policy)))
        if index % 100 == 0 or index == len(tokens):
            print('Scored %d/%d frames' % (index, len(tokens)), flush=True)
    ranked = rank_candidates(candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.figure5-', dir=str(output.parent)) as directory:
        staging = Path(directory)
        export_tables(ranked, staging)
        point_source = PointSource(
            args.data_root,
            sparse_directory=args.sparse_points_dir,
            sparsity_mode=args.sparsity_mode,
            sparsity_severity=args.sparsity_severity,
        )
        render_package(ranked[:60], (dataset, point_source), staging)
        provenance = [
            'Infos: `%s`' % args.infos, 'Source: `%s`' % paths.source_only,
            'CodeMerge: `%s`' % paths.codemerge, 'ReFuse: `%s`' % paths.refuse_tta,
            'Scored %d of %d available tokens; limit=%s.' % (len(ranked), len(dataset.frames), args.limit),
            'Prediction score >= %g; class-aware center distance <= %g m; BEV IoU >= %g.' % (
                args.score_threshold, args.max_center_distance, args.min_bev_iou),
            'Points: exact supplied S5 arrays from `%s` (provenance is caller responsibility).' % args.sparse_points_dir
            if args.sparse_points_dir is not None else
            'Points: deterministic same-protocol reconstruction, not exact saved evaluation points; '
            'five sweeps, seed 1024, %s S%d.' % (
                args.sparsity_mode, args.sparsity_severity),
        ]
        write_summary(ranked, staging, provenance)
        publish_package(staging, output, args.overwrite)
    print('Figure 5 candidate package written to %s' % output, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
