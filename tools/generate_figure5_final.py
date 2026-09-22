#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy>=1.24", "scipy>=1.10", "matplotlib>=3.7"]
# ///
# Run from the repository root with: python3 tools/generate_figure5_final.py --help
"""Export the fixed-token final Figure 5 plate and honest offline Figure 6 inputs."""
import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Sequence

from figure5_utils.candidates import CandidateEvaluation, evaluate_frame
from figure5_utils.final_artifacts import export_final_artifacts, publish_final_artifacts
from figure5_utils.final_rendering import FinalRenderRow
from figure5_utils.final_selection import (
    FINAL_SAMPLE_TOKENS, filter_predictions, select_final_rows,
)
from figure5_utils.input_types import PointSource, ResultPaths
from figure5_utils.io import load_artifacts, load_frame
from figure5_utils.matching import match_frame
from figure5_utils.policies import MatchingPolicy
from figure5_utils.domain import DistanceMeters, SampleToken


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
    overwrite: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('infos', 'data-root', 'source-result', 'codemerge-result',
                 'refuse-result', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--nuscenes-tables', type=Path)
    parser.add_argument('--sparse-points-dir', type=Path)
    parser.add_argument('--score-threshold', type=float, default=0.1)
    parser.add_argument('--max-center-distance', type=float, default=2.0)
    parser.add_argument('--min-bev-iou', type=float, default=0.0)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def _validate(parser: argparse.ArgumentParser, args: Arguments) -> None:
    if not math.isfinite(args.score_threshold) or not 0.0 <= args.score_threshold <= 1.0:
        parser.error('--score-threshold must be finite and in [0, 1]')
    if not math.isfinite(args.max_center_distance) or args.max_center_distance < 0.0:
        parser.error('--max-center-distance must be finite and nonnegative')
    if not math.isfinite(args.min_bev_iou) or not 0.0 <= args.min_bev_iou <= 1.0:
        parser.error('--min-bev-iou must be finite and in [0, 1]')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv, namespace=Arguments())
    _validate(parser, args)
    dataset = load_artifacts(
        ResultPaths(args.source_result, args.codemerge_result, args.refuse_result),
        args.infos,
        args.nuscenes_tables,
    )
    matching_policy = MatchingPolicy(
        DistanceMeters(args.max_center_distance), args.min_bev_iou,
    )
    candidates: Dict[SampleToken, CandidateEvaluation] = {}
    for token in FINAL_SAMPLE_TOKENS:
        if token not in dataset.frames:
            continue
        frame = filter_predictions(dataset.frames[token], args.score_threshold)
        candidates[token] = evaluate_frame(frame, match_frame(frame, matching_policy))
    selections = select_final_rows(candidates)
    point_source = PointSource(
        args.data_root,
        sparse_directory=args.sparse_points_dir,
        sparsity_mode='random_keep',
        sparsity_severity=5,
    )
    rows = tuple(FinalRenderRow(
        selection,
        load_frame(dataset, selection.frame.sample_token, point_source).sparse_points,
    ) for selection in selections)
    output = args.output_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.figure5-final-', dir=str(output.parent)) as directory:
        staging = Path(directory)
        export_final_artifacts(
            rows, staging, supplied_points=args.sparse_points_dir is not None,
        )
        publish_final_artifacts(staging, output, args.overwrite)
    print('Final Figure 5 and offline Figure 6 artifacts written to %s' % output, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
