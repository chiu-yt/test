"""Render Figure 6 from integrity-checked runtime captures and Figure 5 crops."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import _init_path  # noqa: F401

from figure6_utils.export import export_figure6


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render the five-column Figure 6 runtime evidence chain.',
    )
    parser.add_argument('--capture_dir', type=Path, required=True)
    parser.add_argument('--crop_manifest', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--include_alt', action='store_true')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = export_figure6(
        args.capture_dir, args.crop_manifest, args.output_dir, args.overwrite,
        args.include_alt,
    )
    print('Published %d Figure 6 artifacts to %s' % (len(result.artifacts), result.output_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
