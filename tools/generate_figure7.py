import argparse
from pathlib import Path
from typing import Optional, Sequence

import _init_path  # noqa: F401

from figure7_utils.publication import publish_figure7


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render the evidence-driven two-row Figure 7 publication plate.',
    )
    parser.add_argument('--capture_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--point_cloud_range', type=float, nargs=4,
                        metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
                        default=(-54.0, -54.0, 54.0, 54.0))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = publish_figure7(
        args.capture_dir, args.output_dir, args.overwrite, tuple(args.point_cloud_range),
    )
    print('Published %d Figure 7 artifacts to %s' % (len(result.artifacts), result.output_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
