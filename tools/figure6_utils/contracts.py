"""Validated Figure 5 crop contracts used unchanged by Figure 6."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence, Tuple

from pcdet.utils.figure6_schema import FIXED_TOKENS


Crop = Tuple[float, float, float, float]


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class CropRow:
    token: str
    row_number: int
    purpose: str
    crop: Crop


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class CropManifestError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _parse_crop(values: Sequence[float], token: str) -> Crop:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise CropManifestError('horizontal_crop for %s must contain four values' % token)
    crop = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in crop):
        raise CropManifestError('horizontal_crop for %s must be finite' % token)
    if crop[0] >= crop[2] or crop[1] >= crop[3]:
        raise CropManifestError('horizontal_crop for %s must be nondegenerate' % token)
    return crop[0], crop[1], crop[2], crop[3]


def load_crop_manifest(path: Path) -> Tuple[CropRow, ...]:
    """Parse the Figure 5 manifest and preserve its horizontal crop floats."""
    try:
        with path.open(encoding='utf-8') as stream:
            payload = json.load(stream)
        rows = payload['rows']
        conventions = payload['coordinate_conventions']
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CropManifestError('%s: %s' % (path, error)) from error
    tokens = tuple(row.get('sample_token') for row in rows)
    if tokens != FIXED_TOKENS:
        raise CropManifestError(
            'crop manifest tokens must equal canonical FIXED_TOKENS in canonical order'
        )
    expected = ('lidar', 'm', 'y', 'x', ['x_min', 'y_min', 'x_max', 'y_max'])
    actual = (
        conventions.get('canonical_frame'), conventions.get('units'),
        conventions.get('horizontal_axis'), conventions.get('vertical_axis'),
        conventions.get('crop_tuple_order'),
    )
    if actual != expected:
        raise CropManifestError('crop manifest coordinate conventions are incompatible')
    parsed = []
    for expected_number, row in enumerate(rows, start=1):
        if row.get('row_number') != expected_number:
            raise CropManifestError('crop manifest row numbers must be canonical')
        purpose = row.get('purpose')
        if not isinstance(purpose, str) or not purpose:
            raise CropManifestError('crop manifest purposes must be nonempty text')
        token = row['sample_token']
        parsed.append(CropRow(
            token, expected_number, purpose, _parse_crop(row.get('horizontal_crop'), token),
        ))
    return tuple(parsed)
