"""Freeze Figure 5's selected geometry for aligned downstream figures."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Sequence, Tuple, TypedDict

from .domain import ImageRoi, SampleToken
from .final_rendering import FinalRenderRow, horizontal_crop


CROP_MANIFEST_NAME: Final[str] = 'figure5_crop_manifest.json'


class CalloutGeometry(TypedDict):
    roi: ImageRoi
    kind: str
    class_name: str
    distance_m: float


class RowGeometry(TypedDict):
    row_number: int
    sample_token: SampleToken
    purpose: str
    square_crop: ImageRoi
    horizontal_crop: ImageRoi
    callouts: Tuple[CalloutGeometry, ...]


class CoordinateConventions(TypedDict):
    canonical_frame: str
    units: str
    horizontal_axis: str
    vertical_axis: str
    crop_tuple_order: Tuple[str, ...]
    roi_tuple_order: Tuple[str, ...]


class LayoutIdentities(TypedDict):
    square: str
    horizontal: str


class CropManifest(TypedDict):
    schema_version: int
    coordinate_conventions: CoordinateConventions
    layouts: LayoutIdentities
    rows: Tuple[RowGeometry, ...]


def write_crop_manifest(rows: Sequence[FinalRenderRow], output: Path,
                        row_purposes: Sequence[str]) -> Path:
    """Atomically serialize the same selected rows and crops used by the renderer."""
    manifest = CropManifest(
        schema_version=1,
        coordinate_conventions=CoordinateConventions(
            canonical_frame='lidar', units='m', horizontal_axis='y', vertical_axis='x',
            crop_tuple_order=('x_min', 'y_min', 'x_max', 'y_max'),
            roi_tuple_order=('x_min', 'y_min', 'x_max', 'y_max'),
        ),
        layouts=LayoutIdentities(square='figure5_final', horizontal='figure5_horizontal'),
        rows=tuple(RowGeometry(
            row_number=row.selection.row_number,
            sample_token=row.selection.frame.sample_token,
            purpose=row_purposes[row.selection.row_number - 1],
            square_crop=row.selection.crop,
            horizontal_crop=horizontal_crop(row),
            callouts=tuple(CalloutGeometry(
                roi=callout.roi, kind=callout.kind.value,
                class_name=callout.class_name.value, distance_m=callout.distance_m,
            ) for callout in row.selection.callouts),
        ) for row in rows),
    )
    path = output / CROP_MANIFEST_NAME
    temporary_file = NamedTemporaryFile(
        mode='w', encoding='utf-8', prefix='.' + CROP_MANIFEST_NAME + '.',
        suffix='.tmp', dir=str(output), delete=False,
    )
    temporary = Path(temporary_file.name)
    try:
        with temporary_file as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
