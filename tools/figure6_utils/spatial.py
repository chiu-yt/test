"""Calibrated BEV map reduction and cropping in canonical LiDAR coordinates."""

import math
from typing import Optional, Tuple

import numpy as np

from .contracts import Crop


SpatialImage = Tuple[np.ndarray, Tuple[float, float, float, float]]


def spatial_map(values: np.ndarray) -> Optional[np.ndarray]:
    array = np.asarray(values)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        array = np.nanmean(array.astype(np.float64), axis=0)
    if array.ndim != 2 or not array.size:
        return None
    return array.astype(np.float64)


def crop_bev_map(values: np.ndarray, full: Optional[Crop], crop: Crop) -> Optional[SpatialImage]:
    """Crop a [y, x] BEV map and transpose it for horizontal-y/vertical-x display."""
    spatial = spatial_map(values)
    if spatial is None or full is None:
        return None
    full_x_min, full_y_min, full_x_max, full_y_max = full
    crop_x_min = max(crop[0], full_x_min)
    crop_y_min = max(crop[1], full_y_min)
    crop_x_max = min(crop[2], full_x_max)
    crop_y_max = min(crop[3], full_y_max)
    if crop_x_min >= crop_x_max or crop_y_min >= crop_y_max:
        return None
    height, width = spatial.shape
    x_scale = width / (full_x_max - full_x_min)
    y_scale = height / (full_y_max - full_y_min)
    x_start = max(0, int(math.floor((crop_x_min - full_x_min) * x_scale)))
    x_stop = min(width, int(math.ceil((crop_x_max - full_x_min) * x_scale)))
    y_start = max(0, int(math.floor((crop_y_min - full_y_min) * y_scale)))
    y_stop = min(height, int(math.ceil((crop_y_max - full_y_min) * y_scale)))
    if x_start >= x_stop or y_start >= y_stop:
        return None
    extent = (
        full_y_min + y_start / y_scale, full_y_min + y_stop / y_scale,
        full_x_min + x_start / x_scale, full_x_min + x_stop / x_scale,
    )
    return spatial[y_start:y_stop, x_start:x_stop].T, extent
