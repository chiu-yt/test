"""Figure 6 visual tokens aligned with the Figure 5 nuScenes palette."""

from typing import Final, Tuple


CLASS_NAMES: Final[Tuple[str, ...]] = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
)
CLASS_COLORS: Final[Tuple[Tuple[int, int, int], ...]] = (
    (255, 158, 0),
    (255, 99, 71),
    (233, 150, 70),
    (255, 127, 80),
    (255, 140, 0),
    (112, 128, 144),
    (255, 61, 99),
    (220, 20, 60),
    (0, 0, 230),
    (47, 79, 79),
)
