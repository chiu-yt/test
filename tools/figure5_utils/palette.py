from enum import Enum, unique
from types import MappingProxyType
from typing import Final, Mapping, Tuple


Color = Tuple[int, int, int]


@unique
class DetectionClass(str, Enum):
    CAR = 'car'
    TRUCK = 'truck'
    CONSTRUCTION_VEHICLE = 'construction_vehicle'
    BUS = 'bus'
    TRAILER = 'trailer'
    BARRIER = 'barrier'
    MOTORCYCLE = 'motorcycle'
    BICYCLE = 'bicycle'
    PEDESTRIAN = 'pedestrian'
    TRAFFIC_CONE = 'traffic_cone'


NUSCENES_DETECTION_CLASSES: Final[Tuple[DetectionClass, ...]] = (
    DetectionClass.CAR,
    DetectionClass.TRUCK,
    DetectionClass.CONSTRUCTION_VEHICLE,
    DetectionClass.BUS,
    DetectionClass.TRAILER,
    DetectionClass.BARRIER,
    DetectionClass.MOTORCYCLE,
    DetectionClass.BICYCLE,
    DetectionClass.PEDESTRIAN,
    DetectionClass.TRAFFIC_CONE,
)

NUSCENES_DETECTION_PALETTE: Final[Mapping[DetectionClass, Color]] = MappingProxyType({
    DetectionClass.CAR: (255, 158, 0),
    DetectionClass.TRUCK: (255, 99, 71),
    DetectionClass.CONSTRUCTION_VEHICLE: (233, 150, 70),
    DetectionClass.BUS: (255, 127, 80),
    DetectionClass.TRAILER: (255, 140, 0),
    DetectionClass.BARRIER: (112, 128, 144),
    DetectionClass.MOTORCYCLE: (255, 61, 99),
    DetectionClass.BICYCLE: (220, 20, 60),
    DetectionClass.PEDESTRIAN: (0, 0, 230),
    DetectionClass.TRAFFIC_CONE: (47, 79, 79),
})
