from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray


AUGMENTATION_LAW_ID: Final = 'spcra_legacy_bernoulli_world_v1'


class AugmentationConfigurationError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


class WorldOperator(Protocol):
    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class FlipOperator:
    axis: str

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        matrix = np.eye(4, dtype=np.float64)
        if bool(rng.choice([False, True], p=[.5, .5])):
            coordinate = 1 if self.axis == 'x' else 0
            matrix[coordinate, coordinate] = -1.
        return matrix


@dataclass(frozen=True, slots=True)
class RotationOperator:
    minimum: float
    maximum: float

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        angle = rng.uniform(self.minimum, self.maximum)
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = ((cosine, -sine, 0.), (sine, cosine, 0.), (0., 0., 1.))
        return matrix


@dataclass(frozen=True, slots=True)
class ScalingOperator:
    minimum: float
    maximum: float

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        matrix = np.eye(4, dtype=np.float64)
        if self.maximum - self.minimum >= 1e-3:
            matrix[:3, :3] *= rng.uniform(self.minimum, self.maximum)
        return matrix


@dataclass(frozen=True, slots=True)
class TranslationOperator:
    standard_deviation: tuple[float, float, float]

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = rng.normal(0., self.standard_deviation)
        return matrix


def _finite_values(values, expected_size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected_size or not np.isfinite(result).all():
        raise AugmentationConfigurationError(f'{name} must contain {expected_size} finite values')
    return result


def _flip_operators(config) -> tuple[WorldOperator, ...]:
    axes = tuple(config['ALONG_AXIS_LIST'])
    if any(axis not in ('x', 'y') for axis in axes):
        raise AugmentationConfigurationError('random_world_flip supports only x and y axes')
    return tuple(FlipOperator(axis) for axis in axes)


def _rotation_operators(config) -> tuple[WorldOperator, ...]:
    configured = config['WORLD_ROT_ANGLE']
    configured_array = np.asarray(configured, dtype=np.float64)
    scalar = float(configured_array.reshape(-1)[0])
    values = (-scalar, scalar) if configured_array.ndim == 0 else configured_array
    minimum, maximum = _finite_values(values, 2, 'WORLD_ROT_ANGLE')
    if minimum > maximum:
        raise AugmentationConfigurationError('WORLD_ROT_ANGLE must be ordered')
    return (RotationOperator(minimum, maximum),)


def _scaling_operators(config) -> tuple[WorldOperator, ...]:
    minimum, maximum = _finite_values(config['WORLD_SCALE_RANGE'], 2, 'WORLD_SCALE_RANGE')
    if minimum <= 0. or minimum > maximum:
        raise AugmentationConfigurationError('WORLD_SCALE_RANGE must be positive and ordered')
    return (ScalingOperator(minimum, maximum),)


def _translation_operators(config) -> tuple[WorldOperator, ...]:
    values = _finite_values(config['NOISE_TRANSLATE_STD'], 3, 'NOISE_TRANSLATE_STD')
    if any(value < 0. for value in values):
        raise AugmentationConfigurationError('NOISE_TRANSLATE_STD must be nonnegative')
    return (TranslationOperator((values[0], values[1], values[2])),)


_OPERATOR_FACTORIES: Final = {
    'random_world_flip': _flip_operators,
    'random_world_rotation': _rotation_operators,
    'random_world_scaling': _scaling_operators,
    'random_world_translation': _translation_operators,
}


def should_skip_imgaug(
    *, camera_imgs_present: bool, camera_is_tensor: bool, img_process_infos_present: bool,
) -> bool:
    return ((camera_imgs_present and camera_is_tensor)
            or not camera_imgs_present or not img_process_infos_present)


def world_operators_from_queue(
    queue: Sequence[partial], *, skip_imgaug: bool,
) -> tuple[WorldOperator, ...]:
    operators: list[WorldOperator] = []
    for augmentor in queue:
        if not isinstance(augmentor, partial):
            raise AugmentationConfigurationError(
                f'Formal K4 does not support active augmentor: {type(augmentor).__name__}'
            )
        name = augmentor.func.__name__
        if name == 'imgaug' and skip_imgaug:
            continue
        factory = _OPERATOR_FACTORIES.get(name)
        if factory is None:
            raise AugmentationConfigurationError(f'Formal K4 does not support active augmentor: {name}')
        config = augmentor.keywords.get('config')
        if config is None:
            raise AugmentationConfigurationError(f'Formal K4 augmentor lacks config: {name}')
        operators.extend(factory(config))
    return tuple(operators)


@dataclass(frozen=True, slots=True)
class LegacyAugmentationSampler:
    point_count: int
    drop_rate: float
    operators: tuple[WorldOperator, ...]

    def __post_init__(self) -> None:
        if type(self.point_count) is not int or self.point_count < 0:
            raise AugmentationConfigurationError('point_count must be a nonnegative integer')
        if not np.isfinite(self.drop_rate):
            raise AugmentationConfigurationError('drop_rate must be finite')

    def __call__(
        self, drop_rng: np.random.Generator, geometry_rng: np.random.Generator,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        rate = float(np.clip(self.drop_rate, 0., .95))
        retained = np.flatnonzero(drop_rng.random(self.point_count) >= rate).astype(np.int64)
        if self.point_count and retained.size == 0:
            retained = np.array([0], dtype=np.int64)
        transform = np.eye(4, dtype=np.float64)
        for operator in self.operators:
            transform = operator.sample(geometry_rng) @ transform
        return retained, transform.astype(np.float32)
