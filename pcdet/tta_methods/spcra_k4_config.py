from dataclasses import dataclass

import numpy as np

from .spcra_k4_augmentation import _OPERATOR_FACTORIES

CLASS_NAMES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')


class K4ConfigurationError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


def validate_k4_config(config, *, world_size=1) -> bool:
    tta = config.get('TTA', {})
    spcra = tta.get('SPCRA') or {}
    if not (spcra.get('ENABLED', False) and spcra.get('VERSION') == 'k4_v1'):
        return False
    if world_size != 1:
        raise K4ConfigurationError('Formal K4 requires a single non-DDP GPU')
    required = {'K': 4, 'RELIABILITY_FLOOR': 0., 'RELIABILITY_CLAMP_MAX': 1.,
                'TOPK_PER_CLASS': 0}
    for key, expected in required.items():
        if spcra.get(key) != expected:
            raise K4ConfigurationError(f'Formal K4 requires SPCRA.{key}={expected}')
    if tuple(spcra.get('TARGET_CLASSES', ())) != CLASS_NAMES:
        raise K4ConfigurationError('Formal K4 requires all ten ordered nuScenes classes')
    if tuple(config.get('CLASS_NAMES', ())) != CLASS_NAMES:
        raise K4ConfigurationError('Formal K4 requires nuScenes class order')
    if tta.get('METHOD', 'mos') != 'mos':
        raise K4ConfigurationError('Formal K4 is composed only with MOS')
    if tta.get('MOS_SETTING', {}).get('AGGREGATION_ENABLED', True):
        raise K4ConfigurationError('Formal K4 requires explicit aggregation disablement')
    for name in ('FIGURE6_CAPTURE', 'DPO_MATCHER'):
        if tta.get(name, {}).get('ENABLED', False):
            raise K4ConfigurationError(f'Formal K4 is incompatible with TTA.{name}')
    model = config.get('MODEL', {})
    adapter = model.get('TTA_FUSION_ADAPTER') or {}
    density = adapter.get('SG_DFA') or {}
    freeze = tta.get('FREEZE') or {}
    for name, enabled in (
        ('MODEL.TTA_FUSION_ADAPTER.ENABLED', adapter.get('ENABLED')),
        ('MODEL.TTA_FUSION_ADAPTER.SG_DFA.ENABLED', density.get('ENABLED')),
        ('TTA.FREEZE.ENABLED', freeze.get('ENABLED')),
        ('TTA.FREEZE.ADAPTER_ONLY', freeze.get('ADAPTER_ONLY')),
        ('TTA.SPCRA.CAMERA_RESCUE_ENABLED', spcra.get('CAMERA_RESCUE_ENABLED')),
    ):
        if enabled is not True:
            raise K4ConfigurationError(f'Formal K4 requires {name}=True')
    if tta.get('TTA_STRENGTH', 'mid') != 'mid':
        raise K4ConfigurationError('Formal K4 requires the shared mid TTA augmentation law')
    data = config.get('DATA_CONFIG_TAR', config.get('DATA_CONFIG', {}))
    augmentation = data.get('TTA_DATA_AUGMENTOR') or {}
    disabled = augmentation.get('DISABLE_AUG_LIST', ())
    active = tuple(item for item in augmentation.get('AUG_CONFIG_LIST', ())
                   if item.get('NAME') not in disabled)
    required_law = ('random_world_flip', 'random_world_rotation',
                    'random_world_scaling', 'random_world_translation')
    if tuple(item.get('NAME') for item in active) != required_law:
        raise K4ConfigurationError('Formal K4 requires explicit active TTA flip/rotation/scaling/translation in order')
    try:
        for item in active:
            _OPERATOR_FACTORIES[item['NAME']](item)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise K4ConfigurationError(f'Unsupported formal TTA augmentation law: {error}') from error
    for name in ('FORWARD_ONLY_CONFLICT_ANALYSIS',
                 'FORWARD_ONLY_GEOMETRY_ANALYSIS', 'DEPTH_ENTROPY_ANALYSIS'):
        if model.get(name, {}).get('ENABLED', False):
            raise K4ConfigurationError(f'Formal K4 is incompatible with MODEL.{name}')
    if model.get('NAME', 'BevFusion') != 'BevFusion':
        raise K4ConfigurationError('Formal K4 supports only BevFusion')
    if model.get('DENSE_HEAD', {}).get('NAME', 'TransFusionHead') != 'TransFusionHead':
        raise K4ConfigurationError('Formal K4 supports only TransFusionHead')
    training = config.get('SELF_TRAIN', {})
    for name in ('MEMORY_ENSEMBLE', 'HARD_PSEUDO_MINING', 'PS_FILTER', 'ADAPTIVE_CAP',
                 'DEPTH_FILTER', 'DEPTH_ENTROPY_FILTER', 'GEOMETRY_FILTER',
                 'CONFLICT_FILTER', 'RG_PLM'):
        if training.get(name, {}).get('ENABLED', False):
            raise K4ConfigurationError(f'Formal K4 is incompatible with SELF_TRAIN.{name}')
    for name in ('NEG_THRESH', 'SCORE_THRESH'):
        thresholds = np.asarray(training.get(name, [0.]*10), dtype=np.float64)
        if thresholds.shape != (10,) or not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
            raise K4ConfigurationError(f'{name} must contain ten finite probabilities')
    for data_key in ('DATA_CONFIG', 'DATA_CONFIG_TAR'):
        for aug_key in ('DATA_AUGMENTOR', 'TTA_DATA_AUGMENTOR'):
            augmentation = config.get(data_key, {}).get(aug_key, {})
            disabled = augmentation.get('DISABLE_AUG_LIST', ())
            for augmentor in augmentation.get('AUG_CONFIG_LIST', ()):
                name = augmentor['NAME']
                if name not in disabled and name not in ('random_world_flip', 'random_world_rotation',
                                                         'random_world_scaling', 'random_world_translation', 'imgaug'):
                    raise K4ConfigurationError(f'Formal K4 forbids row-changing/GT augmentation: {name}')
    return True
