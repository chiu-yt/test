import functools
import pickle
import unittest

import numpy as np

from test_spcra_k4_runtime import load_contract_module


LAW_ID = 'spcra_legacy_bernoulli_world_v1'


def queued(name, config):
    def augment(data_dict=None, config=None):
        return data_dict

    augment.__name__ = name
    return functools.partial(augment, config=config)


class TestK4Augmentation(unittest.TestCase):
    def setUp(self):
        self.augmentation = load_contract_module('spcra_k4_augmentation')

    def test_bernoulli_retention_when_sampling_repeated_views(self):
        counts = []
        for seed in range(12):
            sampler = self.augmentation.LegacyAugmentationSampler(40, .25, ())
            indices, _ = sampler(np.random.default_rng(seed), np.random.default_rng(seed + 100))
            counts.append(len(indices))
        self.assertGreater(len(set(counts)), 1)

    def test_first_point_fallback_when_bernoulli_drops_every_point(self):
        sampler = self.augmentation.LegacyAugmentationSampler(1, .95, ())
        indices, transform = sampler(np.random.default_rng(1), np.random.default_rng(2))
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_array_equal(transform, np.eye(4, dtype=np.float32))

    def test_retained_indices_preserve_source_row_and_feature_order(self):
        seed = 23
        expected_rng = np.random.default_rng(seed)
        expected = np.flatnonzero(expected_rng.random(30) >= .35)
        sampler = self.augmentation.LegacyAugmentationSampler(30, .35, ())
        indices, _ = sampler(np.random.default_rng(seed), np.random.default_rng(99))
        points = np.arange(150, dtype=np.float32).reshape(30, 5)
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_array_equal(points[indices, 3:], points[expected, 3:])

    def test_world_operators_follow_effective_queue_and_configured_laws(self):
        queue = (
            queued('random_world_translation', {'NOISE_TRANSLATE_STD': [1., 2., 3.]}),
            queued('random_world_flip', {'ALONG_AXIS_LIST': ['x', 'y']}),
            queued('random_world_rotation', {'WORLD_ROT_ANGLE': [-.4, .2]}),
            queued('random_world_scaling', {'WORLD_SCALE_RANGE': [.8, 1.1]}),
            queued('imgaug', {'RAND_FLIP': True, 'ROT_LIM': [-5., 5.]}),
        )
        operators = self.augmentation.world_operators_from_queue(queue, skip_imgaug=True)
        sampler = self.augmentation.LegacyAugmentationSampler(3, 0., operators)
        expected_rng = np.random.default_rng(91)
        translation = expected_rng.normal(0., [1., 2., 3.])
        flip_x = bool(expected_rng.choice([False, True], p=[.5, .5]))
        flip_y = bool(expected_rng.choice([False, True], p=[.5, .5]))
        yaw = expected_rng.uniform(-.4, .2)
        scale = expected_rng.uniform(.8, 1.1)
        expected = np.eye(4)
        expected[:3, 3] = translation
        for enabled, diagonal in ((flip_x, [1., -1., 1.]), (flip_y, [-1., 1., 1.])):
            if enabled:
                operator = np.eye(4)
                operator[:3, :3] = np.diag(diagonal)
                expected = operator @ expected
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.eye(4)
        rotation[:3, :3] = [[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]]
        expected = rotation @ expected
        scaling = np.eye(4)
        scaling[:3, :3] *= scale
        expected = scaling @ expected
        indices, actual = sampler(np.random.default_rng(7), np.random.default_rng(91))
        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    def test_tensor_camera_condition_skips_imgaug_but_not_unknown_operators(self):
        imgaug = queued('imgaug', {'RAND_FLIP': True})
        self.assertTrue(self.augmentation.should_skip_imgaug(
            camera_imgs_present=True, camera_is_tensor=True, img_process_infos_present=True,
        ))
        self.assertTrue(self.augmentation.should_skip_imgaug(
            camera_imgs_present=False, camera_is_tensor=False, img_process_infos_present=False,
        ))
        self.assertFalse(self.augmentation.should_skip_imgaug(
            camera_imgs_present=True, camera_is_tensor=False, img_process_infos_present=True,
        ))
        self.assertEqual(self.augmentation.world_operators_from_queue((imgaug,), skip_imgaug=True), ())
        for item in (imgaug, queued('gt_sampling', {}), queued('random_local_scaling', {})):
            with self.subTest(name=item.func.__name__):
                with self.assertRaises(self.augmentation.AugmentationConfigurationError):
                    self.augmentation.world_operators_from_queue((item,), skip_imgaug=False)
        with self.assertRaises(self.augmentation.AugmentationConfigurationError):
            self.augmentation.world_operators_from_queue((lambda: None,), skip_imgaug=True)

    def test_local_generators_leave_global_numpy_rng_unchanged(self):
        queue = (queued('random_world_translation', {'NOISE_TRANSLATE_STD': [.5, .5, .5]}),)
        operators = self.augmentation.world_operators_from_queue(queue, skip_imgaug=True)
        sampler = self.augmentation.LegacyAugmentationSampler(20, .4, operators)
        before = pickle.dumps(np.random.get_state())
        sampler(np.random.default_rng(5), np.random.default_rng(6))
        self.assertEqual(pickle.dumps(np.random.get_state()), before)
        self.assertEqual(self.augmentation.AUGMENTATION_LAW_ID, LAW_ID)


if __name__ == '__main__':
    unittest.main()
