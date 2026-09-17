import ast
from pathlib import Path
import tempfile
import unittest

from pcdet.utils.efficiency_profiler import (
    EfficiencyProfileConfigurationError,
    EfficiencyProfileRun,
    EfficiencyProfiler,
)
from pcdet.utils.inference_utils import build_inference_batch, forward_without_annotations
from tests.test_efficiency_profiler import FakeCuda, FakeParameter


REPO_ROOT = Path(__file__).resolve().parents[1]


def _function(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=8)
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(node, name):
    return sorted(
        (
            child for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and (
                isinstance(child.func, ast.Name) and child.func.id == name
                or isinstance(child.func, ast.Attribute) and child.func.attr == name
            )
        ),
        key=lambda child: child.lineno,
    )


class EfficiencyProfileFairnessTest(unittest.TestCase):
    def test_inference_batch_excludes_annotations_without_mutating_input(self):
        original = {
            'batch_size': 2,
            'points': 'points',
            'gt_boxes': 'boxes',
            'gt_names': 'names',
            'sample_annotation_tokens': 'tokens',
        }

        inference = build_inference_batch(original)

        self.assertEqual(inference, {'batch_size': 2, 'points': 'points'})
        self.assertIn('gt_boxes', original)

    def test_supported_prediction_paths_strip_annotations(self):
        targets = (
            ('tools/eval_utils/eval_utils.py', 'eval_one_epoch', 1),
            ('tools/eval_utils/tent_eval_utils.py', 'eval_tent_one_epoch', 1),
            ('pcdet/tta_methods/mos.py', 'optimize', 2),
            ('pcdet/tta_methods/mos.py', '_run_spcra_diagnostic', 1),
            ('pcdet/tta_methods/mos.py', '_apply_dpo_matcher', 1),
            ('pcdet/tta_methods/mos.py', '_perform_aggregation', 1),
            ('pcdet/tta_methods/codemerge.py', '_perform_aggregation', 1),
        )

        for relative_path, function_name, expected_count in targets:
            with self.subTest(path=relative_path):
                function = _function(REPO_ROOT / relative_path, function_name)
                builders = [
                    node for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'forward_without_annotations'
                ]
                self.assertEqual(len(builders), expected_count)

    def test_forward_sidecars_are_merged_without_restoring_annotations(self):
        original = {'batch_size': 1, 'gt_boxes': 'ground-truth'}

        def model(batch):
            self.assertNotIn('gt_boxes', batch)
            batch['spatial_features_img'] = 'camera-features'
            return 'predictions'

        output = forward_without_annotations(model, original)

        self.assertEqual(output, 'predictions')
        self.assertEqual(original['gt_boxes'], 'ground-truth')
        self.assertEqual(original['spatial_features_img'], 'camera-features')

    def test_train_checkpoint_bank_save_is_inside_profiled_batch(self):
        function = _function(
            REPO_ROOT / 'tools/train_utils/train_st_utils.py', 'train_model_st'
        )
        begins = _calls(function, 'begin_segment')
        ends = _calls(function, 'end_segment')
        end_batches = _calls(function, 'end_batch')
        saves = _calls(function, 'save_checkpoint')

        self.assertEqual(len(begins), 2)
        self.assertEqual(len(ends), 2)
        self.assertLess(begins[1].lineno, saves[0].lineno)
        self.assertLess(saves[0].lineno, ends[1].lineno)
        self.assertLess(ends[1].lineno, end_batches[0].lineno)

    def test_incomplete_window_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = EfficiencyProfiler(
                {'ENABLED': True, 'WARMUP_ITERS': 0, 'MEASURE_ITERS': 1},
                EfficiencyProfileRun(
                    method='source_only',
                    entrypoint='test.py',
                    config='bevfusion',
                    boundary='prediction',
                    output_dir=Path(temp_dir),
                    model_parameters=[FakeParameter(1)],
                    updated_parameters=(),
                ),
                cuda_backend=FakeCuda([]),
            )

            with self.assertRaises(EfficiencyProfileConfigurationError):
                profiler.finalize()


if __name__ == '__main__':
    unittest.main()
