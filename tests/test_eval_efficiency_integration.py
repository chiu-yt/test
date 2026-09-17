import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).parents[1]
EVAL_FILES = {
    'source_only': REPO_ROOT / 'tools' / 'eval_utils' / 'eval_utils.py',
    'tent': REPO_ROOT / 'tools' / 'eval_utils' / 'tent_eval_utils.py',
    'sar': REPO_ROOT / 'tools' / 'eval_utils' / 'sar_eval_utils.py',
}
FUNCTION_NAMES = {
    'source_only': 'eval_one_epoch',
    'tent': 'eval_tent_one_epoch',
    'sar': 'eval_sar_one_epoch',
}


def _function(method):
    tree = ast.parse(EVAL_FILES[method].read_text(encoding='utf-8'))
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == FUNCTION_NAMES[method]
    )


def _call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls(node, name):
    return sorted(
        (
            child for child in ast.walk(node)
            if isinstance(child, ast.Call) and _call_name(child) == name
        ),
        key=lambda child: child.lineno,
    )


def _tokens(node):
    return {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    } | {
        child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
    } | {
        child.value for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _keyword(call, name):
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _loop_with_call(function, name):
    return next(
        node for node in ast.walk(function)
        if isinstance(node, ast.For) and _calls(node, name)
    )


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return '%s.%s' % (parent, node.attr)
    return None


def _called_name(node):
    if not isinstance(node, ast.Call):
        return None
    return _dotted_name(node.func)


def _assert_common_run_contract(test_case, method):
    function = _function(method)
    profiler_calls = _calls(function, 'EfficiencyProfiler')
    run_calls = _calls(function, 'EfficiencyProfileRun')
    test_case.assertEqual(len(profiler_calls), 1)
    test_case.assertEqual(len(run_calls), 1)

    run = run_calls[0]
    test_case.assertEqual(ast.literal_eval(_keyword(run, 'method')), method)
    test_case.assertEqual(ast.literal_eval(_keyword(run, 'entrypoint')), 'test.py')
    config = _keyword(run, 'config')
    assert isinstance(config, ast.Call)
    test_case.assertEqual(_called_name(config), 'str')
    test_case.assertEqual(_dotted_name(config.args[0]), 'cfg.TAG')
    test_case.assertEqual(_dotted_name(_keyword(run, 'output_dir')), 'result_dir')
    test_case.assertEqual(_called_name(_keyword(run, 'model_parameters')), 'model.parameters')
    test_case.assertEqual(len(_calls(function, 'finalize')), 1)
    test_case.assertEqual(len(_calls(function, 'end_batch')), 1)
    test_case.assertEqual(len(_calls(function, 'begin_batch')), 1)
    return function, run


class EvalEfficiencyIntegrationTest(unittest.TestCase):
    def test_all_evaluators_build_one_profile_in_the_result_directory(self):
        # Given the three supported test.py evaluation paths.
        methods = ('source_only', 'tent', 'sar')

        # When their profiler construction contracts are inspected.
        runs = {
            method: _assert_common_run_contract(self, method)[1]
            for method in methods
        }

        # Then Source-only is explicit zero-update and TTA uses actual optimizers.
        source_updates = _keyword(runs['source_only'], 'updated_parameters')
        assert isinstance(source_updates, ast.Tuple)
        self.assertEqual(source_updates.elts, [])
        self.assertEqual(_dotted_name(_keyword(runs['tent'], 'optimizer')), 'optimizer')
        self.assertEqual(_dotted_name(_keyword(runs['sar'], 'optimizer')), 'optimizer')

    def test_source_only_times_only_prediction_and_continues_serialization(self):
        # Given the standard source-only loop after all method dispatches.
        function, _ = _assert_common_run_contract(self, 'source_only')
        loop = _loop_with_call(function, 'begin_batch')

        # When ordered calls in the loop are inspected.
        profiler_line = _calls(function, 'EfficiencyProfiler')[0].lineno
        dispatch_lines = [
            call.lineno for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and (_call_name(call) or '').startswith('eval_')
            and _call_name(call) != 'eval_one_epoch'
        ]
        load_line = _calls(loop, 'load_data_to_gpu')[0].lineno
        begin_batch_line = _calls(loop, 'begin_batch')[0].lineno
        begin_line = _calls(loop, 'begin_segment')[0].lineno
        model_line = _calls(loop, 'forward_without_annotations')[0].lineno
        end_line = _calls(loop, 'end_segment')[0].lineno
        serialize_line = _calls(loop, 'generate_prediction_dicts')[0].lineno
        end_batch_line = _calls(loop, 'end_batch')[0].lineno

        # Then loading and serialization are outside the sole prediction segment.
        self.assertTrue(dispatch_lines)
        self.assertLess(max(dispatch_lines), profiler_line)
        self.assertLess(load_line, begin_batch_line)
        self.assertLess(begin_batch_line, begin_line)
        self.assertLess(begin_line, model_line)
        self.assertLess(model_line, end_line)
        self.assertLess(end_line, end_batch_line)
        self.assertLess(end_batch_line, serialize_line)
        self.assertFalse(any(isinstance(node, (ast.Break, ast.Return)) for node in loop.body))
        self.assertLess(_calls(function, 'finalize')[0].lineno, _calls(function, 'evaluation')[0].lineno)

    def test_tent_splits_prediction_from_serialization_and_times_update(self):
        # Given Tent's predict-serialize-update order.
        function, _ = _assert_common_run_contract(self, 'tent')
        loop = _loop_with_call(function, 'begin_batch')

        # When its two timing segments are inspected.
        begins = _calls(loop, 'begin_segment')
        ends = _calls(loop, 'end_segment')
        load_line = _calls(loop, 'load_data_to_gpu')[0].lineno
        begin_batch_line = _calls(loop, 'begin_batch')[0].lineno
        model_line = _calls(loop, 'forward_without_annotations')[0].lineno
        serialize_line = _calls(loop, 'generate_prediction_dicts')[0].lineno
        backward_line = _calls(loop, 'backward')[0].lineno
        optimizer_step_line = _calls(loop, 'step')[0].lineno

        # Then CPU prediction serialization is between GPU prediction and update.
        self.assertEqual(len(begins), 2)
        self.assertEqual(len(ends), 2)
        self.assertLess(load_line, begin_batch_line)
        self.assertLess(begin_batch_line, begins[0].lineno)
        self.assertLess(begins[0].lineno, model_line)
        self.assertLess(model_line, ends[0].lineno)
        self.assertLess(ends[0].lineno, serialize_line)
        self.assertLess(serialize_line, begins[1].lineno)
        self.assertLess(begins[1].lineno, backward_line)
        self.assertLess(backward_line, optimizer_step_line)
        self.assertLess(optimizer_step_line, ends[1].lineno)
        self.assertLess(ends[1].lineno, _calls(loop, 'end_batch')[0].lineno)
        self.assertFalse(any(isinstance(node, ast.Break) for node in loop.body))
        self.assertLess(_calls(function, 'finalize')[0].lineno, _calls(function, 'evaluation')[0].lineno)

    def test_sar_splits_prediction_from_serialization_and_times_adaptation(self):
        # Given SAR's predict-serialize-adapt order.
        function, _ = _assert_common_run_contract(self, 'sar')
        loop = _loop_with_call(function, 'begin_batch')

        # When its two timing segments are inspected.
        begins = _calls(loop, 'begin_segment')
        ends = _calls(loop, 'end_segment')
        load_line = _calls(loop, 'load_data_to_gpu')[0].lineno
        begin_batch_line = _calls(loop, 'begin_batch')[0].lineno
        model_line = _calls(loop, 'model')[0].lineno
        serialize_line = _calls(loop, 'generate_prediction_dicts')[0].lineno
        adapt_line = _calls(loop, 'adapt')[0].lineno

        # Then serialization is excluded while prediction and adapter SAM work are timed.
        self.assertEqual(len(begins), 2)
        self.assertEqual(len(ends), 2)
        self.assertLess(load_line, begin_batch_line)
        self.assertLess(begin_batch_line, begins[0].lineno)
        self.assertLess(begins[0].lineno, model_line)
        self.assertLess(model_line, ends[0].lineno)
        self.assertLess(ends[0].lineno, serialize_line)
        self.assertLess(serialize_line, begins[1].lineno)
        self.assertLess(begins[1].lineno, adapt_line)
        self.assertLess(adapt_line, ends[1].lineno)
        self.assertLess(ends[1].lineno, _calls(loop, 'end_batch')[0].lineno)
        self.assertFalse(any(isinstance(node, ast.Break) for node in loop.body))
        self.assertLess(
            _calls(function, 'finalize')[0].lineno,
            _calls(function, '_finalize_sar_results')[0].lineno,
        )

    def test_profiler_modes_reject_infer_time_and_distributed_execution(self):
        # Given source-only and Tent conditionally support modes the profiler forbids.
        for method in ('source_only', 'tent'):
            function = _function(method)
            profiler_calls = _calls(function, 'EfficiencyProfiler')
            self.assertEqual(len(profiler_calls), 1)
            profiler_line = profiler_calls[0].lineno

            # When startup guards before profiler construction are inspected.
            guards = [
                node for node in ast.walk(function)
                if isinstance(node, ast.If)
                and node.lineno < profiler_line
                and any(isinstance(child, ast.Raise) for child in ast.walk(node))
            ]
            guard_tokens = [_tokens(guard.test) for guard in guards]

            # Then enabled profiling rejects both incompatible timing modes.
            self.assertTrue(any({'profile_enabled', 'dist_test'} <= tokens for tokens in guard_tokens))
            self.assertTrue(any({'profile_enabled', 'args', 'infer_time'} <= tokens for tokens in guard_tokens))

        # Given SAR already rejects both modes unconditionally.
        sar_function = _function('sar')
        sar_guards = [node for node in ast.walk(sar_function) if isinstance(node, ast.If)]
        sar_tokens = [_tokens(guard.test) for guard in sar_guards]

        # Then its existing stricter restrictions remain in force.
        self.assertTrue(any('dist_test' in tokens for tokens in sar_tokens))
        self.assertTrue(any({'args', 'infer_time'} <= tokens for tokens in sar_tokens))

    def test_tta_profilers_are_built_after_optimizer_configuration(self):
        # Given Tent and SAR optimizer-backed profiling.
        for method, optimizer_builder in (
            ('tent', 'build_tent_optimizer'),
            ('sar', 'build_sar_optimizer'),
        ):
            function = _function(method)

            # When construction order is inspected.
            optimizer_line = _calls(function, optimizer_builder)[0].lineno
            profiler_calls = _calls(function, 'EfficiencyProfiler')
            self.assertEqual(len(profiler_calls), 1)
            profiler_line = profiler_calls[0].lineno

            # Then the profiler sees the optimizer's actual parameter groups.
            self.assertLess(optimizer_line, profiler_line)


if __name__ == '__main__':
    unittest.main()
