import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ST_PATH = REPO_ROOT / 'tools' / 'train_utils' / 'train_st_utils.py'
MOS_PATH = REPO_ROOT / 'pcdet' / 'tta_methods' / 'mos.py'


def _tree(path):
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _function(tree, name):
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, '%s must exist exactly once' % name
    return matches[0]


def _calls(node, name):
    return sorted(
        [
            child for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and (
                isinstance(child.func, ast.Name) and child.func.id == name
                or isinstance(child.func, ast.Attribute) and child.func.attr == name
            )
        ],
        key=lambda child: child.lineno,
    )


def _keyword(call, name):
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    assert len(matches) == 1, '%s must be passed exactly once' % name
    return matches[0]


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return '%s.%s' % (parent, node.attr)
    return None


def _containing_if(function, target):
    return [
        node for node in ast.walk(function)
        if isinstance(node, ast.If) and target in ast.walk(node)
    ]


def test_mos_gpu_load_seam_preserves_disabled_default():
    # Given the train-side MOS optimizer entrypoint.
    optimize = _function(_tree(MOS_PATH), 'optimize')

    # When its signature and GPU transfer are inspected.
    defaults = dict(zip(
        [argument.arg for argument in optimize.args.args[-len(optimize.args.defaults):]],
        optimize.args.defaults,
    ))
    gpu_loads = _calls(optimize, 'load_data_to_gpu')

    # Then legacy callers still load once, while profiled callers can opt out.
    data_already_on_gpu_default = defaults.get('data_already_on_gpu')
    assert isinstance(data_already_on_gpu_default, ast.Constant)
    assert data_already_on_gpu_default.value is False
    assert len(gpu_loads) == 1
    guards = _containing_if(optimize, gpu_loads[0])
    assert len(guards) == 1
    condition = guards[0].test
    assert isinstance(condition, ast.UnaryOp)
    assert isinstance(condition.op, ast.Not)
    assert _dotted_name(condition.operand) == 'data_already_on_gpu'


def test_train_tta_profile_uses_run_directory_and_method_labels():
    # Given the train-based TTA entrypoint.
    train_model_st = _function(_tree(TRAIN_ST_PATH), 'train_model_st')

    # When the shared profiler run contract is constructed.
    runs = _calls(train_model_st, 'EfficiencyProfileRun')
    profilers = _calls(train_model_st, 'EfficiencyProfiler')

    # Then one profiler uses real train objects, the directory beside ckpt, and both labels.
    assert len(runs) == 1
    assert len(profilers) == 1
    method = _keyword(runs[0], 'method')
    assert isinstance(method, ast.IfExp)
    assert ast.literal_eval(method.body) == 'codemerge'
    assert ast.literal_eval(method.orelse) == 'refuse_tta'
    assert _dotted_name(_keyword(runs[0], 'output_dir')) == 'ckpt_save_dir.parent'
    model_parameters = _keyword(runs[0], 'model_parameters')
    assert isinstance(model_parameters, ast.Call)
    assert _dotted_name(model_parameters.func) == 'model.parameters'
    assert _dotted_name(_keyword(runs[0], 'optimizer')) == 'optimizer'


def test_train_tta_profile_rejects_distributed_execution():
    # Given profiling setup in the train-based TTA entrypoint.
    train_model_st = _function(_tree(TRAIN_ST_PATH), 'train_model_st')

    # When profile-enabled guards are inspected.
    raises = [node for node in ast.walk(train_model_st) if isinstance(node, ast.Raise)]
    guards = [node for node in ast.walk(train_model_st) if isinstance(node, ast.If)]

    # Then distributed initialization/world size is rejected before measurement.
    guarded_nodes = [node for node in guards if raises and any(
        raised in ast.walk(node) for raised in raises
    )]
    guarded_names = {
        _dotted_name(node) for guard in guarded_nodes for node in ast.walk(guard)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    assert 'profile_enabled' in guarded_names
    assert 'torch.distributed' in guarded_names
    assert 'torch.distributed.get_world_size' in guarded_names


def test_profile_boundary_excludes_gpu_load_and_checkpoint_saves():
    # Given the complete train-based TTA loop.
    train_model_st = _function(_tree(TRAIN_ST_PATH), 'train_model_st')

    # When lifecycle, adaptation, and checkpoint calls are ordered.
    gpu_load = _calls(train_model_st, 'load_data_to_gpu')
    begin_batch = _calls(train_model_st, 'begin_batch')
    begin_segment = _calls(train_model_st, 'begin_segment')
    zero_grad = _calls(train_model_st, 'zero_grad')
    optimize = _calls(train_model_st, 'optimize')
    clip_grad = _calls(train_model_st, 'clip_grad_norm_')
    optimizer_steps = [
        call for call in _calls(train_model_st, 'step')
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == 'optimizer'
    ]
    scheduler_steps = [
        call for call in _calls(train_model_st, 'step')
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == 'cur_scheduler'
    ]
    end_segment = _calls(train_model_st, 'end_segment')
    end_batch = _calls(train_model_st, 'end_batch')
    checkpoints = _calls(train_model_st, 'save_checkpoint')
    progress_updates = _calls(train_model_st, 'update')
    tensorboard_writes = _calls(train_model_st, 'add_scalar')

    # Then only adaptation work is inside the single measured segment.
    assert len(gpu_load) == 1
    assert len(begin_batch) == len(end_batch) == 1
    assert len(begin_segment) == len(end_segment) == 2
    assert len(zero_grad) == len(clip_grad) == 1
    assert len(optimize) == 2
    assert len(optimizer_steps) == len(scheduler_steps) == 1
    assert scheduler_steps[0].lineno < gpu_load[0].lineno
    assert gpu_load[0].lineno < begin_batch[0].lineno < begin_segment[0].lineno
    assert begin_segment[0].lineno < zero_grad[0].lineno < min(call.lineno for call in optimize)
    assert max(call.lineno for call in optimize) < clip_grad[0].lineno < optimizer_steps[0].lineno
    assert optimizer_steps[0].lineno < end_segment[0].lineno
    assert end_segment[0].lineno < begin_segment[1].lineno
    assert begin_segment[1].lineno < checkpoints[0].lineno < end_segment[1].lineno
    assert end_segment[1].lineno < end_batch[0].lineno
    assert end_batch[0].lineno < checkpoints[-1].lineno
    assert all(end_batch[0].lineno < update.lineno for update in progress_updates)
    assert all(end_batch[0].lineno < write.lineno for write in tensorboard_writes)
    profiled_optimize = [call for call in optimize if call.keywords]
    assert len(profiled_optimize) == 1
    data_already_on_gpu = _keyword(profiled_optimize[0], 'data_already_on_gpu')
    assert isinstance(data_already_on_gpu, ast.Constant)
    assert data_already_on_gpu.value is True
    optimize_guard = _containing_if(train_model_st, profiled_optimize[0])
    assert len(optimize_guard) == 1
    assert _dotted_name(optimize_guard[0].test) == 'profile_enabled'
    profile_guard = _containing_if(train_model_st, gpu_load[0])
    assert len(profile_guard) == 1
    assert _dotted_name(profile_guard[0].test) == 'profile_enabled'
    adaptation_loops = [
        node for node in ast.walk(train_model_st)
        if isinstance(node, ast.For) and begin_batch[0] in ast.walk(node)
    ]
    assert adaptation_loops
    assert all(
        not any(isinstance(child, ast.Break) for child in ast.walk(loop))
        for loop in adaptation_loops
    )


def test_profile_finalizes_once_after_epoch_processing():
    # Given the complete train-based TTA entrypoint.
    train_model_st = _function(_tree(TRAIN_ST_PATH), 'train_model_st')

    # When finalization and epoch checkpoint calls are inspected.
    finalizers = _calls(train_model_st, 'finalize')
    checkpoints = _calls(train_model_st, 'save_checkpoint')
    returns = [node for node in ast.walk(train_model_st) if isinstance(node, ast.Return)]

    # Then the profiler is finalized once after all saves and before successful return.
    assert len(finalizers) == 1
    assert checkpoints
    assert max(call.lineno for call in checkpoints) < finalizers[0].lineno
    completion_returns = [node for node in returns if node.value is not None]
    assert completion_returns
    assert finalizers[0].lineno < max(node.lineno for node in completion_returns)


class TrainTTAEfficiencyProfileTest(unittest.TestCase):
    def test_mos_gpu_load_seam_preserves_disabled_default(self):
        test_mos_gpu_load_seam_preserves_disabled_default()

    def test_train_tta_profile_uses_run_directory_and_method_labels(self):
        test_train_tta_profile_uses_run_directory_and_method_labels()

    def test_train_tta_profile_rejects_distributed_execution(self):
        test_train_tta_profile_rejects_distributed_execution()

    def test_profile_boundary_excludes_gpu_load_and_checkpoint_saves(self):
        test_profile_boundary_excludes_gpu_load_and_checkpoint_saves()

    def test_profile_finalizes_once_after_epoch_processing(self):
        test_profile_finalizes_once_after_epoch_processing()


if __name__ == '__main__':
    unittest.main()
