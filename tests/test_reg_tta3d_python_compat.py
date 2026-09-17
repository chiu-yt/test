import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REG_TTA3D_PATHS = (
    REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d.py',
    REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d_utils.py',
    REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d_npg.py',
)


def test_reg_tta3d_dataclasses_support_python_before_310():
    # Given Reg-TTA3D modules imported by the online evaluator.
    modules = [
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for path in REG_TTA3D_PATHS
    ]

    # When their annotation mode and dataclass decorators are inspected.
    core_module = modules[0]
    future_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == '__future__'
        and any(alias.name == 'annotations' for alias in node.names)
        for node in core_module.body
    )
    decorators = [
        decorator
        for module in modules
        for node in module.body
        if isinstance(node, ast.ClassDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == 'dataclass'
    ]

    # Then Python-3.10-only slots are absent while frozen behavior is retained.
    assert future_annotations
    assert decorators
    for decorator in decorators:
        keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
        assert 'slots' not in keywords
        frozen = keywords.get('frozen')
        assert isinstance(frozen, ast.Constant)
        assert frozen.value is True
