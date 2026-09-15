import ast
from pathlib import Path


def test_sar_dataclasses_support_python_before_310():
    # Given: the SAR module is parsed without importing CUDA dependencies.
    source_path = Path(__file__).parents[1] / 'pcdet' / 'tta_methods' / 'sar.py'
    module = ast.parse(source_path.read_text())

    # When: its future imports and dataclass decorators are inspected.
    future_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == '__future__'
        and any(alias.name == 'annotations' for alias in node.names)
        for node in module.body
    )
    dataclass_decorators = [
        decorator
        for node in module.body
        if isinstance(node, ast.ClassDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == 'dataclass'
    ]

    # Then: annotations are deferred and no Python-3.10-only option is used.
    assert future_annotations
    assert dataclass_decorators
    assert all(
        keyword.arg != 'slots'
        for decorator in dataclass_decorators
        for keyword in decorator.keywords
    )
