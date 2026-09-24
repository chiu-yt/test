from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILTIN_GENERICS = {'dict', 'frozenset', 'list', 'set', 'tuple', 'type'}


def compatibility_paths():
    paths = []
    for pattern in (
        'pcdet/tta_methods/spcra_k4_*.py',
        'pcdet/utils/figure7_*.py',
        'tools/figure7_utils/*.py',
        'tools/generate_figure7.py',
        'tests/figure7_fixtures.py',
        'tests/test_spcra_k4_*.py',
        'tests/test_figure7_*.py',
    ):
        paths.extend(ROOT.glob(pattern))
    return tuple(sorted(set(paths)))


def has_future_annotations(tree):
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == '__future__'
        and any(alias.name == 'annotations' for alias in node.names)
        for node in tree.body
    )


def annotation_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):  # noqa: IF_VARIANT_OK - Python 3.8.
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if argument.annotation is not None:
                    yield argument.annotation
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                yield node.args.vararg.annotation
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                yield node.args.kwarg.annotation
            if node.returns is not None:
                yield node.returns


def uses_runtime_modern_annotation(tree):
    for annotation in annotation_nodes(tree):
        for node in ast.walk(annotation):
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in BUILTIN_GENERICS):
                return True
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                return True
    return False


def unsupported_python38_api(node):
    if not isinstance(node, ast.Attribute):
        return None
    supported_readlink = isinstance(node.value, ast.Name) and node.value.id == 'os'
    unsupported = node.attr in {'is_relative_to', 'unparse'} or (
        node.attr == 'readlink' and not supported_readlink
    )
    return (node.attr, node.lineno) if unsupported else None


class TestPython38Compatibility(unittest.TestCase):
    def test_readlink_detection_distinguishes_pathlib_from_os(self):
        path_call = ast.parse('path.readlink()', mode='eval').body
        os_call = ast.parse('os.readlink(path)', mode='eval').body
        if not isinstance(path_call, ast.Call) or not isinstance(os_call, ast.Call):
            raise AssertionError('Expected call expressions')
        self.assertIsNotNone(unsupported_python38_api(path_call.func))
        self.assertIsNone(unsupported_python38_api(os_call.func))

    def test_formal_sources_use_python38_runtime_constructs(self):
        failures = []
        for path in compatibility_paths():
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source, filename=str(path), feature_version=8)
            relative = path.relative_to(ROOT)

            if uses_runtime_modern_annotation(tree) and not has_future_annotations(tree):
                failures.append(f'{relative}: modern annotations are evaluated at runtime')

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dataclass':
                    if any(keyword.arg == 'slots' for keyword in node.keywords):
                        failures.append(f'{relative}:{node.lineno}: dataclass slots require Python 3.10')
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    if (isinstance(value, ast.Subscript)
                            and isinstance(value.value, ast.Name)
                            and value.value.id in BUILTIN_GENERICS):
                        failures.append(f'{relative}:{node.lineno}: evaluated built-in generic alias')
                unsupported_api = unsupported_python38_api(node)
                if unsupported_api is not None:
                    name, line = unsupported_api
                    failures.append(f'{relative}:{line}: unsupported Python 3.8 API `{name}`')

        self.assertEqual(failures, [], '\n' + '\n'.join(failures))


if __name__ == '__main__':
    unittest.main()
