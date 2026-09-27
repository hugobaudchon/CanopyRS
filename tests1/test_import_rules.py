"""
Checks the import rules of canopyrs1, by reading the source of every module without importing it:

- canopyrs1 never imports canopyrs or geodataset;
- imports are absolute (``from canopyrs1.core.tables import Tiles``), never relative;
- core/ never imports torch or a model framework, nor anything in canopyrs1 outside core/;
- every other part only imports itself or the parts listed before it in LAYERS.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).parent.parent / "canopyrs1"

# The parts of canopyrs1, lowest first. A part may import itself and the parts before it.
LAYERS = [
    "core",
    "config_definitions",
    "models",
    "pipeline",
    "public_datasets",
    "benchmark",
    "training",
    "installers",
    "tools",
    "doctor",
    "cli",
]

FORBIDDEN_PACKAGES = {"canopyrs", "geodataset"}

FORBIDDEN_IN_CORE = {
    "torch", "torchvision", "detectron2", "detrex", "mmdet", "mmcv", "mmengine",
    "sam2", "sam3", "transformers", "timm", "deepforest",
}


def _imported_modules(tree):
    """Yield (line number, module name) for every import in ``tree``, including imports inside
    functions. ``from canopyrs1 import models`` yields ``canopyrs1.models``. A relative import
    yields its dots followed by its module, e.g. ``..tables``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                yield node.lineno, "." * node.level + (node.module or "")
            elif node.module == "canopyrs1":
                for alias in node.names:
                    yield node.lineno, f"canopyrs1.{alias.name}"
            else:
                yield node.lineno, node.module


def import_violations(relative_path, source):
    """Return one message per broken import rule in the module at ``relative_path`` (a path inside
    canopyrs1/, e.g. "core/tables/imagery.py") whose code is ``source``. Returns an empty list when
    the module follows every rule."""
    parts = Path(relative_path).with_suffix("").parts
    layer = None if parts == ("__init__",) else parts[0]      # None: canopyrs1/__init__.py itself
    if layer is not None and layer not in LAYERS:
        return [f"{relative_path}: '{layer}' is not in LAYERS; add it there"]

    violations = []
    for line, module in _imported_modules(ast.parse(source)):
        where = f"{relative_path}:{line} imports {module}"
        top = module.split(".")[0]
        if module.startswith("."):
            violations.append(f"{where}: use an absolute import")
        elif top in FORBIDDEN_PACKAGES:
            violations.append(f"{where}: canopyrs1 must not use the old code")
        elif layer == "core" and top in FORBIDDEN_IN_CORE:
            violations.append(f"{where}: core/ must not use torch or a model framework")
        elif top == "canopyrs1" and layer is not None and len(module.split(".")) > 1:
            target = module.split(".")[1]
            if target not in LAYERS:
                violations.append(f"{where}: '{target}' is not in LAYERS")
            elif layer == "core" and target != "core":
                violations.append(f"{where}: core/ must only import core/")
            elif LAYERS.index(target) > LAYERS.index(layer):
                violations.append(f"{where}: '{layer}' comes before '{target}' in LAYERS")
    return violations


def test_canopyrs1_follows_the_import_rules():
    violations = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        violations += import_violations(path.relative_to(PACKAGE_DIR).as_posix(), path.read_text())
    assert not violations, "\n".join(violations)


# The checker itself, on small made-up modules.

@pytest.mark.parametrize("path, source", [
    ("core/tables/imagery.py", "import numpy\nfrom canopyrs1.core.geometry import georef"),
    ("models/base.py", "import torch\nfrom canopyrs1.core.tables import Tiles"),
    ("models/base.py", "from canopyrs1 import config_definitions"),
    ("training/sam.py", "from canopyrs1.pipeline import Pipeline\nfrom canopyrs1.benchmark import evaluator"),
    ("cli.py", "from canopyrs1.installers import setup"),
    ("__init__.py", "from importlib.metadata import version"),
])
def test_allowed_imports(path, source):
    assert import_violations(path, source) == []


@pytest.mark.parametrize("path, source, reason", [
    ("pipeline/run.py", "from canopyrs.engine.data import Tiles", "old code"),
    ("core/io/hf.py", "import geodataset", "old code"),
    ("core/tables/imagery.py", "from .table import Table", "absolute import"),
    ("core/raster/read.py", "import torch", "torch"),
    ("core/tiling/grid.py", "def f():\n    from detectron2 import model_zoo", "torch"),
    ("core/tiling/grid.py", "from canopyrs1.config_definitions import TilerizerConfig", "only import core"),
    ("models/base.py", "from canopyrs1.training import trainer", "comes before"),
    ("models/base.py", "from canopyrs1 import pipeline", "comes before"),
    ("models/base.py", "from canopyrs1.engine import data", "not in LAYERS"),
    ("engine/data.py", "import numpy", "not in LAYERS"),
])
def test_forbidden_imports(path, source, reason):
    violations = import_violations(path, source)
    assert len(violations) == 1 and reason in violations[0], violations
