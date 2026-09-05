"""What the notebooks assume about the package, checked without running them.

A notebook is the one place in this repo where a broken reference costs nothing until a
human is sitting in front of it. Nothing imports a notebook, so nothing type-checks its
imports, and ruff cannot resolve `from theme.model import mu_at` to a module attribute --
it only sees a name that is used. The suite ran green for a day with `notebooks/analysis.py`
importing a function that had been renamed to `mean_utility_at` during the naming pass; the
notebook failed at its first cell and reported "an ancestor raised an exception" on the two
cells below, which is the failure mode that wastes an evening.

These tests are static: they parse each notebook and resolve what it imports against the
real modules. No cell runs, so the whole file costs milliseconds and can run on every
change -- which is the only kind of test that catches drift, because drift arrives on the
commits nobody thought were risky.

The second test encodes the other half of the same lesson. A notebook launched on the
wrong interpreter fails in a way that names the symptom (`NameError`, once per downstream
cell) and never the cause (a sibling project's environment, chosen by the editor for the
whole workspace). The remedy is that every notebook has a pixi task, because a task cannot
pick an interpreter -- so the presence of that task is a contract worth asserting.
"""

import ast
import contextlib
import importlib
import tomllib
from pathlib import Path

import pytest

from theme.paths import ROOT

NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.py"))


def _package_imports(notebook: Path):
    """Every (module, name) this notebook imports from the `theme` package."""
    tree = ast.parse(notebook.read_text(), filename=str(notebook))
    return [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "theme"
        for alias in node.names
    ]


def _import_cases():
    return [
        pytest.param(notebook, module, name, id=f"{notebook.name}::{module}.{name}")
        for notebook in NOTEBOOKS
        for module, name in _package_imports(notebook)
    ]


def test_the_notebooks_are_discovered():
    """A guard on the guard: an empty parametrisation would pass in silence."""
    assert [path.name for path in NOTEBOOKS] == ["analysis.py", "gallery.py", "vision.py"]


@pytest.mark.parametrize(("notebook", "module", "name"), _import_cases())
def test_notebook_imports_resolve(notebook, module, name):
    """Every name a notebook imports from `theme` exists in the module it names."""
    imported = importlib.import_module(module)
    if not hasattr(imported, name):
        # `from theme import vision` names a submodule, which is an attribute of the package
        # only once something has imported it. Importing it IS the check.
        with contextlib.suppress(ModuleNotFoundError):
            importlib.import_module(f"{module}.{name}")
    assert hasattr(imported, name), (
        f"{notebook.name} imports `{name}` from {module}, which no longer defines it. "
        "A rename in the package is a rename in the notebooks."
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_every_notebook_has_a_task_that_launches_it(notebook):
    """`pixi run <task>` exists for each notebook, and points at that notebook."""
    tasks = tomllib.loads((ROOT / "pixi.toml").read_text())["tasks"]
    commands = [task["cmd"] if isinstance(task, dict) else task for task in tasks.values()]
    launching = [command for command in commands if f"notebooks/{notebook.name}" in command]
    assert launching, (
        f"no pixi task runs notebooks/{notebook.name}, so the only way to open it is to let "
        "an editor choose an interpreter -- which is how it gets run against a sibling "
        "project's environment, where `theme` does not exist."
    )
