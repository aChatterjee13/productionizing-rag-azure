"""Repository wiring invariants: Makefile, CI workflow, packaging and web scripts.

These assertions belong to no single service — they check that the *operational*
entry points a developer or CI actually runs resolve to things that exist. They
live under ``packages/ragcore/tests`` because ragcore is the shared package every
member depends on and this is a collected test path; nothing here imports a
service-private symbol.

Each test pins one divergence found by the documentation audit against
``docs/CONTRACTS.md``:

* Addendum I — "The root ``Makefile`` targets ``bootstrap``, ``seed``, ``smoke``
  and ``ingest-local`` must point at these four paths."
* Addendum L — "``packages/ragcore/pyproject.toml`` needs ``prometheus-client``
  in ``dependencies``."
* Addendum I — "``web/package.json`` must expose a ``build`` script ... and may
  expose ``lint`` and ``typecheck``, which CI runs with ``--if-present``."
* ``--strict-markers`` is on, so a marker CI names must be one a test carries.
"""

from __future__ import annotations

import importlib.util
import json
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MAKEFILE = REPO_ROOT / "Makefile"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
RAGCORE_PYPROJECT = REPO_ROOT / "packages" / "ragcore" / "pyproject.toml"
WEB_PACKAGE_JSON = REPO_ROOT / "web" / "package.json"

# `scripts/<name>.py` anywhere in the Makefile.
_SCRIPT_RE = re.compile(r"scripts/[A-Za-z0-9_./-]+\.py")
# `python -m <module>` — the module path a target executes.
_DASH_M_RE = re.compile(r"-m\s+([A-Za-z_][A-Za-z0-9_.]*)")
# `uvicorn <module>:<attribute>`.
_ASGI_RE = re.compile(r"uvicorn\s+([A-Za-z_][A-Za-z0-9_.]*):([A-Za-z_][A-Za-z0-9_]*)")
# `>=20.19` / `>=20.19.0` at the start of an engines range.
_LOWER_BOUND_RE = re.compile(r">=\s*(\d+)\.(\d+)(?:\.(\d+))?")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _web_package() -> dict:
    return json.loads(_read(WEB_PACKAGE_JSON))


def _normalise(name: str) -> str:
    """Fold a distribution name to PEP 503 form so `_` and `-` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


# ------------------------------------------------------------------- Makefile


def test_makefile_scripts_exist() -> None:
    """Every `scripts/*.py` a Makefile target runs must be a real file.

    `make bootstrap`, `make seed` and `make smoke` are documented quickstart
    steps; a stale short name makes all three fail at the shell.
    """
    referenced = sorted(set(_SCRIPT_RE.findall(_read(MAKEFILE))))
    assert referenced, "the Makefile should still drive the operational scripts"
    missing = [name for name in referenced if not (REPO_ROOT / name).is_file()]
    assert not missing, f"Makefile references non-existent scripts: {missing}"


def test_makefile_names_the_four_contract_scripts() -> None:
    """Addendum I fixes the four script paths the Makefile must invoke."""
    text = _read(MAKEFILE)
    for name in (
        "scripts/bootstrap_qdrant.py",
        "scripts/seed_demo_tenant.py",
        "scripts/smoke_test.py",
    ):
        assert name in text, f"{name} is the path docs/CONTRACTS.md pins"


@pytest.mark.parametrize("module", sorted(set(_DASH_M_RE.findall(_read(MAKEFILE)))))
def test_makefile_python_modules_resolve(module: str) -> None:
    """`python -m <module>` targets must name an importable module."""
    if "." in module:
        # `eval.run` is an alias registered by `eval/__init__.py`'s meta-path
        # finder, so the parent has to be imported before the child resolves.
        importlib.import_module(module.split(".", 1)[0])
    assert importlib.util.find_spec(module) is not None, f"{module} is not importable"


def test_makefile_asgi_target_resolves() -> None:
    """`make api` must name the ASGI application the Dockerfile also starts."""
    match = _ASGI_RE.search(_read(MAKEFILE))
    assert match is not None, "the api target should still start uvicorn"
    module, attribute = match.groups()
    assert importlib.util.find_spec(module) is not None
    imported = importlib.import_module(module)
    assert hasattr(imported, attribute), f"{module} has no attribute {attribute}"


def test_makefile_alembic_config_exists() -> None:
    """The migrate/downgrade/revision targets share one alembic.ini path."""
    match = re.search(r"ALEMBIC_INI\s*:=\s*(\S+)", _read(MAKEFILE))
    assert match is not None
    assert (REPO_ROOT / match.group(1)).is_file()


# ------------------------------------------------------------------- packaging


def test_prometheus_client_is_a_ragcore_dependency() -> None:
    """Addendum L requires `prometheus-client` in ragcore's dependencies.

    Without it `PROMETHEUS_AVAILABLE` is False and `/metrics` serves a single
    comment line, so all seventeen `rag_*` series are silently dead.
    """
    declared = {
        _normalise(re.split(r"[<>=!\[;\s]", spec, maxsplit=1)[0])
        for spec in _toml(RAGCORE_PYPROJECT)["project"]["dependencies"]
    }
    assert "prometheus-client" in declared


def test_metrics_registry_renders_real_series() -> None:
    """`render_metrics()` must emit samples, not the disabled stub."""
    from ragcore.observability import metrics

    assert metrics.PROMETHEUS_AVAILABLE is True
    metrics.observe_guardrail(stage="input", kind="prompt_injection", action="blocked")
    body = metrics.render_metrics().decode("utf-8")
    assert "rag_guardrail_events_total" in body
    assert "prometheus_client is not installed" not in body


# ------------------------------------------------------------------------- web


def test_web_exposes_build_lint_and_typecheck_scripts() -> None:
    """CI runs `npm run lint --if-present`; without the script it passes blind."""
    scripts = _web_package()["scripts"]
    for name in ("build", "lint", "typecheck"):
        assert name in scripts, f"web/package.json is missing the {name!r} script"
    assert "tsc" in scripts["lint"] or "eslint" in scripts["lint"], (
        "the lint script must run a real checker"
    )


def test_web_engines_are_fully_qualified_and_admit_the_ci_toolchain() -> None:
    """`engines` must be unambiguous and admit the Node major CI installs.

    A partial bound such as `>=20.19` reads to a human as "the 20.19 line" even
    though semver expands it to `>=20.19.0`; spelling the patch out removes the
    ambiguity the audit flagged.
    """
    engines = _web_package()["engines"]
    ci = _read(CI_WORKFLOW)
    node_major = int(re.search(r'NODE_VERSION:\s*"(\d+)"', ci).group(1))

    for key, floor_major in (("node", node_major), ("npm", 0)):
        match = _LOWER_BOUND_RE.search(engines[key])
        assert match is not None, f"engines.{key} needs a >= lower bound"
        assert match.group(3) is not None, (
            f"engines.{key} = {engines[key]!r} is a partial version; "
            "write the patch component out"
        )
        assert int(match.group(1)) <= floor_major or floor_major == 0


# -------------------------------------------------------------------- markers


def test_ci_only_names_pytest_markers_that_tests_actually_carry() -> None:
    """A marker CI selects on, or advertises, must be one some test applies.

    `-m "not llm"` and a comment promising that "integration-marked" tests run
    are both dead weight while no test carries either marker: the filter selects
    everything and the comment describes coverage that does not exist.
    """
    registered = [
        entry.split(":", 1)[0].strip()
        for entry in _toml(ROOT_PYPROJECT)["tool"]["pytest"]["ini_options"]["markers"]
    ]
    ci = _read(CI_WORKFLOW)
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in REPO_ROOT.rglob("test_*.py")
        if ".venv" not in path.parts
    )
    for marker in registered:
        if re.search(rf"\b{re.escape(marker)}\b", ci) is None:
            continue
        assert re.search(rf"@pytest\.mark\.{re.escape(marker)}\b", sources), (
            f"ci.yml names the {marker!r} marker but no test carries it"
        )
