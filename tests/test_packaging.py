"""Tests for how the distribution is built.

The framework is one source file. It lives in a package directory purely so
that PEP 561 can work: a type checker only trusts a distribution's inline
annotations when it finds a `py.typed` marker, and a marker can only be
attached to a package, not to a top-level module. That is easy to undo by
accident - a stray `py-modules` entry, a missing `package-data` line - and the
symptom appears only in someone else's editor, so it is checked here.

    pytest tests/test_packaging.py
"""

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The source layout
# ---------------------------------------------------------------------------
def test_the_framework_is_still_one_source_file():
    sources = sorted(p.name for p in (_ROOT / "unchained").glob("*.py"))
    assert sources == ["__init__.py"], f"the core grew a second module: {sources}"


def test_the_py_typed_marker_exists_and_is_a_marker():
    marker = _ROOT / "unchained" / "py.typed"
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").strip() == ""  # PEP 561: presence is the signal


def test_pyproject_ships_the_package_and_the_marker():
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["unchained"]' in pyproject
    assert 'unchained = ["py.typed"]' in pyproject
    # py-modules would build a bare unchained.py again, which cannot carry a
    # marker - the exact regression this file exists to catch.
    assert "py-modules" not in pyproject


def test_mypy_checks_the_package_not_a_stale_path():
    assert 'files = ["unchained"]' in (_ROOT / "pyproject.toml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The built wheel
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    """Build a real wheel, or skip if the build backend is unavailable."""
    pytest.importorskip("build", reason="pip install -e '.[dev]' to run the wheel tests")
    # setuptools reuses ./build/lib between builds and never prunes it, so a
    # file left there by an earlier layout is copied into the new wheel. That
    # is not hypothetical: it silently produced a wheel carrying both a stale
    # flat unchained.py and the package. Both paths are gitignored build
    # artifacts, and rebuilding them costs nothing.
    for stale in ("build", "unchained_ai.egg-info"):
        shutil.rmtree(_ROOT / stale, ignore_errors=True)
    out = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(_ROOT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:  # pragma: no cover - only on a broken toolchain
        pytest.skip(f"wheel build failed: {result.stderr[-400:]}")
    built = sorted(out.glob("*.whl"))
    assert built, "no wheel produced"
    return built[0]


def test_the_wheel_contains_the_py_typed_marker(wheel):
    # Without this, `pip install unchained-ai` gives downstream code no types
    # at all - every symbol becomes Any and nothing is checked.
    names = zipfile.ZipFile(wheel).namelist()
    assert "unchained/py.typed" in names


def test_the_wheel_ships_the_source_as_a_package(wheel):
    names = zipfile.ZipFile(wheel).namelist()
    assert "unchained/__init__.py" in names
    assert "unchained.py" not in names  # a flat module cannot carry the marker


def test_the_wheel_carries_no_surprises(wheel):
    payload = [
        name
        for name in zipfile.ZipFile(wheel).namelist()
        if not name.startswith("unchained_ai-") and not name.endswith("/")
    ]
    assert sorted(payload) == ["unchained/__init__.py", "unchained/py.typed"]


def test_the_installed_package_is_importable_and_typed(wheel, tmp_path):
    """Unpack the wheel where a checker would find it, and confirm both."""
    site = tmp_path / "site-packages"
    site.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)

    assert (site / "unchained" / "py.typed").is_file()

    # It still imports as `unchained`, from the installed layout.
    probe = subprocess.run(
        [sys.executable, "-c", "import unchained; print(unchained.__version__)"],
        capture_output=True,
        text=True,
        cwd=str(site),
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip()


def _mypy_reveal(python: Path, workdir: Path, cache: Path) -> str:
    """Run mypy against `python`'s site-packages, from outside this repository.

    Both details matter. ``--python-executable`` is what makes mypy apply PEP
    561 to an *installed* distribution; running from a neutral directory stops
    it finding this repository's own ``unchained/`` on the search path and
    reporting types that no installed package provided. An earlier version of
    this test did neither and passed with the marker deleted.
    """
    consumer = workdir / "consumer.py"
    consumer.write_text(
        "from unchained import Agent\n\ndef check(a: Agent) -> None:\n    reveal_type(a.run)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-incremental",
            "--cache-dir",
            str(cache),
            "--python-executable",
            str(python),
            "consumer.py",
        ],
        capture_output=True,
        text=True,
        cwd=str(workdir),
    )
    return result.stdout + result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None, reason="pip install -e '.[dev]' for the type check"
)
def test_a_type_checker_resolves_the_installed_package(wheel, tmp_path):
    """The end of the story: a consumer's checker sees real types, not Any.

    Asserted both ways round. Without the second half this test would pass
    for the wrong reason - which is exactly what happened while writing it.
    """
    environment = tmp_path / "env"
    created = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:  # pragma: no cover - only on a broken toolchain
        pytest.skip(f"could not create a probe environment: {created.stderr[-300:]}")

    site = next(environment.glob("**/site-packages"), None)
    assert site is not None, "no site-packages in the probe environment"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)

    python = next(environment.glob("**/python.exe"), None) or next(
        environment.glob("**/bin/python")
    )
    workdir = tmp_path / "consumer"
    workdir.mkdir()

    typed = _mypy_reveal(python, workdir, tmp_path / "cache-typed")
    assert "missing library stubs or py.typed marker" not in typed, typed
    assert 'Revealed type is "Any"' not in typed, typed
    assert "user_input" in typed, typed  # the real signature came through

    # And prove the marker is what did it.
    (site / "unchained" / "py.typed").unlink()
    untyped = _mypy_reveal(python, workdir, tmp_path / "cache-untyped")
    assert 'Revealed type is "Any"' in untyped, untyped
