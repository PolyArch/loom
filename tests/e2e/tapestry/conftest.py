"""Shared pytest fixtures for Tapestry end-to-end integration tests."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    """Walk up from this file to find the repository root (contains CMakeLists.txt)."""
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "tools" / "tapestry").exists():
            return p
        p = p.parent
    raise RuntimeError("Cannot locate repository root from conftest.py")


REPO_ROOT = _find_repo_root()


def _find_build_dir() -> Path:
    """Locate the build directory. Checks LOOM_BUILD_DIR env, then default 'build/'."""
    env_val = os.environ.get("LOOM_BUILD_DIR", "")
    if env_val:
        p = Path(env_val)
        if p.is_dir():
            return p
    default = REPO_ROOT / "build"
    if default.is_dir():
        return default
    raise RuntimeError(
        "Cannot find build directory. Set LOOM_BUILD_DIR or build under repo/build."
    )


BUILD_DIR = _find_build_dir()
BIN_DIR = BUILD_DIR / "bin"
DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root path."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def build_dir() -> Path:
    """Build directory path."""
    return BUILD_DIR


@pytest.fixture(scope="session")
def bin_dir() -> Path:
    """Path to directory containing built binaries."""
    return BIN_DIR


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to test data directory."""
    return DATA_DIR


@pytest.fixture(scope="session")
def tapestry_pipeline_bin() -> Path:
    """Path to the tapestry-pipeline binary."""
    p = BIN_DIR / "tapestry-pipeline"
    if not p.exists():
        pytest.skip("tapestry-pipeline binary not found; build first")
    return p


@pytest.fixture(scope="session")
def tapestry_compile_bin() -> Path:
    """Path to the tapestry-compile binary."""
    p = BIN_DIR / "tapestry-compile"
    if not p.exists():
        pytest.skip("tapestry-compile binary not found; build first")
    return p


@pytest.fixture(scope="session")
def tapestry_simulate_bin() -> Path:
    """Path to the tapestry-simulate binary."""
    p = BIN_DIR / "tapestry-simulate"
    if not p.exists():
        pytest.skip("tapestry-simulate binary not found; build first")
    return p


@pytest.fixture(scope="session")
def tapestry_rtlgen_bin() -> Path:
    """Path to the tapestry-rtlgen binary."""
    p = BIN_DIR / "tapestry-rtlgen"
    if not p.exists():
        pytest.skip("tapestry-rtlgen binary not found; build first")
    return p


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory for a single test."""
    out = tmp_path / "tapestry_output"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def arch_1core_json() -> Path:
    """Path to single-core architecture JSON."""
    return DATA_DIR / "arch_1core.json"


@pytest.fixture(scope="session")
def arch_2x2_json() -> Path:
    """Path to 2x2 system architecture JSON."""
    return DATA_DIR / "arch_2x2.json"


@pytest.fixture(scope="session")
def simple_2kernel_mlir() -> Path:
    """Path to simple 2-kernel TDG MLIR file."""
    return DATA_DIR / "simple_2kernel.mlir"
