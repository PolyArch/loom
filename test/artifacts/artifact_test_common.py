#!/usr/bin/env python3
"""Shared helpers for artifact regression tests."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def run_command(
    repo: Path,
    argv: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require_success(
    repo: Path,
    argv: list[str],
    label: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = run_command(repo, argv, env)
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed with {result.returncode}\n"
            f"command: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def find_tool(repo: Path, name: str) -> Path:
    candidates = (
        repo / "build" / "tools" / name / name,
        repo / "build" / "bin" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AssertionError(f"missing {name}: checked {list(candidates)}")


def repo_temp_dir(repo: Path, prefix: str) -> tempfile.TemporaryDirectory[str]:
    temp_root = repo / "build" / "test-runs"
    temp_root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=temp_root)
