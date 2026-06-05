#!/usr/bin/env python3
"""Shared helpers for app artifact summary producers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


def discover_app_cases(*required_scripts: str) -> list[str]:
    app_root = ROOT / "test" / "app"
    return sorted(
        path.name
        for path in app_root.iterdir()
        if all((path / script).is_file() for script in required_scripts)
    )


def env_path(env_name: str, fallback: Path | str) -> str:
    value = os.environ.get(env_name)
    if value:
        return value
    return str(fallback)


def build_tool_path(env_name: str, fallback_name: str) -> str:
    return env_path(env_name, ROOT / "build" / "bin" / fallback_name)


def run_bash_script(
    script: Path,
    env: dict[str, str],
    *,
    cwd: Path | None = None,
) -> tuple[str, str]:
    result = subprocess.run(
        ["bash", str(script)],
        cwd=cwd or script.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        return "pass", result.stdout.strip()
    detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
    return "fail", detail[0] if detail else f"{script.name} exited {result.returncode}"


def write_rows(kind: str, output: Path, rows: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows(kind, output, rows)


def write_empty(kind: str, output: str) -> None:
    intermediate_artifacts.write_csv(kind, intermediate_artifacts.output_path(output))
