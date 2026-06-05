#!/usr/bin/env python3
"""Regression test for app runners using caller-provided BUILD_DIR."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


RUNNERS = ("run_check.sh", "raise_check.sh", "dfg_check.sh")


def prepare_case(repo: Path, case_dir: Path, tmp_root: Path) -> Path:
    app_root = tmp_root / "test" / "app"
    app_root.mkdir(parents=True, exist_ok=True)
    shared = repo / "test" / "app" / "dfg_common.sh"
    shutil.copy2(shared, app_root / "dfg_common.sh")
    copied = app_root / case_dir.name
    shutil.copytree(case_dir, copied, ignore=shutil.ignore_patterns("build"))
    return copied


def run_runner(repo: Path, case_dir: Path, runner: str, build_dir: Path) -> None:
    env = os.environ.copy()
    env["BUILD_DIR"] = str(build_dir)
    env["LOOM_CC"] = str(repo / "build" / "bin" / "loom-cc")
    env["LOOM_CXX"] = str(repo / "build" / "bin" / "loom-c++")
    env["LOOM_RAISE"] = str(repo / "build" / "bin" / "loom-raise")
    env["LOOM_LOWER"] = str(repo / "build" / "bin" / "loom-lower")
    env["LOOM_RAISE_OPT"] = str(repo / "build" / "bin" / "loom-raise-opt")
    result = subprocess.run(
        ["bash", str(case_dir / runner)],
        cwd=case_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{case_dir.name}/{runner} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not build_dir.is_dir():
        raise AssertionError(f"{case_dir.name}/{runner} did not create BUILD_DIR={build_dir}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    app_root = repo / "test" / "app"
    cases = sorted(path for path in app_root.iterdir() if path.is_dir() and (path / "run_check.sh").is_file())
    with tempfile.TemporaryDirectory(prefix="loom-app-build-dir-") as tmp:
        tmp_root = Path(tmp)
        for case_dir in cases:
            copied_case = prepare_case(repo, case_dir, tmp_root / case_dir.name)
            default_build = copied_case / "build"
            for runner in RUNNERS:
                if not (copied_case / runner).is_file():
                    continue
                build_dir = tmp_root / case_dir.name / runner.removesuffix(".sh")
                run_runner(repo, copied_case, runner, build_dir)
                if default_build.exists():
                    raise AssertionError(
                        f"{case_dir.name}/{runner} ignored BUILD_DIR and touched {default_build}"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
