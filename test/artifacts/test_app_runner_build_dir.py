#!/usr/bin/env python3
"""Regression test for app runners using caller-provided BUILD_DIR."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import artifact_test_common


RUNNERS = ("run_check.sh", "raise_check.sh", "dfg_check.sh")
AGGREGATE_RUNNERS = ("run_all.sh", "run_raise_all.sh", "run_dfg_all.sh")
SHARED_APP_SCRIPTS = (
    "dfg_common.sh",
    "run_c_variants_common.sh",
    "run_cxx_variants_common.sh",
)


def prepare_case(repo: Path, case_dir: Path, tmp_root: Path) -> Path:
    app_root = tmp_root / "test" / "app"
    app_root.mkdir(parents=True, exist_ok=True)
    for name in SHARED_APP_SCRIPTS:
        shutil.copy2(repo / "test" / "app" / name, app_root / name)
    copied = app_root / case_dir.name
    shutil.copytree(case_dir, copied, ignore=shutil.ignore_patterns("build"))
    return copied


def prepare_app_tree(repo: Path, tmp_root: Path) -> Path:
    app_root = repo / "test" / "app"
    copied_root = tmp_root / "test" / "app"
    copied_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "app_manifest.py",
        "manifest.json",
        "run_all.sh",
        "run_raise_all.sh",
        "run_dfg_all.sh",
        *SHARED_APP_SCRIPTS,
    ):
        shutil.copy2(app_root / name, copied_root / name)
    for case_dir in sorted(path for path in app_root.iterdir() if path.is_dir()):
        shutil.copytree(case_dir, copied_root / case_dir.name, ignore=shutil.ignore_patterns("build"))
    return copied_root


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


def run_aggregate(repo: Path, app_root: Path, runner: str) -> None:
    env = os.environ.copy()
    env.pop("BUILD_DIR", None)
    env["LOOM_CC"] = str(repo / "build" / "bin" / "loom-cc")
    env["LOOM_CXX"] = str(repo / "build" / "bin" / "loom-c++")
    env["LOOM_RAISE"] = str(repo / "build" / "bin" / "loom-raise")
    env["LOOM_LOWER"] = str(repo / "build" / "bin" / "loom-lower")
    env["LOOM_RAISE_OPT"] = str(repo / "build" / "bin" / "loom-raise-opt")
    result = subprocess.run(
        ["bash", str(app_root / runner)],
        cwd=app_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{runner} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    app_root = repo / "test" / "app"
    cases = sorted(path for path in app_root.iterdir() if path.is_dir() and (path / "run_check.sh").is_file())
    with artifact_test_common.repo_temp_dir(repo, "loom-app-build-dir-") as tmp:
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

        copied_app = prepare_app_tree(repo, tmp_root / "aggregate")
        for runner in AGGREGATE_RUNNERS:
            run_aggregate(repo, copied_app, runner)
        touched_defaults = sorted(
            str(path.relative_to(copied_app))
            for path in copied_app.iterdir()
            if path.is_dir() and (path / "build").exists()
        )
        if touched_defaults:
            raise AssertionError(f"aggregate runners touched case-local build dirs: {touched_defaults}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
