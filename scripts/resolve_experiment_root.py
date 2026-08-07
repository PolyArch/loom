#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


MINIMUM_SCRATCH_BYTES = 100 << 30


class ExperimentRootError(RuntimeError):
    pass


def git_ignored(repository: Path, path: Path) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(path)],
        cwd=repository,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def create_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ExperimentRootError(f"experiment root is not a directory: {path}")
    return path.resolve()


def is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def resolve_experiment_root(
    *,
    repository: Path,
    configured_root: Path | None,
    scratch_root: Path = Path("/scratch"),
    cache_root: Path | None = None,
    temporary_root: Path = Path("/tmp"),
) -> Path:
    repository = repository.resolve()
    repository_temp = (repository / "temp").resolve()
    temp_is_ignored = repository_temp.is_dir() and git_ignored(
        repository, repository_temp
    )

    if configured_root is not None:
        if not configured_root.is_absolute():
            raise ExperimentRootError("configured experiment root must be absolute")
        configured_root = configured_root.resolve()
        if is_within(configured_root, repository) and not (
            temp_is_ignored and is_within(configured_root, repository_temp)
        ):
            raise ExperimentRootError(
                "a repository-local experiment root must be beneath the ignored "
                "repository temp directory"
            )
        try:
            return create_directory(configured_root)
        except OSError as error:
            raise ExperimentRootError(
                f"could not create configured experiment root: {error}"
            ) from error

    if temp_is_ignored:
        return repository_temp

    if scratch_root.is_dir():
        try:
            scratch_free = shutil.disk_usage(scratch_root).free
        except OSError:
            scratch_free = 0
        if scratch_free > MINIMUM_SCRATCH_BYTES:
            try:
                return create_directory(scratch_root / f"loom-{os.getuid()}")
            except OSError:
                pass

    if cache_root is not None:
        try:
            return create_directory(cache_root)
        except OSError:
            pass

    try:
        return create_directory(temporary_root / f"loom-{os.getuid()}")
    except OSError as error:
        raise ExperimentRootError(
            f"could not create a fallback experiment root: {error}"
        ) from error


def configured_root_from_file(path: Path | None) -> Path | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentRootError(
            f"could not read local configuration: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ExperimentRootError("local configuration must be a JSON object")
    if value.get("schema") != "loom.local_tool_config" or value.get("version") != "1.0":
        raise ExperimentRootError("local configuration has the wrong schema or version")
    configured = value.get("experiment_root")
    if configured is None:
        return None
    if not isinstance(configured, str):
        raise ExperimentRootError("experiment_root must be a string")
    root = Path(configured)
    if not root.is_absolute():
        raise ExperimentRootError("experiment_root must be an absolute path")
    return root


def repository_root(path: str | None) -> Path:
    if path:
        return Path(path)
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ExperimentRootError("current directory is not in a Git worktree")
    return Path(completed.stdout.strip())


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root")
    parser.add_argument("--loom-local-config", type=Path)
    parsed = parser.parse_args(arguments)
    try:
        root = resolve_experiment_root(
            repository=repository_root(parsed.repository_root),
            configured_root=configured_root_from_file(parsed.loom_local_config),
            cache_root=Path.home() / ".cache" / "loom",
        )
    except ExperimentRootError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
