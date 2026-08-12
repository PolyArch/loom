#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


MINIMUM_SCRATCH_BYTES = 100 << 30
EXTERNAL_TOOL_CACHE_ROOT_ENVIRONMENT = "LOOM_EXTERNAL_TOOL_CACHE_ROOT"
EXTERNAL_TOOL_CACHE_DIRECTORY = "external-tool-cache"
EXTERNAL_TOOL_CACHE_MARKER = ".loom-external-tool-result-cache"
EXTERNAL_TOOL_CACHE_MARKER_CONTENTS = "loom.external_tool_result_cache 1.0\n"
EXTERNAL_TOOL_CACHE_MEMBERS = frozenset(
    {
        EXTERNAL_TOOL_CACHE_MARKER,
        ".loom-external-tool-result-cache.lock",
        "command-entries",
        "command-locks",
        "entries",
        "locks",
    }
)


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
    repository_build = (repository / "build").resolve()
    build_is_ignored = repository_build.is_dir() and git_ignored(
        repository, repository_build
    )

    if configured_root is not None:
        if not configured_root.is_absolute():
            raise ExperimentRootError("configured experiment root must be absolute")
        configured_root = configured_root.resolve()
        if is_within(configured_root, repository) and not git_ignored(
            repository, configured_root
        ):
            raise ExperimentRootError(
                "a repository-local experiment root must be Git-ignored"
            )
        try:
            return create_directory(configured_root)
        except OSError as error:
            raise ExperimentRootError(
                f"could not create configured experiment root: {error}"
            ) from error

    if build_is_ignored:
        return repository_build

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


def resolve_external_tool_cache_root(
    *,
    repository: Path,
    configured_root: Path | None,
    environment: Mapping[str, str] = os.environ,
    scratch_root: Path = Path("/scratch"),
    cache_root: Path | None = None,
    temporary_root: Path = Path("/tmp"),
) -> Path:
    override = environment.get(EXTERNAL_TOOL_CACHE_ROOT_ENVIRONMENT)
    if override:
        selected = Path(override)
        if not selected.is_absolute():
            raise ExperimentRootError(
                f"{EXTERNAL_TOOL_CACHE_ROOT_ENVIRONMENT} must be absolute"
            )
        return Path(os.path.abspath(selected))
    experiment_root = resolve_experiment_root(
        repository=repository,
        configured_root=configured_root,
        scratch_root=scratch_root,
        cache_root=cache_root,
        temporary_root=temporary_root,
    )
    return (experiment_root / EXTERNAL_TOOL_CACHE_DIRECTORY).resolve(strict=False)


def remove_external_tool_cache_root(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_dir():
        raise ExperimentRootError(
            f"external-tool cache root is not an ordinary directory: {path}"
        )
    resolved = path.resolve()
    if resolved == resolved.parent or resolved == Path.home().resolve():
        raise ExperimentRootError(
            f"refusing to remove a broad external-tool cache root: {path}"
        )
    marker = path / EXTERNAL_TOOL_CACHE_MARKER
    try:
        marker_status = marker.stat(follow_symlinks=False)
        marker_contents = marker.read_text(encoding="utf-8")
    except OSError as error:
        raise ExperimentRootError(
            f"refusing to remove unmarked external-tool cache root {path}: {error}"
        ) from error
    if not stat.S_ISREG(marker_status.st_mode) or marker.is_symlink():
        raise ExperimentRootError(
            f"refusing to remove invalid external-tool cache root {path}"
        )
    if marker_contents != EXTERNAL_TOOL_CACHE_MARKER_CONTENTS:
        raise ExperimentRootError(
            f"refusing to remove incompatible external-tool cache root {path}"
        )
    members = {member.name for member in path.iterdir()}
    if not members <= EXTERNAL_TOOL_CACHE_MEMBERS:
        raise ExperimentRootError(
            f"refusing to remove external-tool cache root with foreign members: {path}"
        )
    shutil.rmtree(path)
    return True


def is_external_tool_cache_root(path: Path) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    marker = path / EXTERNAL_TOOL_CACHE_MARKER
    try:
        marker_status = marker.stat(follow_symlinks=False)
        marker_contents = marker.read_text(encoding="utf-8")
        members = {member.name for member in path.iterdir()}
    except OSError:
        return False
    return (
        stat.S_ISREG(marker_status.st_mode)
        and not marker.is_symlink()
        and marker_contents == EXTERNAL_TOOL_CACHE_MARKER_CONTENTS
        and members <= EXTERNAL_TOOL_CACHE_MEMBERS
    )


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
    if value.get("schema") != "loom.local_tool_config" or value.get("version") not in (
        "1.0",
        "1.1",
    ):
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
