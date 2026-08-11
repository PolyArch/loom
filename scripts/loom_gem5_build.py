#!/usr/bin/env python3
"""Build and verify Loom's pinned out-of-tree gem5 component."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass


SCONS_MODULE = "scons/4.10.1"
READINESS_SCHEMA = "loom.gem5_build_readiness.1"
GEM5_TARGET = "RISCV"
GEM5_VARIANT = "opt"


class BuildError(RuntimeError):
    pass


def run_text(command: list[str], *, cwd: pathlib.Path, env=None) -> str:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except FileNotFoundError as error:
        raise BuildError(f"missing executable: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        output = (error.stdout or "").strip()
        raise BuildError(
            f"command failed ({error.returncode}): {' '.join(command)}"
            + (f"\n{output}" if output else "")
        ) from error


def primary_worktree(repository_root: pathlib.Path) -> pathlib.Path:
    output = run_text(
        ["git", "worktree", "list", "--porcelain"], cwd=repository_root
    )
    for line in output.splitlines():
        if line.startswith("worktree "):
            return pathlib.Path(line.removeprefix("worktree ")).resolve()
    raise BuildError("git did not report a primary worktree")


def gem5_gitlink(repository_root: pathlib.Path) -> str:
    output = run_text(
        ["git", "ls-tree", "HEAD", "--", "externals/gem5"],
        cwd=repository_root,
    )
    fields = output.split()
    if len(fields) < 4 or fields[0] != "160000" or fields[1] != "commit":
        raise BuildError("HEAD does not contain the externals/gem5 gitlink")
    return fields[2]


def resolve_gem5_source(repository_root: pathlib.Path) -> pathlib.Path:
    local = repository_root / "externals" / "gem5"
    if (local / ".git").exists() or (local / "SConstruct").is_file():
        source = local
    else:
        source = primary_worktree(repository_root) / "externals" / "gem5"
    if not (source / "SConstruct").is_file():
        raise BuildError(f"gem5 source checkout is unavailable at {source}")
    return source.resolve()


def validate_gem5_source(
    repository_root: pathlib.Path, source: pathlib.Path
) -> str:
    expected = gem5_gitlink(repository_root)
    actual = run_text(["git", "rev-parse", "HEAD"], cwd=source)
    if actual != expected:
        raise BuildError(
            f"gem5 checkout {actual} does not match gitlink {expected}"
        )
    status = run_text(["git", "status", "--short"], cwd=source)
    if status:
        raise BuildError("gem5 checkout is dirty; out-of-tree builds require a clean pin")
    return expected


def module_environment() -> dict[str, str]:
    command = (
        "source /etc/profile.d/modules.sh && "
        f"module load {SCONS_MODULE} && env -0"
    )
    try:
        output = subprocess.run(
            ["bash", "-lc", command],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise BuildError(
            f"could not load required module {SCONS_MODULE}: {detail}"
        ) from error
    environment: dict[str, str] = {}
    for entry in output.split(b"\0"):
        if not entry:
            continue
        key, value = entry.split(b"=", 1)
        environment[key.decode()] = value.decode(errors="surrogateescape")
    return environment


def find_executable(name: str, environment: dict[str, str]) -> str:
    executable = shutil.which(name, path=environment.get("PATH"))
    if not executable:
        raise BuildError(f"{name} is unavailable after loading {SCONS_MODULE}")
    return str(pathlib.Path(executable).resolve())


def hash_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bridge_source_digest(repository_root: pathlib.Path) -> str:
    roots = [
        repository_root / "runtime" / "gem5",
        repository_root / "include" / "Runtime" / "Gem5BridgeWire.h",
    ]
    files: list[pathlib.Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
        else:
            raise BuildError(f"required bridge source is missing: {root}")
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(repository_root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True)
class BuildPaths:
    root: pathlib.Path
    source: pathlib.Path
    build_directory: pathlib.Path
    binary: pathlib.Path
    readiness: pathlib.Path


def build_paths(repository_root: pathlib.Path) -> BuildPaths:
    root = primary_worktree(repository_root) / "build" / "gem5"
    build_directory = root / "build" / GEM5_TARGET
    return BuildPaths(
        root=root,
        source=resolve_gem5_source(repository_root),
        build_directory=build_directory,
        binary=build_directory / f"gem5.{GEM5_VARIANT}",
        readiness=root / "loom-gem5-readiness.json",
    )


def expected_readiness(
    repository_root: pathlib.Path,
    paths: BuildPaths,
    environment: dict[str, str],
) -> dict:
    gem5_commit = validate_gem5_source(repository_root, paths.source)
    scons = find_executable("scons", environment)
    compiler = find_executable("c++", environment)
    scons_version = run_text([scons, "--version"], cwd=repository_root, env=environment)
    compiler_version = run_text(
        [compiler, "--version"], cwd=repository_root, env=environment
    ).splitlines()[0]
    configuration = {
        "bridge_source_digest": bridge_source_digest(repository_root),
        "compiler": {"path": compiler, "version": compiler_version},
        "extras_contract": "runtime/gem5",
        "gem5_commit": gem5_commit,
        "python": sys.version,
        "scons": {"module": SCONS_MODULE, "path": scons, "version": scons_version},
        "target": GEM5_TARGET,
        "variant": GEM5_VARIANT,
    }
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return {
        "schema": READINESS_SCHEMA,
        "gem5_repository_identity": "https://gem5.googlesource.com/public/gem5",
        "gem5_full_commit_identity": gem5_commit,
        "build_configuration_digest": hashlib.sha256(encoded.encode()).hexdigest(),
        "configuration": configuration,
        "binary": str(paths.binary),
    }


def inspect_readiness(paths: BuildPaths, expected: dict) -> tuple[bool, str]:
    if not paths.binary.is_file():
        return False, "binary is missing"
    if not paths.readiness.is_file():
        return False, "readiness stamp is missing"
    try:
        recorded = json.loads(paths.readiness.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "readiness stamp is malformed"
    binary_digest = hash_file(paths.binary)
    if recorded.get("binary_sha256") != binary_digest:
        return False, "binary digest changed"
    comparison = dict(recorded)
    comparison.pop("binary_sha256", None)
    comparison.pop("version_probe", None)
    if comparison != expected:
        return False, "source, tool, or build configuration identity changed"
    if not isinstance(recorded.get("version_probe"), str):
        return False, "version probe is absent"
    return True, "ready"


def atomic_write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build(
    repository_root: pathlib.Path,
    paths: BuildPaths,
    environment: dict[str, str],
    expected: dict,
    jobs: int,
) -> None:
    scons = expected["configuration"]["scons"]["path"]
    paths.build_directory.mkdir(parents=True, exist_ok=True)
    command = [
        scons,
        "-C",
        str(paths.source),
        "--ignore-style",
        f"EXTRAS={repository_root / 'runtime' / 'gem5'}",
        f"-j{jobs}",
        str(paths.binary),
    ]
    subprocess.run(command, cwd=repository_root, env=environment, check=True)
    if not paths.binary.is_file():
        raise BuildError("SCons completed without producing gem5.opt")
    version = run_text(
        [str(paths.binary), "--build-info"], cwd=repository_root, env=environment
    )
    if "gem5" not in version.lower():
        raise BuildError("built binary did not identify itself as gem5")
    record = dict(expected)
    record["binary_sha256"] = hash_file(paths.binary)
    record["version_probe"] = version
    atomic_write_json(paths.readiness, record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-binary", action="store_true")
    arguments = parser.parse_args()
    repository_root = arguments.repository_root.resolve()
    try:
        paths = build_paths(repository_root)
        environment = module_environment()
        expected = expected_readiness(repository_root, paths, environment)
        ready, reason = inspect_readiness(paths, expected)
        if arguments.check:
            print(f"gem5 readiness: {reason}")
            if arguments.print_binary:
                print(paths.binary)
            return 0 if ready else 1
        if arguments.force or not ready:
            print(f"building gem5: {'forced' if arguments.force else reason}")
            build(
                repository_root,
                paths,
                environment,
                expected,
                max(1, min(arguments.jobs, os.cpu_count() or 1)),
            )
        else:
            print(f"reusing gem5: {paths.binary}")
        if arguments.print_binary:
            print(paths.binary)
        return 0
    except (BuildError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
