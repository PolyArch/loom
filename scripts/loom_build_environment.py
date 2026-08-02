#!/usr/bin/env python3
"""Host toolchain and filesystem discovery for Loom build dispatching."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_GIT = (2, 7)
LLVM_C_COMPILER = "gcc"
LLVM_CXX_COMPILER = "g++"
LLVM_GCC_MIN = (7, 4)
LOOM_C_COMPILER = "clang"
LOOM_CXX_COMPILER = "clang++"
LOOM_CLANG_MIN = (21, 1, 8)


def info(message: str) -> None:
    print(f"info: {message}", file=sys.stderr)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def die(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def real(path: Path) -> Path:
    return Path(os.path.realpath(str(path)))


def format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def normalize_version(version: tuple[int, ...], width: int) -> tuple[int, ...]:
    return version + (0,) * max(0, width - len(version))


def version_less(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> bool:
    width = max(len(lhs), len(rhs))
    return normalize_version(lhs, width) < normalize_version(rhs, width)


def parse_version(output: str) -> tuple[int, ...] | None:
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", output)
    if not match:
        return None
    return tuple(int(part) for part in match.groups(default="0"))


def compiler_version(tool: str) -> tuple[tuple[int, ...], str]:
    try:
        output = subprocess.check_output([tool, "--version"], stderr=subprocess.STDOUT).decode(errors="replace")
    except FileNotFoundError:
        die(f"{tool} not found on PATH")
    except subprocess.CalledProcessError as error:
        detail = error.output.decode(errors="replace").strip()
        die(f"could not run {tool} --version: {detail or error}")
    version = parse_version(output)
    if version is None:
        die(f"could not parse {tool} version from {output.splitlines()[0]!r}")
    return version, output.splitlines()[0].strip()


def resolve_compiler_executable(tool: str) -> str:
    """Resolve the compiler itself, not a PATH-level launcher symlink."""
    if os.path.dirname(tool):
        candidates = [Path(tool)]
    else:
        candidates = [Path(directory) / tool for directory in os.get_exec_path()]

    for candidate in candidates:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        resolved = real(candidate)
        if resolved.name == "ccache":
            continue
        return str(candidate.absolute())

    die(f"could not resolve compiler {tool} on PATH without a ccache wrapper")


def check_compiler(tool: str, nice_name: str, minimum: tuple[int, ...]) -> tuple[str, str]:
    version, first_line = compiler_version(tool)
    if version_less(version, minimum):
        die(f"{nice_name} must be at least {format_version(minimum)}, got {format_version(version)} from {first_line}")
    info(f"{nice_name} {format_version(version)} ok ({tool})")
    return resolve_compiler_executable(tool), first_line


def check_llvm_compilers() -> tuple[tuple[str, str], tuple[str, str]]:
    return (
        check_compiler(LLVM_C_COMPILER, "GCC C compiler", LLVM_GCC_MIN),
        check_compiler(LLVM_CXX_COMPILER, "GCC C++ compiler", LLVM_GCC_MIN),
    )


def check_loom_compilers() -> tuple[tuple[str, str], tuple[str, str]]:
    return (
        check_compiler(LOOM_C_COMPILER, "Clang C compiler", LOOM_CLANG_MIN),
        check_compiler(LOOM_CXX_COMPILER, "Clang C++ compiler", LOOM_CLANG_MIN),
    )


def compiler_status(tool: str, minimum: tuple[int, ...]) -> str:
    try:
        version, first_line = compiler_version(tool)
    except SystemExit:
        return f"{tool} unavailable"
    verdict = "ok"
    if version_less(version, minimum):
        verdict = f"too old, need >= {format_version(minimum)}"
    return f"{tool} {format_version(version)} ({verdict}; {first_line})"


def check_git_version() -> None:
    try:
        output = subprocess.check_output(["git", "--version"], stderr=subprocess.DEVNULL).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        die("git not found on PATH")
    parts = output.split()[-1].split(".")
    try:
        version = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        warn(f"could not parse git version from {output!r}; assuming new enough")
        return
    if version < REQUIRED_GIT:
        die(f"git >= {REQUIRED_GIT[0]}.{REQUIRED_GIT[1]} required, got {output}")


def resolve_main_worktree(root: Path) -> Path:
    """Return the canonical primary worktree from Git's porcelain output."""
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "worktree", "list", "--porcelain"],
            stderr=subprocess.STDOUT,
        ).decode(errors="replace")
    except FileNotFoundError:
        die("git not found on PATH")
    except subprocess.CalledProcessError as error:
        detail = error.output.decode(errors="replace").strip()
        die(f"could not resolve primary worktree for {root}: {detail or error}")
    for line in output.splitlines():
        if line.startswith("worktree "):
            return real(Path(line.split(" ", 1)[1]))
    die(f"could not resolve primary worktree for {root}: empty worktree list")


def is_nfs(path: Path) -> bool:
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return False
    resolved = str(real(path))
    nfs_mounts = [fields[1] for line in mounts if len(fields := line.split()) >= 3 and fields[2].startswith("nfs")]
    for mount in sorted(nfs_mounts, key=len, reverse=True):
        prefix = mount.rstrip("/") + "/"
        if resolved == mount or resolved.startswith(prefix):
            return True
    return False
