#!/usr/bin/env python3
"""Worktree-aware build dispatcher used by the top-level Makefile.

Path resolution and the edge cases that pure Make/shell handles poorly
all live here:

  * `resolve_main_worktree` parses `git worktree list --porcelain`,
    which is supported on every supported git version and avoids the
    `git rev-parse --path-format=absolute` flag (git >= 2.31).
  * Paths are run through `realpath` so a symlinked entry into the main
    worktree still compares equal to the canonical worktree path.
  * The main worktree owns every initialized top-level submodule checkout.
    Linked worktrees consume those shared sources and must keep their own
    submodule paths and administrative state uninitialized.
  * The shared LLVM build is gated by an fcntl.flock with a configurable
    timeout, so a wedged build cannot hang every other worktree forever.
  * A deterministic stamp records the validated CIRCT/LLVM pins, exact
    LLVM compiler identities, and semantic CMake arguments. Identity
    drift discards the shared build before reconfiguration.
  * If a per-worktree loom build was configured against a shared LLVM
    that has since been wiped (e.g. main ran `distclean`), the stale
    loom build directory is removed before reconfiguring.
  * NFS-mounted shared trees are detected and surfaced as a warning;
    cross-host flock semantics on NFS are unreliable.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REQUIRED_GIT = (2, 7)
LLVM_C_COMPILER = "gcc"
LLVM_CXX_COMPILER = "g++"
LLVM_GCC_MIN = (7, 4)
LOOM_C_COMPILER = "clang"
LOOM_CXX_COMPILER = "clang++"
LOOM_CLANG_MIN = (21, 1, 8)
LLVM_SEMANTIC_CMAKE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLVM_ENABLE_PROJECTS=mlir;clang",
    "-DLLVM_TARGETS_TO_BUILD=host",
    "-DLLVM_ENABLE_ASSERTIONS=ON",
    "-DLLVM_ENABLE_RTTI=ON",
    "-DLLVM_INSTALL_UTILS=ON",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DLLVM_BUILD_LLVM_DYLIB=ON",
    "-DLLVM_LINK_LLVM_DYLIB=ON",
)


def info(msg: str) -> None:
    print(f"info: {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    info("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kwargs)


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
        out = subprocess.check_output(
            [tool, "--version"], stderr=subprocess.STDOUT
        ).decode(errors="replace")
    except FileNotFoundError:
        die(f"{tool} not found on PATH")
    except subprocess.CalledProcessError as e:
        detail = e.output.decode(errors="replace").strip()
        die(f"could not run {tool} --version: {detail or e}")
    version = parse_version(out)
    if version is None:
        die(f"could not parse {tool} version from {out.splitlines()[0]!r}")
    return version, out.splitlines()[0].strip()


def check_compiler(tool: str, nice_name: str,
                   minimum: tuple[int, ...]) -> tuple[str, str]:
    version, first_line = compiler_version(tool)
    if version_less(version, minimum):
        die(
            f"{nice_name} must be at least {format_version(minimum)}, "
            f"got {format_version(version)} from {first_line}"
        )
    info(f"{nice_name} {format_version(version)} ok ({tool})")
    return resolve_compiler_executable(tool), first_line


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
        return str(resolved)

    die(f"could not resolve compiler {tool} on PATH without a ccache wrapper")


def check_llvm_compilers() -> tuple[tuple[str, str], tuple[str, str]]:
    return (
        check_compiler(LLVM_C_COMPILER, "GCC C compiler", LLVM_GCC_MIN),
        check_compiler(LLVM_CXX_COMPILER, "GCC C++ compiler", LLVM_GCC_MIN),
    )


def check_loom_compilers() -> None:
    check_compiler(LOOM_C_COMPILER, "Clang C compiler", LOOM_CLANG_MIN)
    check_compiler(LOOM_CXX_COMPILER, "Clang C++ compiler", LOOM_CLANG_MIN)


def compiler_status(tool: str, minimum: tuple[int, ...]) -> str:
    try:
        version, first_line = compiler_version(tool)
    except SystemExit:
        return f"{tool} unavailable"
    verdict = "ok"
    if version_less(version, minimum):
        verdict = f"too old, need >= {format_version(minimum)}"
    return f"{tool} {format_version(version)} ({verdict}; {first_line})"


def real(p: Path) -> Path:
    return Path(os.path.realpath(str(p)))


def check_git_version() -> None:
    try:
        out = subprocess.check_output(
            ["git", "--version"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        die("git not found on PATH")
    parts = out.split()[-1].split(".")
    try:
        version = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        warn(f"could not parse git version from {out!r}; assuming new enough")
        return
    if version < REQUIRED_GIT:
        die(f"git >= {REQUIRED_GIT[0]}.{REQUIRED_GIT[1]} required, got {out}")


def resolve_main_worktree(root: Path) -> Path:
    """Return the absolute, realpath-canonicalised main worktree.

    The first `worktree <path>` entry emitted by `git worktree list
    --porcelain` is always the main worktree (the one that owns
    `.git/`), regardless of which worktree we run from.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "worktree", "list", "--porcelain"],
            stderr=subprocess.STDOUT,
        ).decode(errors="replace")
    except FileNotFoundError:
        die("git not found on PATH")
    except subprocess.CalledProcessError as e:
        detail = e.output.decode(errors="replace").strip()
        die(
            f"could not resolve primary worktree for {root}: "
            f"{detail or e}"
        )
    for line in out.splitlines():
        if line.startswith("worktree "):
            return real(Path(line.split(" ", 1)[1]))
    die(f"could not resolve primary worktree for {root}: empty worktree list")


def is_nfs(path: Path) -> bool:
    try:
        with open("/proc/mounts") as f:
            mounts = f.read().splitlines()
    except OSError:
        return False
    rp = str(real(path))
    nfs_mnts = []
    for line in mounts:
        fields = line.split()
        if len(fields) >= 3 and fields[2].startswith("nfs"):
            nfs_mnts.append(fields[1])
    nfs_mnts.sort(key=len, reverse=True)
    for mnt in nfs_mnts:
        prefix = mnt.rstrip("/") + "/"
        if rp == mnt or rp.startswith(prefix):
            return True
    return False


class Paths:
    def __init__(self, root: Path):
        self.root = real(root)
        self.main = resolve_main_worktree(self.root)
        self.externals_root = self.main / "externals"
        self.circt_root = self.externals_root / "circt"
        llvm_external = self.externals_root / "llvm"
        self.llvm_root = llvm_external
        self.llvm_src = llvm_external / "llvm"
        self.llvm_build = llvm_external / "build"
        self.llvm_lock = self.externals_root / ".loom-build.llvm.lock"
        self.llvm_stamp = self.externals_root / ".loom-build.llvm.stamp"
        self.loom_build = self.root / "build"
        self.mlir_dir = self.llvm_build / "lib" / "cmake" / "mlir"
        self.cmake_llvm_dir = self.llvm_build / "lib" / "cmake" / "llvm"
        self.cmake_clang_dir = self.llvm_build / "lib" / "cmake" / "clang"
        self.llvm_lit = self.llvm_build / "bin" / "llvm-lit"

    @property
    def is_main(self) -> bool:
        return self.main == self.root


@dataclass(frozen=True)
class DependencyState:
    circt_commit: str
    llvm_commit: str


def gitlinks_at_head(root: Path) -> tuple[str, ...]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-rz", "--full-tree", "HEAD"],
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        die("git not found on PATH")
    except subprocess.CalledProcessError as e:
        detail = e.output.decode(errors="replace").strip()
        die(f"could not enumerate submodules in {root}: {detail or e}")

    paths = []
    for raw_entry in output.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, object_type, _commit = metadata.split(b" ", 2)
        if mode == b"160000" and object_type == b"commit":
            paths.append(raw_path.decode(errors="surrogateescape"))
    return tuple(sorted(paths))


def initialized_checkout(path: Path) -> bool:
    if not path.exists():
        return False
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return False
    return real(Path(completed.stdout.strip())) == real(path)


def linked_modules_dir(root: Path) -> Path:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--git-path", "modules"],
            stderr=subprocess.STDOUT,
        ).decode(errors="replace").strip()
    except FileNotFoundError:
        die("git not found on PATH")
    except subprocess.CalledProcessError as e:
        detail = e.output.decode(errors="replace").strip()
        die(f"could not resolve worktree submodule state for {root}: {detail or e}")
    path = Path(output)
    return real(path if path.is_absolute() else root / path)


def check_linked_submodule_hygiene(paths: Paths) -> None:
    if paths.is_main:
        return

    initialized = [
        relative_path
        for relative_path in gitlinks_at_head(paths.root)
        if initialized_checkout(paths.root / relative_path)
    ]
    modules_dir = linked_modules_dir(paths.root)
    has_admin_state = modules_dir.is_dir() and any(modules_dir.iterdir())
    if not initialized and not has_admin_state:
        return

    details = []
    if initialized:
        details.append("initialized checkouts: " + ", ".join(initialized))
    if has_admin_state:
        details.append(f"administrative state: {modules_dir}")
    die(
        f"linked worktree {paths.root} must not initialize submodules "
        f"({'; '.join(details)}). Use the primary worktree sources under "
        f"{paths.externals_root}. No automatic repair is safe because Git "
        f"shares submodule configuration between worktrees. Preserve any "
        f"superproject changes, then remove and recreate only the linked "
        f"worktree without initializing submodules. Do not modify the shared "
        f"primary checkouts"
    )


def check_dependency_pins(paths: Paths) -> DependencyState:
    """Validate and return the invoking worktree's shared dependency state."""

    check_linked_submodule_hygiene(paths)

    def git_output(repo: Path, *args: str) -> str:
        cmd = ["git", "-C", str(repo), *args]
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.STDOUT
            ).decode(errors="replace").rstrip()
        except FileNotFoundError:
            die("git not found on PATH")
        except subprocess.CalledProcessError as e:
            detail = e.output.decode(errors="replace").strip()
            die(
                f"could not inspect CIRCT/LLVM dependency state with "
                f"{shlex.join(cmd)}: {detail or e}"
            )

    dependency_paths = ("externals/circt", "externals/llvm")
    unmerged = git_output(
        paths.root, "ls-files", "-u", "--", *dependency_paths
    )
    if unmerged:
        inspect_command = shlex.join([
            "git", "-C", str(paths.root),
            "ls-files", "-u", "--", *dependency_paths,
        ])
        stage_command = shlex.join([
            "git", "-C", str(paths.root), "add", "--", *dependency_paths,
        ])
        rerun_command = shlex.join([
            "make", "-C", str(paths.root), "doctor",
        ])
        die(
            f"invoking worktree {paths.root} has unmerged gitlink entries; "
            f"resolve them manually:\n"
            f"  1. inspect index stages: {inspect_command}\n"
            f"  2. select the intended CIRCT and LLVM gitlinks without "
            f"automatically choosing ours or theirs\n"
            f"  3. stage the resolved gitlinks: {stage_command}\n"
            f"  4. rerun the dependency gate: {rerun_command}"
        )

    circt_commit = git_output(
        paths.root, "rev-parse", "HEAD:externals/circt"
    ).strip()
    llvm_commit = git_output(
        paths.root, "rev-parse", "HEAD:externals/llvm"
    ).strip()
    repair_commands = (
        shlex.join([
            "git", "-C", str(paths.main), "submodule", "update",
            "--init", "--checkout", "--",
            "externals/circt", "externals/llvm",
        ]),
        shlex.join(["git", "-C", str(paths.circt_root), "fetch", "origin"]),
        shlex.join(["git", "-C", str(paths.llvm_root), "fetch", "origin"]),
        shlex.join([
            "git", "-C", str(paths.circt_root),
            "checkout", "--detach", circt_commit,
        ]),
        shlex.join([
            "git", "-C", str(paths.llvm_root),
            "checkout", "--detach", llvm_commit,
        ]),
    )
    repair = "\n  ".join(repair_commands)

    def checkout_head(path: Path, relative_path: str) -> str:
        if not path.exists():
            die(
                f"shared {relative_path} checkout is uninitialized for "
                f"invoking superproject {paths.root}; repair with:\n  {repair}"
            )
        top = real(Path(git_output(path, "rev-parse", "--show-toplevel")))
        if top != real(path):
            die(
                f"shared {relative_path} is not an initialized repository "
                f"under {paths.main}; repair with:\n  {repair}"
            )
        return git_output(path, "rev-parse", "HEAD").strip()

    circt_head = checkout_head(paths.circt_root, "externals/circt")
    llvm_head = checkout_head(paths.llvm_root, "externals/llvm")
    if circt_head != circt_commit or llvm_head != llvm_commit:
        die(
            f"shared dependency checkout drift: invoking superproject "
            f"{paths.root} pins CIRCT {circt_commit} and LLVM {llvm_commit}, "
            f"but shared checkouts under {paths.main} are CIRCT "
            f"{circt_head} and LLVM {llvm_head}; repair with:\n  {repair}"
        )

    circt_llvm_commit = git_output(
        paths.circt_root, "rev-parse", f"{circt_commit}:llvm"
    ).strip()
    if circt_llvm_commit != llvm_commit:
        die(
            f"invoking superproject parent gitlinks are internally "
            f"inconsistent: CIRCT {circt_commit} pins LLVM "
            f"{circt_llvm_commit}, but externals/llvm is pinned to "
            f"{llvm_commit}; the CIRCT and LLVM parent gitlinks in "
            f"{paths.root} must be updated atomically"
        )

    nested_llvm = paths.circt_root / "llvm"
    nested_top = ""
    if nested_llvm.exists():
        nested = subprocess.run(
            ["git", "-C", str(nested_llvm), "rev-parse", "--show-toplevel"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if nested.returncode == 0:
            nested_top = nested.stdout.strip()
    if nested_top and real(Path(nested_top)) == real(nested_llvm):
        deinit_command = shlex.join([
            "git", "-C", str(paths.circt_root), "submodule", "deinit",
            "-f", "--", "llvm",
        ])
        die(
            "externals/circt/llvm must remain uninitialized; repair with: "
            f"{deinit_command}"
        )

    for repo, label in (
        (paths.circt_root, "externals/circt"),
        (paths.llvm_root, "externals/llvm"),
    ):
        dirty = git_output(
            repo, "status", "--porcelain", "--untracked-files=no"
        )
        if dirty:
            die(
                f"shared {label} has tracked modifications:\n{dirty}\n"
                f"restore the tracked changes, then rerun the gate. If the "
                f"changes are intentional, make them an upstream commit and "
                f"update the parent CIRCT and LLVM gitlinks atomically before "
                f"rerunning"
            )

    return DependencyState(
        circt_commit=circt_commit,
        llvm_commit=llvm_commit,
    )


class FileLock:
    """Exclusive fcntl.flock with a polled timeout.

    Polling rather than blocking lets us bail out cleanly when another
    worktree's build is wedged, and emit a one-shot waiting message
    when contention occurs.
    """

    def __init__(self, path: Path, timeout: float):
        self.path = path
        self.timeout = timeout
        self.fd: int | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.timeout
        announced = False
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    raise
                if not announced:
                    info(f"waiting for shared LLVM lock at {self.path}")
                    announced = True
                if time.monotonic() >= deadline:
                    os.close(self.fd)
                    self.fd = None
                    die(
                        f"timed out after {self.timeout:.0f}s waiting for "
                        f"{self.path}; another build may be stuck"
                    )
                time.sleep(1.0)

    def __exit__(self, *exc) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None


def llvm_build_identity(
    state: DependencyState,
    compilers: tuple[tuple[str, str], tuple[str, str]],
) -> str:
    return json.dumps(
        {
            "compilers": {
                "c": {"path": compilers[0][0], "version": compilers[0][1]},
                "cxx": {"path": compilers[1][0], "version": compilers[1][1]},
            },
            "dependencies": {
                "circt": state.circt_commit,
                "llvm": state.llvm_commit,
            },
            "semantic_cmake_args": list(LLVM_SEMANTIC_CMAKE_ARGS),
        },
        sort_keys=True,
    )


def read_stamp(path: Path) -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return ""


def write_stamp(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")


def read_cmake_cache_entry(build_dir: Path, key: str) -> str:
    cache = build_dir / "CMakeCache.txt"
    try:
        lines = cache.read_text().splitlines()
    except FileNotFoundError:
        return ""
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return ""


def compiler_basename(path: str) -> str:
    return Path(path).name if path else ""


def cmake_cache_uses_compilers(build_dir: Path, c_compiler: str,
                               cxx_compiler: str) -> bool:
    cached_c = read_cmake_cache_entry(build_dir, "CMAKE_C_COMPILER")
    cached_cxx = read_cmake_cache_entry(build_dir, "CMAKE_CXX_COMPILER")
    return (
        compiler_basename(cached_c) == compiler_basename(c_compiler) and
        compiler_basename(cached_cxx) == compiler_basename(cxx_compiler)
    )


def configure_llvm(
    paths: Paths,
    compilers: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    run([
        "cmake", "-G", "Ninja",
        "-S", str(paths.llvm_src),
        "-B", str(paths.llvm_build),
        f"-DCMAKE_C_COMPILER={compilers[0][0]}",
        f"-DCMAKE_CXX_COMPILER={compilers[1][0]}",
        *LLVM_SEMANTIC_CMAKE_ARGS,
        "-DLLVM_CCACHE_BUILD=ON",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    ])


def configure_loom(paths: Paths) -> None:
    run([
        "cmake", "-G", "Ninja",
        "-S", str(paths.root),
        "-B", str(paths.loom_build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={LOOM_C_COMPILER}",
        f"-DCMAKE_CXX_COMPILER={LOOM_CXX_COMPILER}",
        f"-DMLIR_DIR={paths.mlir_dir}",
        f"-DLLVM_DIR={paths.cmake_llvm_dir}",
        f"-DClang_DIR={paths.cmake_clang_dir}",
        f"-DLLVM_EXTERNAL_LIT={paths.llvm_lit}",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    ])


def sync_shared_llvm(
    paths: Paths,
    args: argparse.Namespace,
    always_build: bool,
) -> None:
    if not paths.is_main:
        info(f"checking shared LLVM under main worktree {paths.main}")
    if is_nfs(paths.llvm_root):
        warn(
            f"shared LLVM tree {paths.llvm_root} appears to live on NFS; "
            "flock semantics across hosts are unreliable"
        )
    with FileLock(paths.llvm_lock, args.lock_timeout):
        state = check_dependency_pins(paths)
        compilers = check_llvm_compilers()
        current = llvm_build_identity(state, compilers)
        prev = read_stamp(paths.llvm_stamp)
        build_ninja = paths.llvm_build / "build.ninja"
        build_ready = (
            build_ninja.exists() and
            (paths.mlir_dir / "MLIRConfig.cmake").exists() and
            (paths.cmake_clang_dir / "ClangConfig.cmake").exists()
        )
        rebuild = prev != current or not build_ready
        if rebuild:
            if paths.llvm_build.exists():
                info(
                    f"LLVM build is incomplete or its identity changed; removing "
                    f"{paths.llvm_build}"
                )
                shutil.rmtree(paths.llvm_build)
            configure_llvm(paths, compilers)
        if always_build or rebuild:
            run(["cmake", "--build", str(paths.llvm_build), f"-j{args.jobs}"])
            write_stamp(paths.llvm_stamp, current)


def build_llvm(paths: Paths, args: argparse.Namespace) -> None:
    sync_shared_llvm(paths, args, always_build=True)


def loom_build_is_stale(paths: Paths) -> bool:
    """Loom was configured against a shared LLVM that no longer exists.

    Detected by checking whether the cmake packages the loom build was
    pointed at are still on disk. If not, the cached build.ninja will
    fail at link time, so we wipe and reconfigure proactively. Both
    MLIR and Clang configs must be present because configure_loom()
    passes -DMLIR_DIR and -DClang_DIR; an older shared LLVM built
    without LLVM_ENABLE_PROJECTS=...;clang would otherwise silently
    pass the missing Clang config straight to cmake.
    """
    bn = paths.loom_build / "build.ninja"
    if not bn.exists():
        return False
    if not (paths.mlir_dir / "MLIRConfig.cmake").exists():
        return True
    return not (paths.cmake_clang_dir / "ClangConfig.cmake").exists()


def ensure_shared_llvm(paths: Paths, args: argparse.Namespace) -> None:
    sync_shared_llvm(paths, args, always_build=False)


def build_loom(paths: Paths, args: argparse.Namespace) -> None:
    if loom_build_is_stale(paths):
        info(
            f"loom build at {paths.loom_build} references a missing shared "
            f"LLVM ({paths.mlir_dir}); wiping and reconfiguring"
        )
        shutil.rmtree(paths.loom_build, ignore_errors=True)
    bn = paths.loom_build / "build.ninja"
    if bn.exists() and not cmake_cache_uses_compilers(
        paths.loom_build, LOOM_C_COMPILER, LOOM_CXX_COMPILER
    ):
        info(
            f"loom build compiler changed to {LOOM_C_COMPILER}/"
            f"{LOOM_CXX_COMPILER}; removing {paths.loom_build}"
        )
        shutil.rmtree(paths.loom_build, ignore_errors=True)
    ensure_shared_llvm(paths, args)
    if not bn.exists():
        configure_loom(paths)
    run(["cmake", "--build", str(paths.loom_build), f"-j{args.jobs}"])


def cmd_doctor(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
    state = check_dependency_pins(paths)
    for tool in ("cmake", "ninja"):
        if not shutil.which(tool):
            warn(f"{tool} not found on PATH")
    if is_nfs(paths.llvm_root):
        warn(
            f"shared LLVM tree {paths.llvm_root} appears to live on NFS; "
            "flock across hosts may be unreliable"
        )
    print(f"main_worktree   {paths.main}")
    print(f"this_worktree   {paths.root}")
    print(f"is_main         {paths.is_main}")
    print(f"externals_root  {paths.externals_root}")
    print(
        "submodule_mode  "
        + ("primary owner" if paths.is_main else "shared from primary")
    )
    print(f"circt_commit    {state.circt_commit}")
    print(f"llvm_commit     {state.llvm_commit}")
    print(f"circt_llvm_pin  {state.llvm_commit}")
    print("nested_llvm     uninitialized")
    print(f"llvm_src        {paths.llvm_src}")
    print(f"llvm_build      {paths.llvm_build}")
    print(f"llvm_lock       {paths.llvm_lock}")
    print(f"llvm_stamp      {paths.llvm_stamp}")
    print(f"stamp_value     {read_stamp(paths.llvm_stamp) or '(unset)'}")
    print(f"llvm_c          {compiler_status(LLVM_C_COMPILER, LLVM_GCC_MIN)}")
    print(f"llvm_cxx        {compiler_status(LLVM_CXX_COMPILER, LLVM_GCC_MIN)}")
    print(f"loom_c          {compiler_status(LOOM_C_COMPILER, LOOM_CLANG_MIN)}")
    print(f"loom_cxx        {compiler_status(LOOM_CXX_COMPILER, LOOM_CLANG_MIN)}")
    print(f"loom_build      {paths.loom_build}")
    print(f"loom_stale      {loom_build_is_stale(paths)}")


def cmd_externals_root(paths: Paths, args: argparse.Namespace) -> None:
    check_linked_submodule_hygiene(paths)
    print(paths.externals_root)


def cmd_build_llvm(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
    build_llvm(paths, args)


def cmd_build_loom(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
    check_loom_compilers()
    build_loom(paths, args)


def cmd_clean(paths: Paths, args: argparse.Namespace) -> None:
    if paths.loom_build.exists():
        info(f"removing {paths.loom_build}")
        shutil.rmtree(paths.loom_build, ignore_errors=True)


def cmd_distclean(paths: Paths, args: argparse.Namespace) -> None:
    if paths.loom_build.exists():
        info(f"removing {paths.loom_build}")
        shutil.rmtree(paths.loom_build, ignore_errors=True)
    if paths.is_main:
        with FileLock(paths.llvm_lock, args.lock_timeout):
            if paths.llvm_build.exists():
                info(f"removing shared {paths.llvm_build}")
                shutil.rmtree(paths.llvm_build, ignore_errors=True)
            if paths.llvm_stamp.exists():
                paths.llvm_stamp.unlink()
    else:
        info(
            f"distclean from linked worktree only removes {paths.loom_build}; "
            f"shared LLVM at {paths.llvm_build} preserved"
        )


def cmd_test(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
    check_loom_compilers()
    build_loom(paths, args)
    child_env = os.environ.copy()
    extra_args = shlex.split(child_env.pop("LIT_OPTS", ""))
    child_env.setdefault("LOOM_TEST_JOBS", str(args.jobs))
    run(
        [
            str(paths.llvm_lit),
            "-sv",
            "--time-tests",
            f"-j{args.jobs}",
            *extra_args,
            str(paths.loom_build / "test"),
        ],
        env=child_env,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=os.getcwd(),
                   help="worktree root (defaults to CWD)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--lock-timeout", type=float, default=1800.0,
                   help="seconds to wait for the shared LLVM lock")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "externals-root", "build-llvm", "build-loom",
                 "clean", "distclean", "test"):
        sub.add_parser(name)
    args = p.parse_args()

    paths = Paths(Path(args.root))
    dispatch = {
        "doctor": cmd_doctor,
        "externals-root": cmd_externals_root,
        "build-llvm": cmd_build_llvm,
        "build-loom": cmd_build_loom,
        "clean": cmd_clean,
        "distclean": cmd_distclean,
        "test": cmd_test,
    }
    try:
        dispatch[args.command](paths, args)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
