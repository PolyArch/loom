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
  * Shared LLVM and CIRCT products are gated by one SharedProductLock
    transaction. An internal ticket turnstile admits readers and writers in
    arrival order and holds one record per participant still acquiring;
    builds and distclean are exclusive writers, while ordinary Loom builds
    hold a shared product lock during consumption.
  * The shared CIRCT build reuses that same LLVM lock and is held across
    the LLVM ensure phase and the CIRCT configure/build, so the LLVM a
    CIRCT build links against cannot be rebuilt or wiped concurrently.
    `make loom` never builds CIRCT; it only offers an already-built,
    stamped CIRCT whose identity matches the ensured LLVM build.
  * Separate deterministic stamps record the validated LLVM and CIRCT
    identities. LLVM owns its source pin, compiler identities, and semantic
    CMake arguments; CIRCT layers its source pin and semantic arguments on
    the exact LLVM identity.
  * If a per-worktree loom build was configured against a shared LLVM
    that has since been wiped (e.g. main ran `distclean`), the stale
    loom build directory is removed before reconfiguring.
  * NFS-mounted shared trees are detected and surfaced as a warning;
    cross-host flock semantics on NFS are unreliable.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REQUIRED_GIT = (2, 7)
LLVM_C_COMPILER = "gcc"
LLVM_CXX_COMPILER = "g++"
LLVM_GCC_MIN = (7, 4)
LOOM_C_COMPILER = "clang"
LOOM_CXX_COMPILER = "clang++"
LOOM_CLANG_MIN = (21, 1, 8)
MAX_LOCK_TIMEOUT = 3600.0
CHILD_TERMINATION_GRACE = 1.0
_LIBC = (
    ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux")
    else None
)
_PR_SET_PDEATHSIG = 1
_SUPERVISE_COMMAND = "--internal-supervise-command"
_TURNSTILE_RECORD_SIZE = 17
_TURNSTILE_MUTEX_OFFSET = 0
LLVM_SEMANTIC_CMAKE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLVM_ENABLE_PROJECTS=mlir;clang;lld",
    "-DLLVM_TARGETS_TO_BUILD=host;RISCV;ARM;AArch64",
    "-DLLVM_ENABLE_ASSERTIONS=ON",
    "-DLLVM_ENABLE_RTTI=ON",
    "-DLLVM_INSTALL_UTILS=ON",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DLLVM_BUILD_LLVM_DYLIB=ON",
    "-DLLVM_LINK_LLVM_DYLIB=ON",
)
# CIRCT is configured against the shared LLVM build (externals/llvm), never
# against the nested externals/circt/llvm submodule, which must stay
# uninitialized. These are the semantic args that change CIRCT's ABI or the
# exported package, so they participate in the CIRCT build identity.
CIRCT_SEMANTIC_CMAKE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON",
    "-DLLVM_ENABLE_ASSERTIONS=ON",
    "-DLLVM_ENABLE_RTTI=ON",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DCIRCT_INCLUDE_TESTS=OFF",
    "-DCIRCT_INCLUDE_DOCS=OFF",
)


def info(msg: str) -> None:
    print(f"info: {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def validate_lock_timeout(value: str | float) -> float:
    timeout = float(value)
    if (
        not math.isfinite(timeout)
        or timeout < 0
        or timeout > MAX_LOCK_TIMEOUT
    ):
        raise ValueError(
            f"lock timeout must be between 0 and {MAX_LOCK_TIMEOUT:g} seconds"
        )
    return timeout


class _CommandInterrupted(Exception):
    def __init__(self, signum: int):
        self.signum = signum


def _set_parent_death_signal(death_signal: int) -> None:
    if _LIBC is None:
        return
    if _LIBC.prctl(_PR_SET_PDEATHSIG, death_signal, 0, 0, 0) != 0:
        os._exit(127)


def _supervise_command(argv: list[str]) -> None:
    """Run one command and kill its process group if the dispatcher dies."""
    if len(argv) < 3:
        os._exit(127)
    try:
        expected_parent = int(argv[0])
        lease_fds = tuple(
            int(fd) for fd in argv[1].split(",") if fd
        )
        if any(fd < 0 for fd in lease_fds):
            raise ValueError
        for fd in lease_fds:
            os.fstat(fd)
    except (OSError, ValueError):
        os._exit(127)

    def parent_died(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    for watched in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(watched, parent_died)
    _set_parent_death_signal(signal.SIGTERM)
    if os.getppid() != expected_parent:
        parent_died(signal.SIGTERM, None)

    process = subprocess.Popen(
        argv[2:],
        close_fds=True,
        pass_fds=lease_fds,
    )
    returncode = process.wait()
    if returncode < 0:
        returncode = 128 - returncode
    os._exit(returncode)


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=CHILD_TERMINATION_GRACE)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    info("$ " + " ".join(str(c) for c in cmd))
    lease_fds = tuple(sorted(_held_product_fds))
    kwargs["close_fds"] = True
    kwargs["pass_fds"] = lease_fds
    command = [str(part) for part in cmd]
    # The supervisor and command share each product lease. If this dispatcher
    # dies, no command can outlive the lease it owns.
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            _SUPERVISE_COMMAND,
            str(os.getpid()),
            ",".join(str(fd) for fd in lease_fds),
            *command,
        ],
        start_new_session=True,
        **kwargs,
    )
    watched_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {}

    def interrupt(signum, frame):
        raise _CommandInterrupted(signum)

    for watched in watched_signals:
        previous_handlers[watched] = signal.signal(watched, interrupt)
    try:
        returncode = process.wait()
    except _CommandInterrupted as interruption:
        for watched in watched_signals:
            signal.signal(watched, signal.SIG_IGN)
        _terminate_process_group(process)
        if interruption.signum == signal.SIGINT:
            raise KeyboardInterrupt from None
        raise SystemExit(128 + interruption.signum) from None
    except BaseException:
        _terminate_process_group(process)
        raise
    finally:
        for watched, previous in previous_handlers.items():
            signal.signal(watched, previous)

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return subprocess.CompletedProcess(cmd, returncode)


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
        self.circt_build = self.circt_root / "build"
        self.circt_stamp = self.externals_root / ".loom-build.circt.stamp"
        self.circt_cmake_dir = self.circt_build / "lib" / "cmake" / "circt"
        llvm_external = self.externals_root / "llvm"
        self.llvm_root = llvm_external
        self.llvm_src = llvm_external / "llvm"
        self.llvm_build = llvm_external / "build"
        self.llvm_lock = self.externals_root / ".loom-build.llvm.lock"
        self.llvm_lock_turnstile = (
            self.externals_root / ".loom-build.llvm.turnstile.lock"
        )
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


# Product-lock file descriptors currently held in this dispatcher process.
# run() passes them through the supervisor to the command. All holders share
# each open file description, so closing or killing the dispatcher cannot
# release a lease while a mutating command still owns it.
_held_product_fds: set[int] = set()


class SharedProductLock:
    """Fair shared/exclusive lock over a product file and a turnstile file.

    The turnstile file is a header record carrying the next ticket followed
    by the tickets still acquiring, in ticket order. Ticket order is arrival
    order; a record's position carries no meaning and is rewritten freely.

    Turnstile lock offsets are a namespace of their own, unrelated to the
    record bytes stored at the same offsets: offset 0 serializes residency
    changes, and offset T is held by the owner of ticket T for as long as
    that owner is still acquiring. A participant blocks on the highest live
    ticket below its own until none is left, then takes the product lock and
    drops its residency. Only one participant contends for the product lock
    at a time, so a queued writer blocks every later reader while readers
    that already hold the product lock still coexist.

    Losing a participant releases its ticket offset, whether it crashed,
    timed out, or simply took the product lock and left, so every departure
    is reclaimable the same way. Entering the residency mutex rewrites the
    table as the live tickets alone, so residency stays proportional to the
    participants still acquiring, never to the peak concurrency reached or
    the acquisitions performed. A participant reaches the product lock only
    through a residency pass in which no earlier ticket is left, so the last
    one out leaves the header and its own record behind.
    """

    def __init__(
        self,
        product_path: Path,
        turnstile_path: Path,
        timeout: float,
        shared: bool,
    ):
        self.product_path = product_path
        self.turnstile_path = turnstile_path
        self.timeout = validate_lock_timeout(timeout)
        self.product_operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        self.product_fd: int | None = None
        self.turnstile_fd: int | None = None
        self.ticket: int | None = None

    def _timeout(self, label: str, path: Path) -> None:
        die(
            f"timed out after {self.timeout:g}s waiting for "
            f"shared product {label} at {path}; another build may be stuck"
        )

    def _acquire(
        self,
        fd: int,
        operation: int,
        deadline: float,
        label: str,
        path: Path,
        offset: int | None = None,
    ) -> None:
        def lock(nonblocking: bool) -> None:
            flags = operation | (fcntl.LOCK_NB if nonblocking else 0)
            if offset is None:
                fcntl.flock(fd, flags)
            else:
                fcntl.lockf(fd, flags, 1, offset, os.SEEK_SET)

        if self.timeout == 0:
            try:
                lock(True)
                return
            except BlockingIOError:
                self._timeout(label, path)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._timeout(label, path)

        class LockWaitExpired(Exception):
            pass

        def expire_wait(signum, frame):
            raise LockWaitExpired

        previous_handler = signal.signal(signal.SIGALRM, expire_wait)
        started = time.monotonic()
        previous_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            lock(False)
        except LockWaitExpired:
            self._timeout(label, path)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                elapsed = time.monotonic() - started
                restored = max(1e-6, previous_timer[0] - elapsed)
                signal.setitimer(
                    signal.ITIMER_REAL, restored, previous_timer[1]
                )

    def _release_turnstile(self) -> None:
        if self.turnstile_fd is None:
            return
        try:
            # Closing the descriptor drops this process's residency: a close
            # releases every record lock the process holds on the file.
            os.close(self.turnstile_fd)
        finally:
            self.turnstile_fd = None
            self.ticket = None

    def _release_product(self) -> None:
        if self.product_fd is None:
            return
        try:
            os.close(self.product_fd)
        finally:
            self.product_fd = None

    def _turnstile_lock(self, operation: int, offset: int) -> None:
        fcntl.lockf(self.turnstile_fd, operation, 1, offset, os.SEEK_SET)

    def _read_records(self) -> list[int]:
        size = os.fstat(self.turnstile_fd).st_size
        if size % _TURNSTILE_RECORD_SIZE:
            die(
                f"invalid shared product turnstile state at "
                f"{self.turnstile_path}"
            )
        contents = os.pread(self.turnstile_fd, size, 0)
        if len(contents) != size:
            die(
                f"could not read shared product turnstile state at "
                f"{self.turnstile_path}"
            )
        records = []
        for offset in range(0, size, _TURNSTILE_RECORD_SIZE):
            record = contents[offset:offset + _TURNSTILE_RECORD_SIZE]
            try:
                ticket = int(record[:-1], 16) if record.endswith(b"\n") else 0
            except ValueError:
                ticket = 0
            # Tickets start at 1 because offset 0 is the residency mutex.
            if ticket < 1:
                die(
                    f"invalid shared product turnstile record at "
                    f"{self.turnstile_path}"
                )
            records.append(ticket)
        return records

    def _write_record(self, index: int, ticket: int) -> None:
        record = f"{ticket:016x}\n".encode()
        offset = index * _TURNSTILE_RECORD_SIZE
        if os.pwrite(self.turnstile_fd, record, offset) != len(record):
            die(
                f"could not update shared product turnstile at "
                f"{self.turnstile_path}"
            )

    def _live_tickets(self, records: list[int]) -> list[int]:
        """Tickets whose owner is still acquiring. Holds the mutex."""
        live = []
        for ticket in records[1:]:
            # A process never conflicts with its own record locks, so this
            # participant's own ticket is live without being probed.
            if ticket == self.ticket:
                live.append(ticket)
                continue
            try:
                self._turnstile_lock(fcntl.LOCK_EX | fcntl.LOCK_NB, ticket)
            except BlockingIOError:
                live.append(ticket)
                continue
            self._turnstile_lock(fcntl.LOCK_UN, ticket)
        return live

    def _compact(self, watermark: int, tickets: list[int]) -> None:
        """Rewrite the table as the watermark followed by `tickets`.

        A record's position carries no meaning, so the table is written in
        ticket order and truncated to exactly the tickets given.
        """
        self._write_record(0, watermark)
        for slot, entry in enumerate(tickets, start=1):
            self._write_record(slot, entry)
        os.ftruncate(
            self.turnstile_fd,
            (len(tickets) + 1) * _TURNSTILE_RECORD_SIZE,
        )

    @contextmanager
    def _residency(self, deadline: float):
        """Yield the ticket watermark and live tickets under the mutex.

        Entering reclaims every record whose owner has stopped acquiring,
        so what a participant hands to the next one is the participants
        still there rather than the peak the turnstile once carried.
        """
        self._acquire(
            self.turnstile_fd,
            fcntl.LOCK_EX,
            deadline,
            "turnstile",
            self.turnstile_path,
            _TURNSTILE_MUTEX_OFFSET,
        )
        try:
            records = self._read_records()
            watermark = records[0] if records else 1
            live = self._live_tickets(records)
            if live != records[1:]:
                self._compact(watermark, live)
            yield watermark, live
        finally:
            self._turnstile_lock(fcntl.LOCK_UN, _TURNSTILE_MUTEX_OFFSET)

    def _register(self, deadline: float) -> None:
        """Take the next ticket and add it to the live tickets."""
        with self._residency(deadline) as (watermark, live):
            # Above every live ticket, and never below the recorded
            # watermark, so arrival order survives a quiet turnstile.
            ticket = max(watermark, max(live, default=0) + 1)
            self._turnstile_lock(fcntl.LOCK_EX | fcntl.LOCK_NB, ticket)
            self.ticket = ticket
            self._compact(ticket + 1, sorted(live) + [ticket])

    def _await_predecessors(self, deadline: float) -> None:
        """Block until no participant holds an earlier ticket.

        Tickets are never reused, so each wait retires one candidate for
        good and the loop runs at most once per participant that was
        already acquiring when this one registered. The pass that finds no
        predecessor left is also the one that reclaims their records.
        """
        while True:
            with self._residency(deadline) as (_, live):
                predecessor = max(
                    (ticket for ticket in live if ticket < self.ticket),
                    default=None,
                )
            if predecessor is None:
                return
            self._acquire(
                self.turnstile_fd,
                fcntl.LOCK_EX,
                deadline,
                "turnstile",
                self.turnstile_path,
                predecessor,
            )
            self._turnstile_lock(fcntl.LOCK_UN, predecessor)

    def __enter__(self) -> "SharedProductLock":
        deadline = time.monotonic() + self.timeout
        self.product_path.parent.mkdir(parents=True, exist_ok=True)
        self.turnstile_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.turnstile_fd = os.open(
                str(self.turnstile_path),
                os.O_RDWR | os.O_CREAT,
                0o644,
            )
            self.product_fd = os.open(
                str(self.product_path), os.O_RDWR | os.O_CREAT, 0o644
            )
            self._register(deadline)
            self._await_predecessors(deadline)
            self._acquire(
                self.product_fd,
                self.product_operation,
                deadline,
                "lock",
                self.product_path,
            )
            self._release_turnstile()
            _held_product_fds.add(self.product_fd)
            return self
        except BaseException:
            self._release_product()
            self._release_turnstile()
            raise

    def __exit__(self, *exc) -> None:
        _held_product_fds.discard(self.product_fd)
        self._release_product()


def llvm_build_identity(
    llvm_commit: str,
    compilers: tuple[tuple[str, str], tuple[str, str]],
) -> str:
    return json.dumps(
        {
            "compilers": {
                "c": {"path": compilers[0][0], "version": compilers[0][1]},
                "cxx": {"path": compilers[1][0], "version": compilers[1][1]},
            },
            "dependencies": {"llvm": llvm_commit},
            "semantic_cmake_args": list(LLVM_SEMANTIC_CMAKE_ARGS),
        },
        sort_keys=True,
    )


def circt_build_identity(llvm_identity: str, circt_commit: str) -> str:
    """Identity of the shared CIRCT build.

    Embeds the full LLVM build identity verbatim, so any semantic LLVM
    change (compiler, pins, or LLVM cmake args) invalidates CIRCT even
    though CIRCT's own args are unchanged. Layers CIRCT's own semantic
    cmake args and CIRCT source pin on top.
    """
    return json.dumps(
        {
            "llvm_build_identity": json.loads(llvm_identity),
            "circt_commit": circt_commit,
            "circt_semantic_cmake_args": list(CIRCT_SEMANTIC_CMAKE_ARGS),
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


def llvm_stamp_matches(stamp: str, identity: str) -> bool:
    if stamp == identity:
        return True
    try:
        stamped = json.loads(stamp)
        expected = json.loads(identity)
        dependencies = stamped["dependencies"].copy()
        dependencies.pop("circt", None)
        stamped["dependencies"] = dependencies
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    return stamped == expected


def invalidate_llvm_readiness(paths: Paths) -> None:
    paths.llvm_stamp.unlink(missing_ok=True)
    paths.circt_stamp.unlink(missing_ok=True)


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


def configure_circt(
    paths: Paths,
    compilers: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    run([
        "cmake", "-G", "Ninja",
        "-S", str(paths.circt_root),
        "-B", str(paths.circt_build),
        f"-DCMAKE_C_COMPILER={compilers[0][0]}",
        f"-DCMAKE_CXX_COMPILER={compilers[1][0]}",
        *CIRCT_SEMANTIC_CMAKE_ARGS,
        f"-DMLIR_DIR={paths.mlir_dir}",
        f"-DLLVM_DIR={paths.cmake_llvm_dir}",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    ])


def configure_loom(paths: Paths, circt_dir: str | None) -> None:
    cmd = [
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
    ]
    if circt_dir is not None:
        cmd.append(f"-DCIRCT_DIR={circt_dir}")
    run(cmd)


def llvm_artifacts_present(paths: Paths) -> bool:
    """Artifacts required to configure and build Loom against LLVM."""
    return (
        (paths.llvm_build / "build.ninja").exists()
        and (paths.mlir_dir / "MLIRConfig.cmake").exists()
        and (paths.cmake_llvm_dir / "LLVMConfig.cmake").exists()
        and (paths.cmake_clang_dir / "ClangConfig.cmake").exists()
        and paths.llvm_lit.is_file()
        and os.access(paths.llvm_lit, os.X_OK)
    )


def _sync_llvm_locked(
    paths: Paths,
    args: argparse.Namespace,
    state: DependencyState,
    compilers: tuple[tuple[str, str], tuple[str, str]],
    always_build: bool,
    prev_stamp: str,
) -> str:
    """LLVM ensure/build phase. Caller already holds the exclusive build lock.

    Returns the validated LLVM build identity so a dependent phase (CIRCT)
    can derive an identity that drifts with the exact LLVM build rather
    than only with the dependency pins.

    prev_stamp is an in-memory snapshot taken while the lock is held. An
    explicit build removes the on-disk readiness stamp before entering this
    phase, but may still use the snapshot to retain an incremental build.
    """
    current = llvm_build_identity(state.llvm_commit, compilers)
    rebuild = (
        not llvm_stamp_matches(prev_stamp, current)
        or not llvm_artifacts_present(paths)
    )
    if rebuild:
        if not always_build:
            invalidate_llvm_readiness(paths)
        if paths.llvm_build.exists():
            info(
                f"LLVM build is incomplete or its identity changed; removing "
                f"{paths.llvm_build}"
            )
            shutil.rmtree(paths.llvm_build)
        configure_llvm(paths, compilers)
    if always_build or rebuild:
        run(["cmake", "--build", str(paths.llvm_build), f"-j{args.jobs}"])
        if not llvm_artifacts_present(paths):
            die(
                f"LLVM build at {paths.llvm_build} did not produce the "
                "required CMake package artifacts; refusing to stamp an "
                "incomplete build"
            )
        write_stamp(paths.llvm_stamp, current)
    return current


def sync_shared_llvm(
    paths: Paths,
    args: argparse.Namespace,
    always_build: bool,
) -> str:
    if not paths.is_main:
        info(f"checking shared LLVM under main worktree {paths.main}")
    if is_nfs(paths.llvm_root):
        warn(
            f"shared LLVM tree {paths.llvm_root} appears to live on NFS; "
            "flock semantics across hosts are unreliable"
        )
    with SharedProductLock(
        paths.llvm_lock,
        paths.llvm_lock_turnstile,
        args.lock_timeout,
        shared=False,
    ):
        prev_stamp = read_stamp(paths.llvm_stamp)
        if always_build:
            invalidate_llvm_readiness(paths)
            check_git_version()
        state = check_dependency_pins(paths)
        compilers = check_llvm_compilers()
        return _sync_llvm_locked(
            paths, args, state, compilers, always_build, prev_stamp
        )


def build_llvm(paths: Paths, args: argparse.Namespace) -> None:
    sync_shared_llvm(paths, args, always_build=True)


def circt_artifacts_present(paths: Paths) -> bool:
    """Artifacts required to offer the configured CIRCT package to Loom."""
    return (
        (paths.circt_build / "build.ninja").exists()
        and (paths.circt_cmake_dir / "CIRCTConfig.cmake").exists()
    )


def _sync_circt_locked(
    paths: Paths,
    args: argparse.Namespace,
    compilers: tuple[tuple[str, str], tuple[str, str]],
    llvm_identity: str,
    circt_commit: str,
    always_build: bool,
    prev_stamp: str,
) -> None:
    """CIRCT ensure/build phase. Caller already holds the exclusive build lock.

    Reusing the LLVM lock (rather than a CIRCT-only lock) is what makes
    the LLVM+CIRCT operation race-safe: the LLVM build cannot be rebuilt
    or wiped by another worktree while CIRCT configures and links against
    it. A separate lock released as soon as the LLVM phase ended would
    reopen exactly that window.

    prev_stamp is the caller's in-memory snapshot of the CIRCT stamp taken
    before the caller invalidated it on disk. It drives rebuild selection
    only, so a still-matching complete build is reused with an incremental
    cmake --build instead of being wiped and reconfigured.
    """
    current = circt_build_identity(llvm_identity, circt_commit)
    rebuild = prev_stamp != current or not circt_artifacts_present(paths)
    if rebuild:
        if paths.circt_build.exists():
            info(
                f"CIRCT build is incomplete or its identity changed; "
                f"removing {paths.circt_build}"
            )
            shutil.rmtree(paths.circt_build)
        configure_circt(paths, compilers)
    if always_build or rebuild:
        run([
            "cmake", "--build", str(paths.circt_build),
            f"-j{args.jobs}",
        ])
        if not circt_artifacts_present(paths):
            die(
                f"CIRCT build at {paths.circt_build} did not produce "
                f"{paths.circt_cmake_dir / 'CIRCTConfig.cmake'}; refusing "
                "to stamp an incomplete package"
            )
        write_stamp(paths.circt_stamp, current)


def sync_shared_circt(
    paths: Paths,
    args: argparse.Namespace,
    always_build: bool,
) -> None:
    """Build shared CIRCT against the shared LLVM under a single lock.

    The LLVM lock is held continuously across the LLVM ensure phase and
    the CIRCT configure/build. CIRCT therefore always links against the
    exact LLVM whose identity its own stamp is derived from.

    For an explicit build the sole readiness stamp is snapshotted in memory
    and then invalidated on disk as soon as the lock is held, before the
    failure-capable dependency, compiler, and LLVM prerequisites, so any
    failure in the pipeline leaves no stale CIRCT advertised. The snapshot is
    handed to the CIRCT phase only to reuse a still-matching complete build
    incrementally; the stamp is rewritten only after a build passes artifact
    validation.
    """
    if not paths.is_main:
        info(f"checking shared CIRCT under main worktree {paths.main}")
    if is_nfs(paths.circt_root):
        warn(
            f"shared CIRCT tree {paths.circt_root} appears to live on NFS; "
            "flock semantics across hosts are unreliable"
        )
    with SharedProductLock(
        paths.llvm_lock,
        paths.llvm_lock_turnstile,
        args.lock_timeout,
        shared=False,
    ):
        prev_stamp = read_stamp(paths.circt_stamp)
        if always_build:
            paths.circt_stamp.unlink(missing_ok=True)
            check_git_version()
        llvm_prev_stamp = read_stamp(paths.llvm_stamp)
        state = check_dependency_pins(paths)
        compilers = check_llvm_compilers()
        llvm_identity = _sync_llvm_locked(
            paths, args, state, compilers, False, llvm_prev_stamp
        )
        _sync_circt_locked(
            paths,
            args,
            compilers,
            llvm_identity,
            state.circt_commit,
            always_build,
            prev_stamp,
        )


def build_circt(paths: Paths, args: argparse.Namespace) -> None:
    sync_shared_circt(paths, args, always_build=True)


def available_circt_dir(
    paths: Paths,
    llvm_identity: str,
    circt_commit: str,
) -> str | None:
    """CIRCT package dir offered to the loom configure, or None.

    Never builds CIRCT. A CIRCT build is offered only when both the
    exported CIRCTConfig.cmake and matching CIRCT stamp are present. The
    stamp is derived from the current LLVM and CIRCT identities, so an old,
    incomplete, or unstamped CIRCT build is never silently offered.
    """
    if not circt_artifacts_present(paths):
        return None
    if read_stamp(paths.circt_stamp) != circt_build_identity(
        llvm_identity, circt_commit
    ):
        return None
    return str(paths.circt_cmake_dir)


def loom_build_is_stale(paths: Paths) -> bool:
    """Loom was configured against a shared LLVM that no longer exists.

    Detected by checking whether the cmake packages the loom build was
    pointed at are still on disk. If not, the cached build.ninja will
    fail at link time, so we wipe and reconfigure proactively. Both
    MLIR, LLVM, and Clang configs must be present because configure_loom()
    passes all three package directories.
    """
    bn = paths.loom_build / "build.ninja"
    if not bn.exists():
        return False
    if not (paths.mlir_dir / "MLIRConfig.cmake").exists():
        return True
    if not (paths.cmake_llvm_dir / "LLVMConfig.cmake").exists():
        return True
    return not (paths.cmake_clang_dir / "ClangConfig.cmake").exists()


def inspect_llvm_readiness(
    paths: Paths,
) -> tuple[
    DependencyState,
    tuple[tuple[str, str], tuple[str, str]],
    str,
    bool,
]:
    state = check_dependency_pins(paths)
    compilers = check_llvm_compilers()
    identity = llvm_build_identity(state.llvm_commit, compilers)
    ready = (
        llvm_stamp_matches(read_stamp(paths.llvm_stamp), identity)
        and llvm_artifacts_present(paths)
    )
    return state, compilers, identity, ready


@contextmanager
def shared_llvm_lease(paths: Paths, args: argparse.Namespace):
    """Yield validated shared LLVM state while holding a reader lease.

    The common ready path takes only the shared product lock. Repair releases
    that reader lock before taking the exclusive writer lock, revalidates, and
    reacquires a fresh reader lock after successful repair.
    """
    if not paths.is_main:
        info(f"checking shared LLVM under main worktree {paths.main}")
    if is_nfs(paths.llvm_root):
        warn(
            f"shared LLVM tree {paths.llvm_root} appears to live on NFS; "
            "flock semantics across hosts are unreliable"
        )
    with SharedProductLock(
        paths.llvm_lock,
        paths.llvm_lock_turnstile,
        args.lock_timeout,
        shared=True,
    ):
        state, _, identity, ready = inspect_llvm_readiness(paths)
        if ready:
            yield state, identity
            return

    with SharedProductLock(
        paths.llvm_lock,
        paths.llvm_lock_turnstile,
        args.lock_timeout,
        shared=False,
    ):
        prev_stamp = read_stamp(paths.llvm_stamp)
        state, compilers, identity, ready = inspect_llvm_readiness(paths)
        if not ready:
            identity = _sync_llvm_locked(
                paths, args, state, compilers, False, prev_stamp
            )

    with SharedProductLock(
        paths.llvm_lock,
        paths.llvm_lock_turnstile,
        args.lock_timeout,
        shared=True,
    ):
        state, _, current_identity, ready = inspect_llvm_readiness(paths)
        if not ready or current_identity != identity:
            die(
                "shared LLVM readiness changed after repair; refusing to "
                "consume stale products"
            )
        yield state, current_identity


def _build_loom_with_lease(
    paths: Paths,
    args: argparse.Namespace,
    state: DependencyState,
    llvm_identity: str,
) -> None:
    if loom_build_is_stale(paths):
        info(
            f"loom build at {paths.loom_build} references a missing "
            f"shared LLVM ({paths.mlir_dir}); wiping and reconfiguring"
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

    # `make loom` never builds CIRCT. It only offers an already-built,
    # stamped CIRCT matching the identities held by this reader lease.
    circt_dir = available_circt_dir(
        paths, llvm_identity, state.circt_commit
    )
    if bn.exists() and read_cmake_cache_entry(
        paths.loom_build, "CIRCT_DIR"
    ) != (circt_dir or ""):
        info(
            "loom build CIRCT_DIR changed; "
            f"removing {paths.loom_build} and reconfiguring"
        )
        shutil.rmtree(paths.loom_build, ignore_errors=True)
    if not bn.exists():
        configure_loom(paths, circt_dir)
    run(["cmake", "--build", str(paths.loom_build), f"-j{args.jobs}"])


def build_loom(paths: Paths, args: argparse.Namespace) -> None:
    with shared_llvm_lease(paths, args) as (state, llvm_identity):
        _build_loom_with_lease(paths, args, state, llvm_identity)


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
    if is_nfs(paths.circt_root):
        warn(
            f"shared CIRCT tree {paths.circt_root} appears to live on NFS; "
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
    print(f"circt_build     {paths.circt_build}")
    print(f"circt_stamp     {paths.circt_stamp}")
    print(
        f"circt_stamp_val {read_stamp(paths.circt_stamp) or '(unset)'}"
    )
    # Report the concrete artifacts rather than a single readiness verdict:
    # an old or unstamped build can leave the config on disk without being
    # actually usable, so availability is decided elsewhere from the stamp.
    print(
        "circt_config    "
        + str((paths.circt_cmake_dir / "CIRCTConfig.cmake").exists())
    )


def cmd_externals_root(paths: Paths, args: argparse.Namespace) -> None:
    check_linked_submodule_hygiene(paths)
    print(paths.externals_root)


def cmd_build_llvm(paths: Paths, args: argparse.Namespace) -> None:
    build_llvm(paths, args)


def cmd_build_loom(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
    check_loom_compilers()
    build_loom(paths, args)


def cmd_clean(paths: Paths, args: argparse.Namespace) -> None:
    if paths.loom_build.exists():
        info(f"removing {paths.loom_build}")
        shutil.rmtree(paths.loom_build, ignore_errors=True)


def cmd_build_circt(paths: Paths, args: argparse.Namespace) -> None:
    build_circt(paths, args)


def cmd_distclean(paths: Paths, args: argparse.Namespace) -> None:
    if paths.loom_build.exists():
        info(f"removing {paths.loom_build}")
        shutil.rmtree(paths.loom_build, ignore_errors=True)
    if paths.is_main:
        # Both shared builds are removed under the same exclusive product
        # transaction used by build writers.
        with SharedProductLock(
            paths.llvm_lock,
            paths.llvm_lock_turnstile,
            args.lock_timeout,
            shared=False,
        ):
            invalidate_llvm_readiness(paths)
            if paths.llvm_build.exists():
                info(f"removing shared {paths.llvm_build}")
                shutil.rmtree(paths.llvm_build, ignore_errors=True)
            if paths.circt_build.exists():
                info(f"removing shared {paths.circt_build}")
                shutil.rmtree(paths.circt_build, ignore_errors=True)
    else:
        info(
            f"distclean from linked worktree only removes {paths.loom_build}; "
            f"shared LLVM at {paths.llvm_build} and CIRCT at "
            f"{paths.circt_build} preserved"
        )


def cmd_test(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
    check_loom_compilers()
    with shared_llvm_lease(paths, args) as (state, llvm_identity):
        _build_loom_with_lease(paths, args, state, llvm_identity)
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
    if len(sys.argv) >= 2 and sys.argv[1] == _SUPERVISE_COMMAND:
        _supervise_command(sys.argv[2:])

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=os.getcwd(),
                   help="worktree root (defaults to CWD)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--lock-timeout", type=validate_lock_timeout, default=1800.0,
                   help="seconds to wait for the shared LLVM lock")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "externals-root", "build-llvm", "build-circt",
                 "build-loom", "clean", "distclean", "test"):
        sub.add_parser(name)
    args = p.parse_args()

    paths = Paths(Path(args.root))
    dispatch = {
        "doctor": cmd_doctor,
        "externals-root": cmd_externals_root,
        "build-llvm": cmd_build_llvm,
        "build-circt": cmd_build_circt,
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
