#!/usr/bin/env python3
"""Worktree-aware build dispatcher used by the top-level Makefile.

Path resolution and the edge cases that pure Make/shell handles poorly
all live here:

  * `resolve_main_worktree` parses `git worktree list --porcelain`,
    which is supported on every supported git version and avoids the
    `git rev-parse --path-format=absolute` flag (git >= 2.31).
  * Paths are run through `realpath` so a symlinked entry into the main
    worktree still compares equal to the canonical worktree path.
  * The shared LLVM build is gated by an fcntl.flock with a configurable
    timeout, so a wedged build cannot hang every other worktree forever.
  * A stamp file records the LLVM source commit (or fallback id) used to
    populate the shared build. If a sibling worktree advances the
    submodule pointer, the next builder reconfigures + rebuilds instead
    of silently linking against stale headers.
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
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_GIT = (2, 7)


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
            stderr=subprocess.DEVNULL,
        ).decode()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return real(root)
    for line in out.splitlines():
        if line.startswith("worktree "):
            return real(Path(line.split(" ", 1)[1]))
    return real(root)


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
        externals = self.main / "externals" / "llvm"
        self.llvm_root = externals
        self.llvm_src = externals / "llvm"
        self.llvm_build = externals / "build"
        self.llvm_lock = externals / ".build.lock"
        self.llvm_stamp = externals / ".build.stamp"
        self.loom_build = self.root / "build"
        self.mlir_dir = self.llvm_build / "lib" / "cmake" / "mlir"
        self.cmake_llvm_dir = self.llvm_build / "lib" / "cmake" / "llvm"
        self.llvm_lit = self.llvm_build / "bin" / "llvm-lit"

    @property
    def is_main(self) -> bool:
        return self.main == self.root


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


def llvm_source_id(paths: Paths) -> str:
    """Identifier captured in the stamp file.

    `git rev-parse HEAD` works for both submodule and plain checkouts.
    The fallback only fires when the source tree is not a git checkout
    at all, in which case we degrade to a sentinel that never matches a
    real commit (forcing a reconfigure on first use after the script is
    upgraded, but stable thereafter).
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(paths.llvm_src), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            return f"git:{out}"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return "unknown:non-git-source"


def read_stamp(path: Path) -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return ""


def write_stamp(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")


def configure_llvm(paths: Paths) -> None:
    run([
        "cmake", "-G", "Ninja",
        "-S", str(paths.llvm_src),
        "-B", str(paths.llvm_build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DLLVM_TARGETS_TO_BUILD=host",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_ENABLE_RTTI=ON",
        "-DLLVM_INSTALL_UTILS=ON",
        "-DLLVM_CCACHE_BUILD=ON",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLVM_BUILD_LLVM_DYLIB=ON",
        "-DLLVM_LINK_LLVM_DYLIB=ON",
    ])


def configure_loom(paths: Paths) -> None:
    run([
        "cmake", "-G", "Ninja",
        "-S", str(paths.root),
        "-B", str(paths.loom_build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
        f"-DMLIR_DIR={paths.mlir_dir}",
        f"-DLLVM_DIR={paths.cmake_llvm_dir}",
        f"-DLLVM_EXTERNAL_LIT={paths.llvm_lit}",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    ])


def build_llvm(paths: Paths, args: argparse.Namespace) -> None:
    """Configure (if needed) and build the shared LLVM under flock."""
    if not paths.is_main:
        info(f"building shared LLVM under main worktree {paths.main}")
    if is_nfs(paths.llvm_root):
        warn(
            f"shared LLVM tree {paths.llvm_root} appears to live on NFS; "
            "flock semantics across hosts are unreliable"
        )
    with FileLock(paths.llvm_lock, args.lock_timeout):
        current = llvm_source_id(paths)
        prev = read_stamp(paths.llvm_stamp)
        build_ninja = paths.llvm_build / "build.ninja"
        if not build_ninja.exists():
            configure_llvm(paths)
        elif prev and prev != current:
            info(
                f"LLVM source id changed ({prev} -> {current}); "
                "reconfiguring shared build"
            )
            configure_llvm(paths)
        run(["cmake", "--build", str(paths.llvm_build), f"-j{args.jobs}"])
        write_stamp(paths.llvm_stamp, current)


def loom_build_is_stale(paths: Paths) -> bool:
    """Loom was configured against a shared LLVM that no longer exists.

    Detected by checking whether the cmake package the loom build was
    pointed at is still on disk. If not, the cached build.ninja will
    fail at link time, so we wipe and reconfigure proactively.
    """
    bn = paths.loom_build / "build.ninja"
    if not bn.exists():
        return False
    return not (paths.mlir_dir / "MLIRConfig.cmake").exists()


def ensure_shared_llvm(paths: Paths, args: argparse.Namespace) -> None:
    cfg = paths.mlir_dir / "MLIRConfig.cmake"
    current = llvm_source_id(paths)
    prev = read_stamp(paths.llvm_stamp)
    if not cfg.exists():
        info(f"shared MLIR not found at {paths.llvm_build}; building it now")
        build_llvm(paths, args)
        return
    if prev and prev != current:
        info(
            f"shared LLVM source id drifted ({prev} -> {current}); "
            "rebuilding before loom"
        )
        build_llvm(paths, args)


def build_loom(paths: Paths, args: argparse.Namespace) -> None:
    if loom_build_is_stale(paths):
        info(
            f"loom build at {paths.loom_build} references a missing shared "
            f"LLVM ({paths.mlir_dir}); wiping and reconfiguring"
        )
        shutil.rmtree(paths.loom_build, ignore_errors=True)
    ensure_shared_llvm(paths, args)
    bn = paths.loom_build / "build.ninja"
    if not bn.exists():
        configure_loom(paths)
    run(["cmake", "--build", str(paths.loom_build), f"-j{args.jobs}"])


def cmd_doctor(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
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
    print(f"llvm_src        {paths.llvm_src}")
    print(f"llvm_build      {paths.llvm_build}")
    print(f"llvm_lock       {paths.llvm_lock}")
    print(f"llvm_stamp      {paths.llvm_stamp}")
    print(f"stamp_value     {read_stamp(paths.llvm_stamp) or '(unset)'}")
    print(f"current_src_id  {llvm_source_id(paths)}")
    print(f"loom_build      {paths.loom_build}")
    print(f"loom_stale      {loom_build_is_stale(paths)}")


def cmd_build_llvm(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
    build_llvm(paths, args)


def cmd_build_loom(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
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
    build_loom(paths, args)
    env = os.environ.copy()
    extra = env.get("LIT_OPTS", "").strip()
    env["LIT_OPTS"] = ("-sv --time-tests " + extra).strip()
    top = paths.root / "test" / "lit_top_slowest.py"
    cmake = subprocess.Popen(
        [
            "cmake", "--build", str(paths.loom_build),
            f"-j{args.jobs}", "--target", "check-fabric",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    py = subprocess.Popen(
        [sys.executable, str(top)],
        stdin=cmake.stdout,
    )
    # Closing our handle to cmake.stdout lets cmake see SIGPIPE if the
    # filter exits early.
    assert cmake.stdout is not None
    cmake.stdout.close()
    py_rc = py.wait()
    cmake_rc = cmake.wait()
    if cmake_rc != 0:
        sys.exit(cmake_rc)
    if py_rc != 0:
        sys.exit(py_rc)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=os.getcwd(),
                   help="worktree root (defaults to CWD)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--lock-timeout", type=float, default=1800.0,
                   help="seconds to wait for the shared LLVM lock")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "build-llvm", "build-loom",
                 "clean", "distclean", "test"):
        sub.add_parser(name)
    args = p.parse_args()

    paths = Paths(Path(args.root))
    dispatch = {
        "doctor": cmd_doctor,
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
