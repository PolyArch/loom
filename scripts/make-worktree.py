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
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_GIT = (2, 7)
LLVM_C_COMPILER = "gcc"
LLVM_CXX_COMPILER = "g++"
LLVM_GCC_MIN = (7, 4)
LOOM_C_COMPILER = "clang"
LOOM_CXX_COMPILER = "clang++"
LOOM_CLANG_MIN = (21, 1, 8)
HEAVY_LIT_TESTS = (
    "artifacts/cmsis_cgra_status_rollup.mlir",
    "artifacts/cgra_sim_evidence_sweep.mlir",
    "artifacts/artifact_gates.mlir",
    "artifacts/sim_cycle_summary.mlir",
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
                   minimum: tuple[int, ...]) -> tuple[int, ...]:
    version, first_line = compiler_version(tool)
    if version_less(version, minimum):
        die(
            f"{nice_name} must be at least {format_version(minimum)}, "
            f"got {format_version(version)} from {first_line}"
        )
    info(f"{nice_name} {format_version(version)} ok ({tool})")
    return version


def check_llvm_compilers() -> None:
    check_compiler(LLVM_C_COMPILER, "GCC C compiler", LLVM_GCC_MIN)
    check_compiler(LLVM_CXX_COMPILER, "GCC C++ compiler", LLVM_GCC_MIN)


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
        externals = self.main / "externals"
        llvm_external = externals / "llvm"
        self.llvm_root = llvm_external
        self.llvm_src = llvm_external / "llvm"
        self.llvm_build = llvm_external / "build"
        self.llvm_lock = externals / ".loom-build.llvm.lock"
        self.llvm_stamp = externals / ".loom-build.llvm.stamp"
        self.loom_build = self.root / "build"
        self.mlir_dir = self.llvm_build / "lib" / "cmake" / "mlir"
        self.cmake_llvm_dir = self.llvm_build / "lib" / "cmake" / "llvm"
        self.cmake_clang_dir = self.llvm_build / "lib" / "cmake" / "clang"
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


def configure_llvm(paths: Paths) -> None:
    run([
        "cmake", "-G", "Ninja",
        "-S", str(paths.llvm_src),
        "-B", str(paths.llvm_build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={LLVM_C_COMPILER}",
        f"-DCMAKE_CXX_COMPILER={LLVM_CXX_COMPILER}",
        "-DLLVM_ENABLE_PROJECTS=mlir;clang",
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
        f"-DCMAKE_C_COMPILER={LOOM_C_COMPILER}",
        f"-DCMAKE_CXX_COMPILER={LOOM_CXX_COMPILER}",
        f"-DMLIR_DIR={paths.mlir_dir}",
        f"-DLLVM_DIR={paths.cmake_llvm_dir}",
        f"-DClang_DIR={paths.cmake_clang_dir}",
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
        if build_ninja.exists() and not cmake_cache_uses_compilers(
            paths.llvm_build, LLVM_C_COMPILER, LLVM_CXX_COMPILER
        ):
            info(
                f"LLVM build compiler changed to {LLVM_C_COMPILER}/"
                f"{LLVM_CXX_COMPILER}; removing {paths.llvm_build}"
            )
            shutil.rmtree(paths.llvm_build, ignore_errors=True)
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
    mlir_cfg = paths.mlir_dir / "MLIRConfig.cmake"
    clang_cfg = paths.cmake_clang_dir / "ClangConfig.cmake"
    current = llvm_source_id(paths)
    prev = read_stamp(paths.llvm_stamp)
    if not mlir_cfg.exists():
        info(f"shared MLIR not found at {paths.llvm_build}; building it now")
        check_llvm_compilers()
        build_llvm(paths, args)
        return
    if not clang_cfg.exists():
        info(
            f"shared Clang not found at {paths.cmake_clang_dir}; "
            "rebuilding LLVM with clang enabled"
        )
        check_llvm_compilers()
        build_llvm(paths, args)
        return
    if prev and prev != current:
        info(
            f"shared LLVM source id drifted ({prev} -> {current}); "
            "rebuilding before loom"
        )
        check_llvm_compilers()
        build_llvm(paths, args)


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
    print(f"llvm_c          {compiler_status(LLVM_C_COMPILER, LLVM_GCC_MIN)}")
    print(f"llvm_cxx        {compiler_status(LLVM_CXX_COMPILER, LLVM_GCC_MIN)}")
    print(f"loom_c          {compiler_status(LOOM_C_COMPILER, LOOM_CLANG_MIN)}")
    print(f"loom_cxx        {compiler_status(LOOM_CXX_COMPILER, LOOM_CLANG_MIN)}")
    print(f"loom_build      {paths.loom_build}")
    print(f"loom_stale      {loom_build_is_stale(paths)}")


def cmd_build_llvm(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
    check_llvm_compilers()
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


def lit_opts(*parts: str) -> str:
    return " ".join(part for part in parts if part).strip()


def lit_extra_args(extra: str) -> list[str]:
    return shlex.split(extra) if extra else []


def explicit_lit_workers(extra: str) -> int | None:
    patterns = (
        r"(^|\s)-j([0-9]+)(?=\s|$)",
        r"(^|\s)-j\s+([0-9]+)(?=\s|$)",
        r"(^|\s)--workers=([0-9]+)(?=\s|$)",
        r"(^|\s)--workers\s+([0-9]+)(?=\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, extra)
        if match:
            return int(match.group(2))
    return None


def lit_jobs_opt(jobs: int, extra: str) -> str:
    if explicit_lit_workers(extra) is not None:
        return ""
    if re.search(r"(^|\s)--workers(=|\s|$)", extra):
        return ""
    return f"-j{jobs}"


def lit_jobs_arg(jobs: int, extra: str) -> list[str]:
    value = lit_jobs_opt(jobs, extra)
    return [value] if value else []


def positive_env_int(env: dict[str, str], name: str) -> int | None:
    value = env.get(name, "").strip()
    if not value:
        return None
    if not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
        die(f"{name} must be a positive integer")
    return int(value)


def heavy_lit_workers(total_jobs: int, extra: str, env: dict[str, str]) -> int:
    explicit = explicit_lit_workers(extra)
    if explicit is not None:
        return max(1, explicit)
    configured = positive_env_int(env, "LOOM_HEAVY_LIT_WORKERS")
    if configured is not None:
        return max(1, min(total_jobs, configured))
    return max(1, min(total_jobs, max(2, min(4, total_jobs // 6))))


def heavy_nested_jobs(total_jobs: int, workers: int, extra: str, env: dict[str, str]) -> int:
    explicit = explicit_lit_workers(extra)
    if explicit is not None:
        return max(1, explicit)
    configured = positive_env_int(env, "LOOM_HEAVY_TEST_JOBS")
    if configured is not None:
        return configured
    return max(1, (total_jobs * 2) // (3 * workers))


def broad_artifact_jobs(total_jobs: int, heavy_workers: int, heavy_nested: int) -> int:
    return max(1, total_jobs - (heavy_workers * heavy_nested))


def broad_filter_out_pattern() -> str:
    return "|".join(["techmap/perf", *(re.escape(test) for test in HEAVY_LIT_TESTS)])


def run_with_lit_filter(
    cmd: list[str],
    lit_filter: Path,
    env: dict[str, str] | None = None,
) -> None:
    proc, py = start_lit_filter(cmd, lit_filter, env)
    proc_rc, py_rc = wait_lit_filter(proc, py)
    if proc_rc != 0:
        sys.exit(proc_rc)
    if py_rc != 0:
        sys.exit(py_rc)


def start_lit_filter(
    cmd: list[str],
    lit_filter: Path,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen, subprocess.Popen]:
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    py = subprocess.Popen(
        [sys.executable, str(lit_filter)],
        stdin=proc.stdout,
    )
    proc.stdout.close()
    return proc, py


def wait_lit_filter(proc: subprocess.Popen, py: subprocess.Popen) -> tuple[int, int]:
    py_rc = py.wait()
    proc_rc = proc.wait()
    return proc_rc, py_rc


def run_lit_filters_parallel(
    runs: list[tuple[list[str], dict[str, str]]],
    lit_filter: Path,
) -> None:
    launched = [start_lit_filter(cmd, lit_filter, env) for cmd, env in runs]
    failure = 0
    for proc, py in launched:
        proc_rc, py_rc = wait_lit_filter(proc, py)
        if failure == 0 and proc_rc != 0:
            failure = proc_rc
        if failure == 0 and py_rc != 0:
            failure = py_rc
    if failure != 0:
        sys.exit(failure)


def cmd_test(paths: Paths, args: argparse.Namespace) -> None:
    check_git_version()
    check_loom_compilers()
    build_loom(paths, args)
    base_env = os.environ.copy()
    extra = base_env.get("LIT_OPTS", "").strip()
    nested_jobs = explicit_lit_workers(extra) or args.jobs
    extra_args = lit_extra_args(extra)
    broad_env = base_env.copy()
    broad_env.setdefault("LOOM_TEST_JOBS", str(nested_jobs))
    broad_env["LIT_OPTS"] = lit_opts(extra)
    lit_filter = paths.root / "test" / "lit_top_slowest.py"
    broad_cmd = [
        str(paths.llvm_lit),
        "-sv",
        "--time-tests",
        *lit_jobs_arg(args.jobs, extra),
        "--filter-out",
        broad_filter_out_pattern(),
        *extra_args,
        str(paths.loom_build / "test"),
    ]

    heavy_workers = heavy_lit_workers(args.jobs, extra, base_env)
    heavy_nested = heavy_nested_jobs(args.jobs, heavy_workers, extra, base_env)
    broad_env.setdefault("LOOM_ARTIFACT_TEST_JOBS", str(broad_artifact_jobs(args.jobs, heavy_workers, heavy_nested)))
    heavy_env = base_env.copy()
    heavy_env.setdefault("LOOM_TEST_JOBS", str(heavy_nested))
    heavy_env.setdefault("LOOM_ARTIFACT_TEST_JOBS", str(heavy_nested))
    heavy_env["LIT_OPTS"] = lit_opts(extra)
    heavy_cmd = [
        str(paths.llvm_lit),
        "-sv",
        "--time-tests",
        f"-j{heavy_workers}",
        *extra_args,
        *(str(paths.loom_build / "test" / test) for test in HEAVY_LIT_TESTS),
    ]
    run_lit_filters_parallel(
        [
            (broad_cmd, broad_env),
            (heavy_cmd, heavy_env),
        ],
        lit_filter,
    )

    perf_env = base_env.copy()
    perf_env.setdefault("LOOM_TEST_JOBS", str(nested_jobs))
    perf_env["LIT_OPTS"] = lit_opts(extra)
    run_with_lit_filter(
        [
            str(paths.llvm_lit),
            "-sv",
            "--time-tests",
            "-j1",
            *extra_args,
            str(paths.loom_build / "test" / "techmap" / "perf"),
        ],
        lit_filter,
        env=perf_env,
    )


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
