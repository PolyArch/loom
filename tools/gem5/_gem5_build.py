"""Shared gem5 build helpers for linked worktree support."""

import os
import pathlib
import re
import shutil
import subprocess


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEM5_BIN = REPO_ROOT / "build/RISCV/gem5.opt"
GEM5_RISCV_DIR = REPO_ROOT / "build/RISCV"

# All loom source directories whose changes require a gem5 rebuild.
# This covers the EXTRAS sources (src/gem5dev/) and all transitive
# header dependencies used by the Simulator code compiled into gem5.
GEM5_REBUILD_INPUTS = [
    REPO_ROOT / "src/gem5dev",
    REPO_ROOT / "lib/loom/Simulator",
    REPO_ROOT / "include/loom/Simulator",
    REPO_ROOT / "include/loom/Mapper",
    REPO_ROOT / "include/loom/Dialect/Fabric",
]


def derive_main_root(repo_root: pathlib.Path):
    """Return the main worktree root, or None if this IS the main worktree.

    Parses the .git file (present in linked worktrees) to find the main
    worktree root, using the same logic as CMakeLists.txt.
    """
    dot_git = repo_root / ".git"
    if not dot_git.exists() or dot_git.is_dir():
        return None
    content = dot_git.read_text(encoding="utf-8").strip()
    m = re.match(r"gitdir:\s*(.+)", content)
    if not m:
        return None
    gitdir = pathlib.Path(m.group(1))
    if not gitdir.is_absolute():
        gitdir = (repo_root / gitdir).resolve()
    else:
        gitdir = gitdir.resolve()
    if "/.git/worktrees/" not in str(gitdir):
        return None
    # <main>/.git/worktrees/<name> -> go up 3 levels
    main_root = gitdir.parent.parent.parent
    if not main_root.is_dir():
        return None
    return main_root


def newest_input_mtime(paths):
    """Return the newest mtime across all files under the given paths.

    Uses os.walk with followlinks=True so that symlinked directories
    (common in linked worktrees) are traversed correctly.
    """
    newest = 0.0
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            newest = max(newest, root.stat().st_mtime)
            continue
        for dirpath, _, filenames in os.walk(str(root), followlinks=True):
            for fname in filenames:
                newest = max(newest, os.stat(os.path.join(dirpath, fname)).st_mtime)
    return newest


def gem5_needs_rebuild(gem5_bin: pathlib.Path):
    """Check whether gem5 needs rebuilding.

    Returns (needs_rebuild: bool, reason: str).
    """
    if os.environ.get("LOOM_GEM5_FORCE_REBUILD", "") not in ("", "0"):
        return True, "LOOM_GEM5_FORCE_REBUILD is set"
    if not gem5_bin.exists():
        return True, f"missing {gem5_bin}"
    binary_mtime = gem5_bin.stat().st_mtime
    newest_input = newest_input_mtime(GEM5_REBUILD_INPUTS)
    if newest_input > binary_mtime:
        if GEM5_RISCV_DIR.is_symlink():
            return True, "EXTRAS sources newer than main worktree gem5.opt"
        return True, "local gem5/simulator sources are newer than gem5.opt"
    if GEM5_RISCV_DIR.is_symlink():
        return False, "reusing main worktree gem5.opt (EXTRAS unchanged)"
    return False, "existing gem5.opt is up to date"


def _find_ccache_wrappers_dir():
    """Find the ccache compiler wrappers directory.

    ccache provides compiler-named symlinks (e.g., g++ -> ccache) in a
    directory like /usr/lib64/ccache/. Prepending this to PATH makes scons
    (and any build system) use ccache transparently without modifying
    compiler variables -- which gem5's SConstruct does not support.
    """
    candidates = [
        "/usr/lib64/ccache",
        "/usr/lib/ccache",
        "/usr/local/lib/ccache",
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


def run_scons_build(repo_root: pathlib.Path):
    """Run the scons gem5 build with ccache if available."""
    riscv_dir = repo_root / "build/RISCV"
    if riscv_dir.is_symlink():
        riscv_dir.unlink()

    jobs = max(os.cpu_count() or 1, 1)

    # Inject ccache via PATH wrappers (gem5's SConstruct ignores CXX arguments)
    path_prefix = ""
    ccache_dir = _find_ccache_wrappers_dir()
    if ccache_dir:
        path_prefix = f"export PATH={ccache_dir}:$PATH && "

    cmd = (
        "source /etc/profile.d/modules.sh && "
        "module load scons && "
        f"{path_prefix}"
        f"scons -C {repo_root / 'externals/gem5'} "
        f"EXTRAS={repo_root / 'src/gem5dev'} "
        f"-j{jobs} build/RISCV/gem5.opt"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, cwd=repo_root)


def build_gem5(force_rebuild: bool = False):
    """Build gem5 if needed, returning the path to the binary.

    Handles linked worktree symlink lifecycle: if build/RISCV is a symlink
    to the main worktree and EXTRAS sources haven't changed, reuses the
    main binary. Otherwise breaks the symlink and does a local build.
    """
    gem5_bin = GEM5_BIN
    if not force_rebuild:
        needs_rebuild, reason = gem5_needs_rebuild(gem5_bin)
        if not needs_rebuild:
            print(f"reusing gem5 binary: {gem5_bin} ({reason})")
            return gem5_bin
    else:
        reason = "requested by --rebuild-gem5"

    print(f"rebuilding gem5 binary: {reason}")
    run_scons_build(REPO_ROOT)
    return gem5_bin
