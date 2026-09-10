#!/usr/bin/env python3
"""Remove old scratch/cache files while preserving Markdown and symlinks.

Stop builds and simulations using the selected roots before applying cleanup.
Age is file modification time, not access time. Files are removed oldest first
across all roots; empty parent directories are then pruned. This can invalidate
old cache entries and experimental outputs, which must be regenerated.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import errno
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

from resolve_experiment_root import (
    EXTERNAL_TOOL_CACHE_DIRECTORY,
    EXTERNAL_TOOL_CACHE_MEMBERS,
)


GIB = 1 << 30
SECONDS_PER_DAY = 86400
PREVIEW_LINES = 20
SCAN_PROGRESS_FILES = 100_000
# Preserve cache markers and lock namespaces, including in nested test caches.
CACHE_CONTROLS = EXTERNAL_TOOL_CACHE_MEMBERS - {"entries", "command-entries"}
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


@dataclass(frozen=True, slots=True)
class Root:
    path: Path
    descriptor: int
    device: int


@dataclass(frozen=True, slots=True)
class Candidate:
    root: int
    relative: Path
    fingerprint: tuple[int, ...]
    allocated_bytes: int

    @property
    def modified_ns(self) -> int:
        return self.fingerprint[2]


def fingerprint(status: os.stat_result) -> tuple[int, ...]:
    return (status.st_dev, status.st_ino, status.st_mtime_ns, status.st_size)


def protected(path: Path, keep: list[Path]) -> bool:
    return (
        path.name.lower().endswith(".md")
        or path.name == ".git"
        or path.name in CACHE_CONTROLS
        or path.name.endswith(".lock")
        or any(path.is_relative_to(retained) for retained in keep)
    )


def scan(roots: list[Root], keep: list[Path], cutoff_ns: int) -> list[Candidate]:
    candidates = []
    files_seen = 0

    def fail(error: OSError) -> None:
        raise error

    for index, root in enumerate(roots):
        for directory, children, files, descriptor in os.fwalk(
            ".", dir_fd=root.descriptor, follow_symlinks=False, onerror=fail
        ):
            relative = Path(directory)
            retained_children = []
            for name in children:
                status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    stat.S_ISDIR(status.st_mode)
                    and status.st_dev == root.device
                    and not protected(root.path / relative / name, keep)
                ):
                    retained_children.append(name)
            children[:] = retained_children
            for name in files:
                files_seen += 1
                if files_seen % SCAN_PROGRESS_FILES == 0:
                    print(
                        f"Scanned {files_seen} files; {len(candidates)} eligible.",
                        flush=True,
                    )
                path = relative / name
                if protected(root.path / path, keep):
                    continue
                status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    stat.S_ISREG(status.st_mode)
                    and status.st_dev == root.device
                    and status.st_mtime_ns <= cutoff_ns
                ):
                    candidates.append(
                        Candidate(
                            index, path, fingerprint(status), status.st_blocks * 512
                        )
                    )
    candidates.sort(
        key=lambda item: (item.modified_ns, str(roots[item.root].path / item.relative))
    )
    return candidates


def open_parent(root: Root, relative: Path) -> int:
    """Resolve every parent beneath the held root without following symlinks."""
    descriptor = os.dup(root.descriptor)
    try:
        for component in relative.parent.parts:
            child = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def remove(candidate: Candidate, root: Root) -> bool:
    try:
        descriptor = open_parent(root, candidate.relative)
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
            return False
        raise
    try:
        try:
            status = os.stat(
                candidate.relative.name, dir_fd=descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISREG(status.st_mode)
            or fingerprint(status) != candidate.fingerprint
        ):
            return False
        os.unlink(candidate.relative.name, dir_fd=descriptor)
        return True
    finally:
        os.close(descriptor)


def prune(roots: list[Root], parents: set[tuple[int, Path]]) -> None:
    for index, relative in sorted(
        parents, key=lambda item: len(item[1].parts), reverse=True
    ):
        try:
            descriptor = open_parent(roots[index], relative)
            try:
                os.rmdir(relative.name, dir_fd=descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            if error.errno not in (
                errno.ENOENT,
                errno.ENOTEMPTY,
                errno.EEXIST,
                errno.ENOTDIR,
                errno.ELOOP,
            ):
                raise


def select_roots(repository: Path, requested: list[Path] | None) -> list[Path]:
    paths = (
        requested
        if requested is not None
        else [repository / "temp", repository / "build" / EXTERNAL_TOOL_CACHE_DIRECTORY]
    )
    selected: list[Path] = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"cleanup root cannot be a symlink: {path}")
        if not path.exists() and requested is None:
            continue
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"cleanup root is not a directory: {path}")
        if path == Path.home().resolve() or repository.is_relative_to(path):
            raise ValueError(f"refusing broad cleanup root: {path}")
        if path.is_relative_to(repository):
            ignored = (
                subprocess.run(
                    ["git", "check-ignore", "--quiet", "--", str(path)],
                    cwd=repository,
                    check=False,
                ).returncode
                == 0
            )
            if not ignored:
                raise ValueError(f"repository cleanup root must be Git-ignored: {path}")
        selected.append(path)
    # Parent roots already cover their descendants; never count/delete twice.
    unique = sorted(set(selected), key=lambda path: (len(path.parts), str(path)))
    return [
        path
        for path in unique
        if not any(path != parent and path.is_relative_to(parent) for parent in unique)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  scripts/cleanup.py                         # Preview files older than 7 days
  scripts/cleanup.py --apply --free-gib 200   # Stop at 200 GiB free per filesystem
  scripts/cleanup.py --apply --older-than-days 30
  scripts/cleanup.py --root build/external-tool-cache --keep build/external-tool-cache/entries --older-than-days 0
  scripts/cleanup.py --root ~/.cache/ccache   # Explicit alternative cache root

Markdown (.md, case-insensitive), symlinks, mount subtrees, Git metadata and
cache lock/control files are preserved. --root replaces the default roots:
the repository temp directory and build/external-tool-cache/. Stop their producers
before --apply. Preview sizes are estimates; hardlinks/reflinks can reduce
actual reclaimed space. Application checks actual free space after deletions.
""",
    )
    parser.add_argument(
        "--root", type=Path, action="append", help="root to clean; repeatable"
    )
    parser.add_argument(
        "--keep",
        type=Path,
        action="append",
        default=[],
        help="file or directory subtree to retain; repeatable",
    )
    parser.add_argument(
        "--older-than-days",
        type=float,
        default=7,
        help="minimum modification age (default: 7; 0 includes recent files)",
    )
    parser.add_argument(
        "--free-gib",
        type=float,
        help="stop when each selected filesystem has this much free space",
    )
    parser.add_argument(
        "--apply", action="store_true", help="delete; default is preview only"
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.older_than_days) or args.older_than_days < 0:
        parser.error("--older-than-days must be finite and nonnegative")
    if args.free_gib is not None and (
        not math.isfinite(args.free_gib) or args.free_gib <= 0
    ):
        parser.error("--free-gib must be finite and positive")
    repository = Path(__file__).resolve().parent.parent
    paths = select_roots(repository, args.root)
    keep = [path.resolve() for path in args.keep]
    target = None if args.free_gib is None else int(args.free_gib * GIB)
    cutoff = time.time_ns() - int(args.older_than_days * SECONDS_PER_DAY * 10**9)
    with ExitStack() as stack:
        roots = []
        for path in paths:
            descriptor = os.open(path, DIRECTORY_FLAGS)
            stack.callback(os.close, descriptor)
            roots.append(Root(path, descriptor, os.fstat(descriptor).st_dev))
        devices = {root.device: root.path for root in roots}
        free = {
            device: shutil.disk_usage(path).free for device, path in devices.items()
        }
        if target is not None:
            for path in devices.values():
                if target > shutil.disk_usage(path).total:
                    raise ValueError(
                        f"free-space target exceeds filesystem capacity: {path}"
                    )
        if target is not None and all(value >= target for value in free.values()):
            print("Free-space target already met; nothing to remove.")
            return 0
        print("Mode:", "APPLY" if args.apply else "PREVIEW", flush=True)
        for path in paths:
            print("Root:", path, flush=True)
        candidates = scan(roots, keep, cutoff)
        count = allocated = changed = failures = 0
        parents: set[tuple[int, Path]] = set()
        for candidate in candidates:
            root = roots[candidate.root]
            if target is not None and free[root.device] >= target:
                continue
            if args.apply:
                try:
                    if not remove(candidate, root):
                        changed += 1
                        continue
                except OSError as error:
                    print(
                        f"Cannot remove {root.path / candidate.relative}: {error}",
                        file=sys.stderr,
                    )
                    failures += 1
                    continue
                for parent in candidate.relative.parents:
                    if parent != Path("."):
                        parents.add((candidate.root, parent))
                if target is not None:
                    free[root.device] = shutil.disk_usage(root.path).free
            else:
                free[root.device] += candidate.allocated_bytes
            count += 1
            allocated += candidate.allocated_bytes
            if count <= PREVIEW_LINES:
                date = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(candidate.modified_ns / 10**9)
                )
                print(
                    f"{date}  {candidate.allocated_bytes / GIB:.3f} GiB  "
                    f"{root.path / candidate.relative}"
                )
        if args.apply:
            prune(roots, parents)
            for device, path in devices.items():
                free[device] = shutil.disk_usage(path).free
                print(f"Free space on {path}: {free[device] / GIB:.3f} GiB")
        verb = "Removed" if args.apply else "Would remove"
        print(f"{verb} {count} files; {allocated / GIB:.3f} GiB allocated (estimate).")
        if count > PREVIEW_LINES:
            print(f"Only the first {PREVIEW_LINES} paths are shown, oldest first.")
        if changed or failures:
            print(
                f"Skipped changed/missing files: {changed}; removal errors: {failures}."
            )
        if target is not None:
            unmet = any(value < target for value in free.values())
            if unmet:
                print(
                    "Not enough eligible files to meet the free-space target.",
                    file=sys.stderr,
                )
                return 1
        return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"cleanup: {error}", file=sys.stderr)
        sys.exit(2)
