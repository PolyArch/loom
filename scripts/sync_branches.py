#!/usr/bin/env python3
"""Rebase a target branch onto the current branch without losing target WIP."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class SyncError(RuntimeError):
    pass


@dataclass(frozen=True)
class Worktree:
    path: Path
    head: str
    branch: str | None


@dataclass(frozen=True)
class SyncState:
    current_root: Path
    current_branch: str
    current_oid: str
    target_worktree: Path
    target_branch: str
    target_oid: str
    base_oid: str
    remote: str
    push_url: str
    remote_oid: str
    committer_name: str
    committer_email: str


@dataclass(frozen=True)
class ProbeResult:
    candidate_oid: str
    repository: Path


@dataclass(frozen=True)
class WipFingerprint:
    digest: str


def run(
    cwd: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        command = " ".join(arguments)
        stderr = completed.stderr.decode(errors="replace").strip()
        raise SyncError(f"command failed ({completed.returncode}): {command}\n{stderr}")
    return completed


def git(cwd: Path, *arguments: str, check: bool = True) -> str:
    return run(cwd, "git", *arguments, check=check).stdout.decode().strip()


def git_bytes(cwd: Path, *arguments: str) -> bytes:
    return run(cwd, "git", *arguments).stdout


def parse_worktrees(raw: bytes) -> tuple[Worktree, ...]:
    records: list[Worktree] = []
    fields: dict[str, str] = {}
    for token in raw.split(b"\0"):
        if not token:
            if fields:
                records.append(
                    Worktree(
                        Path(fields["worktree"]),
                        fields["HEAD"],
                        fields.get("branch", "").removeprefix("refs/heads/") or None,
                    )
                )
                fields = {}
            continue
        key, _, value = token.partition(b" ")
        fields[key.decode()] = os.fsdecode(value)
    if fields:
        records.append(
            Worktree(
                Path(fields["worktree"]),
                fields["HEAD"],
                fields.get("branch", "").removeprefix("refs/heads/") or None,
            )
        )
    return tuple(records)


def resolve_state(target_branch: str, remote_override: str | None) -> SyncState:
    current_root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    current_branch = git(current_root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if current_branch == target_branch:
        raise SyncError("target branch must differ from the current branch")
    run(current_root, "git", "check-ref-format", "--branch", target_branch)
    run(
        current_root,
        "git",
        "show-ref",
        "--verify",
        f"refs/heads/{target_branch}",
    )
    if git_bytes(current_root, "status", "--porcelain=v1", "-z"):
        raise SyncError(f"current branch {current_branch} worktree is not clean")

    worktrees = parse_worktrees(
        git_bytes(current_root, "worktree", "list", "--porcelain", "-z")
    )
    matches = [worktree for worktree in worktrees if worktree.branch == target_branch]
    if len(matches) != 1:
        raise SyncError(
            f"target branch {target_branch} must be checked out in exactly one worktree"
        )
    target_worktree = matches[0].path.resolve()

    current_oid = git(current_root, "rev-parse", f"refs/heads/{current_branch}")
    target_oid = git(current_root, "rev-parse", f"refs/heads/{target_branch}")
    base_oid = git(current_root, "merge-base", current_oid, target_oid)
    remote = remote_override or git(
        current_root, "config", "--get", f"branch.{target_branch}.remote"
    )
    if not remote:
        raise SyncError(
            f"target branch {target_branch} has no configured remote; use --remote"
        )
    push_urls = git(
        current_root,
        "remote",
        "get-url",
        "--push",
        "--all",
        remote,
    ).splitlines()
    if len(push_urls) != 1:
        raise SyncError(
            f"remote {remote} must have exactly one push URL; found {len(push_urls)}"
        )
    push_url = push_urls[0]
    remote_line = git(
        current_root,
        "ls-remote",
        "--heads",
        push_url,
        f"refs/heads/{target_branch}",
    )
    if not remote_line:
        raise SyncError(f"remote branch {remote}/{target_branch} does not exist")
    remote_oid = remote_line.split()[0]
    committer_ident = git(target_worktree, "var", "GIT_COMMITTER_IDENT")
    match = re.fullmatch(r"(.*) <([^<>]+)> \d+ [+-]\d{4}", committer_ident)
    if match is None:
        raise SyncError(f"cannot parse Git committer identity: {committer_ident}")
    committer_name, committer_email = match.groups()
    return SyncState(
        current_root,
        current_branch,
        current_oid,
        target_worktree,
        target_branch,
        target_oid,
        base_oid,
        remote,
        push_url,
        remote_oid,
        committer_name,
        committer_email,
    )


def copy_untracked(source_root: Path, destination_root: Path) -> None:
    raw = git_bytes(source_root, "ls-files", "--others", "--exclude-standard", "-z")
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        source = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
        elif source.is_file():
            shutil.copy2(source, destination)
        else:
            raise SyncError(f"unsupported untracked entry: {relative}")


def wip_fingerprint(worktree: Path) -> WipFingerprint:
    digest = hashlib.sha256()
    for arguments in (
        ("status", "--porcelain=v1", "-z"),
        ("diff", "--cached", "--binary", "--full-index"),
        ("diff", "--binary", "--full-index"),
    ):
        value = git_bytes(worktree, *arguments)
        digest.update(len(value).to_bytes(8, byteorder="big"))
        digest.update(value)
    untracked = git_bytes(worktree, "ls-files", "--others", "--exclude-standard", "-z")
    for encoded in sorted(value for value in untracked.split(b"\0") if value):
        relative = Path(os.fsdecode(encoded))
        path = worktree / relative
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
        if path.is_symlink():
            content = os.fsencode(os.readlink(path))
            digest.update(b"L")
            digest.update(content)
        elif path.is_file():
            digest.update(b"F")
            digest.update(path.stat().st_mode.to_bytes(8, byteorder="big"))
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        else:
            raise SyncError(f"unsupported untracked entry: {relative}")
    return WipFingerprint(digest.hexdigest())


def nul_paths(raw: bytes) -> set[bytes]:
    return {value for value in raw.split(b"\0") if value}


def paths_collide(left: bytes, right: bytes) -> bool:
    return (
        left == right or left.startswith(right + b"/") or right.startswith(left + b"/")
    )


def ignored_paths(worktree: Path) -> set[bytes]:
    return nul_paths(
        git_bytes(
            worktree,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        )
    )


def ensure_ignored_wip_safe(
    worktree: Path,
    branch: str,
    operation_paths: set[bytes],
) -> None:
    collisions = sorted(
        ignored
        for ignored in ignored_paths(worktree)
        if any(paths_collide(ignored, affected) for affected in operation_paths)
    )
    if not collisions:
        return
    rendered = "\n".join(f"  {os.fsdecode(path)}" for path in collisions)
    raise SyncError(
        f"ignored WIP in {branch} would be overwritten by synchronized history:\n"
        f"{rendered}"
    )


@contextmanager
def repository_lock(current_root: Path) -> Iterator[None]:
    common_dir = git_common_dir(current_root)
    lock_path = common_dir / "loom-branch-sync.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SyncError("another branch synchronization is running") from error
        yield


def git_common_dir(current_root: Path) -> Path:
    common_dir = Path(git(current_root, "rev-parse", "--git-common-dir"))
    if not common_dir.is_absolute():
        common_dir = (current_root / common_dir).resolve()
    return common_dir


def reproduce_wip(state: SyncState, probe_root: Path) -> bool:
    staged = git_bytes(
        state.target_worktree, "diff", "--cached", "--binary", "--full-index"
    )
    unstaged = git_bytes(state.target_worktree, "diff", "--binary", "--full-index")
    if staged:
        run(probe_root, "git", "apply", "--binary", "--index", "-", input_bytes=staged)
    if unstaged:
        run(probe_root, "git", "apply", "--binary", "-", input_bytes=unstaged)
    copy_untracked(state.target_worktree, probe_root)
    has_wip = bool(git_bytes(probe_root, "status", "--porcelain=v1", "-z"))
    if has_wip:
        run(
            probe_root,
            "git",
            "stash",
            "push",
            "--include-untracked",
            "--message",
            "branch sync preflight",
        )
    return has_wip


def probe_conflict(
    probe_root: Path,
    category: str,
    completed: subprocess.CompletedProcess[bytes],
) -> SyncError:
    commit = ""
    if category == "commit-rebase":
        commit = git(
            probe_root,
            "show",
            "--no-patch",
            "--format=%H %s",
            "REBASE_HEAD",
            check=False,
        )
    paths = git(probe_root, "diff", "--name-only", "--diff-filter=U", check=False)
    stages = git(probe_root, "ls-files", "--unmerged", check=False)
    details = (
        b"\n".join((completed.stdout, completed.stderr))
        .decode(errors="replace")
        .strip()
    )
    lines = [f"{category} conflict detected in isolated preflight"]
    if commit:
        lines.append(f"commit: {commit}")
    if paths:
        lines.append("conflicting paths:\n" + paths)
    if stages:
        lines.append("index stages:\n" + stages)
    if details:
        lines.append("git details:\n" + details)
    lines.append("real branches, worktrees, stashes, and remote were not changed")
    return SyncError("\n".join(lines))


def push_target(
    cwd: Path,
    state: SyncState,
    candidate_oid: str,
    *,
    dry_run: bool,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    arguments = [
        "git",
        "-c",
        "push.followTags=false",
        "push",
        "--no-verify",
        "--no-follow-tags",
        "--recurse-submodules=no",
    ]
    if dry_run:
        arguments.append("--dry-run")
    arguments.extend(
        (
            f"--force-with-lease=refs/heads/{state.target_branch}:{state.remote_oid}",
            state.push_url,
            f"{candidate_oid}:refs/heads/{state.target_branch}",
        )
    )
    return run(cwd, *arguments, check=check)


@contextmanager
def probe(state: SyncState) -> Iterator[ProbeResult]:
    temp_parent = git_common_dir(state.current_root) / "loom-branch-sync-probes"
    temp_parent.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="probe-", dir=temp_parent) as raw:
            probe_root = Path(raw) / "repository"
            run(
                temp_parent,
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                "--quiet",
                str(state.current_root),
                str(probe_root),
            )
            git(probe_root, "config", "user.name", state.committer_name)
            git(probe_root, "config", "user.email", state.committer_email)
            git(probe_root, "switch", "--detach", state.target_oid)
            has_wip = reproduce_wip(state, probe_root)
            run(
                probe_root,
                "git",
                "fetch",
                "--quiet",
                state.push_url,
                f"refs/heads/{state.target_branch}",
            )
            ancestor = run(
                probe_root,
                "git",
                "merge-base",
                "--is-ancestor",
                state.remote_oid,
                state.target_oid,
                check=False,
            )
            if ancestor.returncode != 0:
                raise SyncError(
                    f"remote {state.remote}/{state.target_branch} contains commits "
                    "that are not in the local target branch"
                )
            rebased = run(
                probe_root,
                "git",
                "rebase",
                "--rebase-merges",
                "--reapply-cherry-picks",
                "--empty=keep",
                "--onto",
                state.current_oid,
                state.base_oid,
                check=False,
            )
            if rebased.returncode != 0:
                raise probe_conflict(probe_root, "commit-rebase", rebased)
            candidate_oid = git(probe_root, "rev-parse", "HEAD")
            target_changes = nul_paths(
                git_bytes(
                    probe_root,
                    "diff",
                    "--name-only",
                    "-z",
                    state.target_oid,
                    candidate_oid,
                )
            )
            ensure_ignored_wip_safe(
                state.target_worktree,
                state.target_branch,
                target_changes,
            )
            current_changes = nul_paths(
                git_bytes(
                    probe_root,
                    "diff",
                    "--name-only",
                    "-z",
                    state.current_oid,
                    candidate_oid,
                )
            )
            ensure_ignored_wip_safe(
                state.current_root,
                state.current_branch,
                current_changes,
            )
            if has_wip:
                restored = run(
                    probe_root,
                    "git",
                    "stash",
                    "pop",
                    "--index",
                    check=False,
                )
                if restored.returncode != 0:
                    raise probe_conflict(probe_root, "wip-restore", restored)
            push_target(probe_root, state, candidate_oid, dry_run=True)
            yield ProbeResult(candidate_oid, probe_root)
    finally:
        try:
            temp_parent.rmdir()
        except OSError:
            pass


def print_dry_run(state: SyncState, result: ProbeResult) -> None:
    print("dry-run: branch synchronization preflight succeeded")
    print(f"base:    {state.base_oid}")
    print(
        f"current: {state.current_branch} {state.current_oid} -> {result.candidate_oid}"
    )
    print(
        f"target:  {state.target_branch} {state.target_oid} -> {result.candidate_oid}"
    )
    print(
        f"remote:  {state.remote}/{state.target_branch} {state.remote_oid} "
        f"-> {result.candidate_oid} (not updated)"
    )


def real_stash(target_worktree: Path) -> str | None:
    if not git_bytes(target_worktree, "status", "--porcelain=v1", "-z"):
        return None
    run(
        target_worktree,
        "git",
        "stash",
        "push",
        "--include-untracked",
        "--message",
        "branch sync WIP",
    )
    stash_oid = git(target_worktree, "rev-parse", "refs/stash")
    if git_bytes(target_worktree, "status", "--porcelain=v1", "-z"):
        raise SyncError(
            "target worktree changed while its WIP was being stashed; "
            f"stash {stash_oid} was preserved"
        )
    return stash_oid


def restore_real_stash(target_worktree: Path, stash_oid: str | None) -> None:
    if stash_oid is None:
        return
    current_stash = git(target_worktree, "rev-parse", "refs/stash")
    if current_stash != stash_oid:
        raise SyncError(
            f"target stash list changed concurrently; WIP stash {stash_oid} was preserved"
        )
    restored = run(
        target_worktree,
        "git",
        "stash",
        "pop",
        "--index",
        "stash@{0}",
        check=False,
    )
    if restored.returncode != 0:
        details = (
            b"\n".join((restored.stdout, restored.stderr))
            .decode(errors="replace")
            .strip()
        )
        raise SyncError(
            "unexpected wip-restore conflict after successful preflight; "
            f"stash {stash_oid} was preserved\n{details}"
        )


def apply_candidate_tree(
    worktree: Path,
    branch: str,
    old_oid: str,
    candidate_oid: str,
    stash_oid: str | None,
) -> None:
    tree_patch = git_bytes(
        worktree,
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        old_oid,
        candidate_oid,
    )
    if tree_patch:
        applied = run(
            worktree,
            "git",
            "apply",
            "--binary",
            "--index",
            "-",
            input_bytes=tree_patch,
            check=False,
        )
        if applied.returncode != 0:
            details = (
                b"\n".join((applied.stdout, applied.stderr))
                .decode(errors="replace")
                .strip()
            )
            stash_note = f"; stash {stash_oid} was preserved" if stash_oid else ""
            raise SyncError(
                f"{branch} worktree changed after preflight; local branches were not "
                f"updated{stash_note}\n{details}"
            )
    index_tree = git(worktree, "write-tree")
    candidate_tree = git(worktree, "rev-parse", f"{candidate_oid}^{{tree}}")
    if index_tree != candidate_tree:
        stash_note = f"; stash {stash_oid} was preserved" if stash_oid else ""
        raise SyncError(
            f"candidate tree was not reproduced exactly in {branch}; local branches "
            f"were not updated{stash_note}"
        )


def execute(
    state: SyncState,
    fingerprint: WipFingerprint,
    result: ProbeResult,
) -> str:
    refreshed = resolve_state(state.target_branch, state.remote)
    if refreshed != state:
        raise SyncError("branch or remote state changed during isolated preflight")
    if wip_fingerprint(state.target_worktree) != fingerprint:
        raise SyncError("target WIP changed during isolated preflight")
    run(
        state.current_root,
        "git",
        "fetch",
        "--quiet",
        "--no-tags",
        "--no-write-fetch-head",
        str(result.repository),
        result.candidate_oid,
    )
    run(
        state.current_root,
        "git",
        "cat-file",
        "-e",
        f"{result.candidate_oid}^{{commit}}",
    )
    target_changes = nul_paths(
        git_bytes(
            state.current_root,
            "diff",
            "--name-only",
            "-z",
            state.target_oid,
            result.candidate_oid,
        )
    )
    ensure_ignored_wip_safe(
        state.target_worktree,
        state.target_branch,
        target_changes,
    )
    current_changes = nul_paths(
        git_bytes(
            state.current_root,
            "diff",
            "--name-only",
            "-z",
            state.current_oid,
            result.candidate_oid,
        )
    )
    ensure_ignored_wip_safe(
        state.current_root,
        state.current_branch,
        current_changes,
    )

    if git(state.current_root, "symbolic-ref", "--quiet", "--short", "HEAD") != (
        state.current_branch
    ):
        raise SyncError("current worktree changed branches during synchronization")
    if git_bytes(state.current_root, "status", "--porcelain=v1", "-z"):
        raise SyncError("current worktree became dirty during synchronization")
    if git(state.current_root, "rev-parse", "HEAD") != state.current_oid:
        raise SyncError("current branch advanced during synchronization")

    stash_oid = real_stash(state.target_worktree)
    apply_candidate_tree(
        state.target_worktree,
        state.target_branch,
        state.target_oid,
        result.candidate_oid,
        stash_oid,
    )
    apply_candidate_tree(
        state.current_root,
        state.current_branch,
        state.current_oid,
        result.candidate_oid,
        stash_oid,
    )
    transaction = (
        "start\n"
        f"update refs/heads/{state.target_branch} {result.candidate_oid} "
        f"{state.target_oid}\n"
        f"update refs/heads/{state.current_branch} {result.candidate_oid} "
        f"{state.current_oid}\n"
        "prepare\n"
        "commit\n"
    ).encode()
    updated = run(
        state.current_root,
        "git",
        "update-ref",
        "--stdin",
        "-m",
        "synchronize branches",
        input_bytes=transaction,
        check=False,
    )
    if updated.returncode != 0:
        details = (
            b"\n".join((updated.stdout, updated.stderr))
            .decode(errors="replace")
            .strip()
        )
        raise SyncError(
            "a local branch changed before the ref transaction; it was not overwritten "
            f"and stash {stash_oid or 'none'} was preserved\n{details}"
        )
    restore_real_stash(state.target_worktree, stash_oid)

    target_oid = git(
        state.current_root, "rev-parse", f"refs/heads/{state.target_branch}"
    )
    if target_oid != result.candidate_oid:
        raise SyncError("target branch changed after applying the preflight candidate")
    synchronized_oid = git(state.current_root, "rev-parse", "HEAD")
    target_oid = git(
        state.current_root, "rev-parse", f"refs/heads/{state.target_branch}"
    )
    if target_oid != synchronized_oid or synchronized_oid != result.candidate_oid:
        raise SyncError("local branches did not converge after fast-forward")

    pushed = push_target(
        state.current_root,
        state,
        synchronized_oid,
        dry_run=False,
        check=False,
    )
    if pushed.returncode != 0:
        details = (
            b"\n".join((pushed.stdout, pushed.stderr)).decode(errors="replace").strip()
        )
        actual_line = git(
            state.current_root,
            "ls-remote",
            "--heads",
            state.push_url,
            f"refs/heads/{state.target_branch}",
            check=False,
        )
        actual_remote_oid = actual_line.split()[0] if actual_line else "missing"
        if actual_remote_oid != state.remote_oid:
            raise SyncError(
                "force-with-lease rejected a concurrent remote update; "
                "local branches are synchronized\n"
                f"expected remote: {state.remote_oid}\n"
                f"actual remote:   {actual_remote_oid}\n" + details
            )
        raise SyncError(
            "remote target was not updated; local branches are synchronized\n" + details
        )
    remote_line = git(
        state.current_root,
        "ls-remote",
        "--heads",
        state.push_url,
        f"refs/heads/{state.target_branch}",
    )
    if not remote_line or remote_line.split()[0] != synchronized_oid:
        raise SyncError("remote target does not match the synchronized commit")
    return synchronized_oid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", help="target branch to rebase onto the current branch"
    )
    parser.add_argument(
        "--remote", help="target remote (defaults to target tracking remote)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the isolated preflight without changing local or remote refs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        current_root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
        with repository_lock(current_root):
            state = resolve_state(args.target, args.remote)
            fingerprint = wip_fingerprint(state.target_worktree)
            with probe(state) as result:
                if args.dry_run:
                    print_dry_run(state, result)
                    return 0
                synchronized_oid = execute(state, fingerprint, result)
            print(
                f"synchronized local {state.current_branch} and "
                f"{state.target_branch} at {synchronized_oid}"
            )
            print(
                f"updated remote {state.remote}/{state.target_branch} "
                "with an exact force-with-lease"
            )
            return 0
    except SyncError as error:
        print(f"warning: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
