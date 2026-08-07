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
import time
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
    upstream_ref: str | None
    committer_name: str
    committer_email: str


@dataclass(frozen=True)
class ProbeResult:
    candidate_oid: str
    repository: Path


@dataclass(frozen=True)
class WipFingerprint:
    digest: str
    status_digest: str
    staged_digest: str
    unstaged_digest: str
    untracked_digest: str
    path_states: tuple[tuple[bytes, str], ...]


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


def resolve_state(
    target_branch: str,
    remote_override: str | None,
    known_remote_oid: str | None = None,
) -> SyncState:
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
    configured_remote = git(
        current_root, "config", "--get", f"branch.{target_branch}.remote"
    )
    remote = remote_override or configured_remote
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
    if known_remote_oid is None:
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
    else:
        remote_oid = known_remote_oid
    upstream_ref: str | None = None
    if remote == configured_remote:
        candidate_upstream = git(
            current_root,
            "for-each-ref",
            "--format=%(upstream)",
            f"refs/heads/{target_branch}",
        )
        if candidate_upstream.startswith("refs/remotes/"):
            upstream_ref = candidate_upstream
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
        upstream_ref,
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


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_path(path: Path) -> bytes:
    digest = hashlib.sha256()
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return b"missing"
    digest.update(metadata.st_mode.to_bytes(8, byteorder="big"))
    if path.is_symlink():
        digest.update(b"L")
        digest.update(os.fsencode(os.readlink(path)))
    elif path.is_file():
        digest.update(b"F")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    elif path.is_dir():
        digest.update(b"D")
        nested_head = run(
            path,
            "git",
            "rev-parse",
            "--verify",
            "HEAD",
            check=False,
        )
        if nested_head.returncode == 0:
            digest.update(nested_head.stdout)
    else:
        digest.update(b"O")
    return digest.digest()


def index_path_states(worktree: Path, paths: set[bytes]) -> dict[bytes, list[bytes]]:
    states: dict[bytes, list[bytes]] = {}
    pathspecs = tuple(f":(literal){os.fsdecode(path)}" for path in sorted(paths))
    for offset in range(0, len(pathspecs), 128):
        raw = git_bytes(
            worktree,
            "ls-files",
            "--stage",
            "-z",
            "--",
            *pathspecs[offset : offset + 128],
        )
        for record in (value for value in raw.split(b"\0") if value):
            metadata, separator, path = record.partition(b"\t")
            if not separator:
                raise SyncError("cannot parse staged index entry")
            states.setdefault(path, []).append(metadata)
    return states


def wip_fingerprint(worktree: Path) -> WipFingerprint:
    status = git_bytes(worktree, "status", "--porcelain=v1", "-z")
    staged = git_bytes(worktree, "diff", "--cached", "--binary", "--full-index")
    unstaged = git_bytes(worktree, "diff", "--binary", "--full-index")
    staged_paths = nul_paths(
        git_bytes(worktree, "diff", "--cached", "--name-only", "-z")
    )
    unstaged_paths = nul_paths(git_bytes(worktree, "diff", "--name-only", "-z"))
    untracked = sorted(
        nul_paths(
            git_bytes(worktree, "ls-files", "--others", "--exclude-standard", "-z")
        )
    )

    digest = hashlib.sha256()
    for value in (status, staged, unstaged):
        digest.update(len(value).to_bytes(8, byteorder="big"))
        digest.update(value)
    untracked_state = hashlib.sha256()
    for encoded in untracked:
        relative = Path(os.fsdecode(encoded))
        path_state = hash_path(worktree / relative)
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
        digest.update(path_state)
        untracked_state.update(len(encoded).to_bytes(8, byteorder="big"))
        untracked_state.update(encoded)
        untracked_state.update(path_state)

    untracked_paths = set(untracked)
    paths = staged_paths | unstaged_paths | untracked_paths
    index_states = index_path_states(worktree, paths)
    path_states: list[tuple[bytes, str]] = []
    for encoded in sorted(paths):
        state = hashlib.sha256()
        state.update(b"S" if encoded in staged_paths else b"-")
        state.update(b"U" if encoded in unstaged_paths else b"-")
        state.update(b"?" if encoded in untracked_paths else b"-")
        for entry in index_states.get(encoded, []):
            state.update(len(entry).to_bytes(8, byteorder="big"))
            state.update(entry)
        state.update(hash_path(worktree / Path(os.fsdecode(encoded))))
        path_states.append((encoded, state.hexdigest()))

    return WipFingerprint(
        digest.hexdigest(),
        hash_bytes(status),
        hash_bytes(staged),
        hash_bytes(unstaged),
        untracked_state.hexdigest(),
        tuple(path_states),
    )


def changed_wip_message(before: WipFingerprint, after: WipFingerprint) -> str:
    before_paths = dict(before.path_states)
    after_paths = dict(after.path_states)
    changed_paths = sorted(
        path
        for path in before_paths.keys() | after_paths.keys()
        if before_paths.get(path) != after_paths.get(path)
    )
    components = [
        name
        for name, old, new in (
            ("status", before.status_digest, after.status_digest),
            ("staged", before.staged_digest, after.staged_digest),
            ("unstaged", before.unstaged_digest, after.unstaged_digest),
            ("untracked", before.untracked_digest, after.untracked_digest),
        )
        if old != new
    ]
    lines = ["target WIP changed during isolated preflight"]
    if changed_paths:
        lines.append(
            "changed paths:\n"
            + "\n".join(f"  {os.fsdecode(path)}" for path in changed_paths)
        )
    if components:
        lines.append("changed components: " + ", ".join(components))
    return "\n".join(lines)


def nul_paths(raw: bytes) -> set[bytes]:
    return {value for value in raw.split(b"\0") if value}


def paths_collide(left: bytes, right: bytes) -> bool:
    return (
        left == right or left.startswith(right + b"/") or right.startswith(left + b"/")
    )


def operation_pathspecs(operation_paths: set[bytes]) -> tuple[str, ...]:
    return tuple(f":(literal){os.fsdecode(path)}" for path in sorted(operation_paths))


def blocking_parent_paths(worktree: Path, operation_paths: set[bytes]) -> set[bytes]:
    blockers: set[bytes] = set()
    for path in operation_paths:
        components = path.split(b"/")
        for end in range(1, len(components)):
            encoded = b"/".join(components[:end])
            candidate = worktree / Path(os.fsdecode(encoded))
            if candidate.is_symlink() or (
                candidate.exists() and not candidate.is_dir()
            ):
                blockers.add(encoded)
    return blockers


def ignored_paths(worktree: Path, operation_paths: set[bytes]) -> set[bytes]:
    paths: set[bytes] = set()
    pathspecs = operation_pathspecs(
        operation_paths | blocking_parent_paths(worktree, operation_paths)
    )
    for offset in range(0, len(pathspecs), 128):
        paths.update(
            nul_paths(
                git_bytes(
                    worktree,
                    "ls-files",
                    "--others",
                    "--ignored",
                    "--exclude-standard",
                    "-z",
                    "--",
                    *pathspecs[offset : offset + 128],
                )
            )
        )
    return paths


def ensure_ignored_wip_safe(
    worktree: Path,
    branch: str,
    operation_paths: set[bytes],
) -> None:
    collisions = sorted(
        ignored
        for ignored in ignored_paths(worktree, operation_paths)
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


def prepare_probe_repository(state: SyncState) -> Path:
    common_dir = git_common_dir(state.current_root)
    probe_root = common_dir / "loom-branch-sync-probe"
    if probe_root.is_symlink():
        raise SyncError(f"probe repository path must not be a symlink: {probe_root}")
    if not probe_root.exists():
        run(
            common_dir,
            "git",
            "clone",
            "--shared",
            "--no-checkout",
            "--quiet",
            str(state.current_root),
            str(probe_root),
        )
        git(probe_root, "config", "--local", "loom.branchSyncProbe", "true")
    else:
        if not probe_root.is_dir():
            raise SyncError(
                f"refusing to clean unrecognized probe repository: {probe_root}"
            )
        is_repository = run(
            probe_root,
            "git",
            "rev-parse",
            "--is-inside-work-tree",
            check=False,
        )
        marker = run(
            probe_root,
            "git",
            "config",
            "--local",
            "--get",
            "loom.branchSyncProbe",
            check=False,
        )
        if (
            is_repository.returncode != 0
            or is_repository.stdout.strip() != b"true"
            or marker.returncode != 0
            or marker.stdout.strip() != b"true"
        ):
            raise SyncError(
                f"refusing to clean unrecognized probe repository: {probe_root}"
            )
        run(probe_root, "git", "rebase", "--abort", check=False)
        run(probe_root, "git", "reset", "--hard", "--quiet")
        run(probe_root, "git", "clean", "-ffdx", "--quiet")
        run(probe_root, "git", "stash", "clear")
    git(probe_root, "config", "user.name", state.committer_name)
    git(probe_root, "config", "user.email", state.committer_email)
    git(probe_root, "config", "core.hooksPath", "/dev/null")
    git(probe_root, "config", "commit.gpgSign", "false")
    git(probe_root, "switch", "--detach", "--discard-changes", state.target_oid)
    return probe_root


def probe(state: SyncState) -> ProbeResult:
    probe_root = prepare_probe_repository(state)
    has_wip = reproduce_wip(state, probe_root)
    remote_present = run(
        probe_root,
        "git",
        "cat-file",
        "-e",
        f"{state.remote_oid}^{{commit}}",
        check=False,
    )
    if remote_present.returncode != 0:
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
    return ProbeResult(candidate_oid, probe_root)


def print_preflight(state: SyncState, result: ProbeResult) -> None:
    print("preflight: isolated dry-run succeeded")
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


def synchronize_local_worktrees(state: SyncState, candidate_oid: str) -> None:
    stash_oid = real_stash(state.target_worktree)
    apply_candidate_tree(
        state.target_worktree,
        state.target_branch,
        state.target_oid,
        candidate_oid,
        stash_oid,
    )
    apply_candidate_tree(
        state.current_root,
        state.current_branch,
        state.current_oid,
        candidate_oid,
        stash_oid,
    )
    transaction = (
        "start\n"
        f"update refs/heads/{state.target_branch} {candidate_oid} "
        f"{state.target_oid}\n"
        f"update refs/heads/{state.current_branch} {candidate_oid} "
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


def refresh_upstream_ref(state: SyncState) -> None:
    if state.upstream_ref is None:
        return
    refreshed = run(
        state.current_root,
        "git",
        "fetch",
        "--quiet",
        "--no-tags",
        "--no-write-fetch-head",
        "--recurse-submodules=no",
        state.remote,
        f"+refs/heads/{state.target_branch}:{state.upstream_ref}",
        check=False,
    )
    if refreshed.returncode != 0:
        details = (
            b"\n".join((refreshed.stdout, refreshed.stderr))
            .decode(errors="replace")
            .strip()
        )
        raise SyncError(
            "remote target was updated, but its local upstream ref was not refreshed\n"
            + details
        )


def execute(
    state: SyncState,
    fingerprint: WipFingerprint,
    result: ProbeResult,
    timings: dict[str, float],
) -> str:
    refreshed = resolve_state(state.target_branch, state.remote, state.remote_oid)
    if refreshed != state:
        raise SyncError("branch or remote state changed during isolated preflight")
    refreshed_fingerprint = wip_fingerprint(state.target_worktree)
    if refreshed_fingerprint.digest != fingerprint.digest:
        raise SyncError(changed_wip_message(fingerprint, refreshed_fingerprint))
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

    with record_timing(timings, "target-mutation"):
        synchronize_local_worktrees(state, result.candidate_oid)

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
    refresh_upstream_ref(state)
    return synchronized_oid


@contextmanager
def record_timing(timings: dict[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - started


def print_timings(timings: dict[str, float], total: float) -> None:
    order = ("resolve", "fingerprint", "preflight", "execute", "target-mutation")
    fields = [f"{name}={timings[name]:.3f}s" for name in order if name in timings]
    fields.append(f"total={total:.3f}s")
    print("timing: " + " ".join(fields))


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
    started = time.perf_counter()
    timings: dict[str, float] = {}
    try:
        current_root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
        with repository_lock(current_root):
            with record_timing(timings, "resolve"):
                state = resolve_state(args.target, args.remote)
            with record_timing(timings, "fingerprint"):
                fingerprint = wip_fingerprint(state.target_worktree)
            with record_timing(timings, "preflight"):
                result = probe(state)
            print_preflight(state, result)
            if args.dry_run:
                return 0
            with record_timing(timings, "execute"):
                synchronized_oid = execute(state, fingerprint, result, timings)
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
    finally:
        print_timings(timings, time.perf_counter() - started)


if __name__ == "__main__":
    sys.exit(main())
