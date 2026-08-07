#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sync_branches.py"
TEMP_ROOT = REPO_ROOT / "temp"


def run(
    cwd: Path,
    *arguments: str,
    check: bool = True,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=False,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            **(env or {}),
        },
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def git(cwd: Path, *arguments: str, check: bool = True) -> str:
    return run(cwd, "git", *arguments, check=check).stdout.strip()


class RepositoryFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.remote = root / "remote.git"
        self.target = root / "target"
        self.current = root / "current"

        run(root, "git", "init", "--bare", str(self.remote))
        run(root, "git", "init", "--initial-branch=A", str(self.target))
        git(self.target, "config", "user.name", "Branch Sync Test")
        git(self.target, "config", "user.email", "branch-sync@example.com")
        git(self.target, "config", "commit.gpgSign", "false")
        git(self.target, "config", "core.hooksPath", "/dev/null")
        git(self.target, "remote", "add", "origin", str(self.remote))

        self.write(self.target / "base.txt", "base\n")
        git(self.target, "add", "base.txt")
        git(self.target, "commit", "-m", "Base")
        git(self.target, "push", "-u", "origin", "A")

        git(self.target, "branch", "B")
        git(self.target, "worktree", "add", str(self.current), "B")
        git(self.current, "config", "user.name", "Branch Sync Test")
        git(self.current, "config", "user.email", "branch-sync@example.com")

    @staticmethod
    def write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def commit_current(self, path: str, content: str, message: str) -> None:
        self.write(self.current / path, content)
        git(self.current, "add", path)
        git(self.current, "commit", "-m", message)

    def commit_target(self, path: str, content: str, message: str) -> None:
        self.write(self.target / path, content)
        git(self.target, "add", path)
        git(self.target, "commit", "-m", message)

    def push_branches(self) -> None:
        git(self.target, "push", "origin", "A")
        git(self.current, "push", "-u", "origin", "B")

    def remote_oid(self, branch: str) -> str:
        output = git(
            self.target,
            "ls-remote",
            "--heads",
            "origin",
            f"refs/heads/{branch}",
        )
        return output.split()[0]

    def probe_repository(self) -> Path:
        common_dir = Path(git(self.current, "rev-parse", "--git-common-dir"))
        if not common_dir.is_absolute():
            common_dir = (self.current / common_dir).resolve()
        return common_dir / "loom-branch-sync-probe"

    def invoke(
        self,
        *arguments: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return run(
            self.current,
            "python3",
            str(SCRIPT),
            *arguments,
            check=False,
            env={**os.environ, **(env or {})},
        )


class SyncBranchesTest(unittest.TestCase):
    def setUp(self) -> None:
        TEMP_ROOT.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="sync-branches-test-", dir=TEMP_ROOT
        )
        self.fixture = RepositoryFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dry_run_preserves_refs_remote_and_wip(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()

        fixture.write(fixture.target / "existing-stash.txt", "existing stash\n")
        git(
            fixture.target,
            "stash",
            "push",
            "--include-untracked",
            "--message",
            "Existing stash",
        )
        fixture.write(fixture.target / "staged.txt", "staged WIP\n")
        git(fixture.target, "add", "staged.txt")
        fixture.write(fixture.target / "target.txt", "target WIP\n")
        fixture.write(fixture.target / "untracked.txt", "untracked WIP\n")

        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")
        status_before = git(fixture.target, "status", "--porcelain=v1")
        stash_before = git(fixture.target, "stash", "list", "--format=%H %s")

        completed = fixture.invoke("A", "--dry-run")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("dry-run", completed.stdout.lower())
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)
        self.assertEqual(git(fixture.target, "status", "--porcelain=v1"), status_before)
        self.assertEqual(
            git(fixture.target, "stash", "list", "--format=%H %s"),
            stash_before,
        )
        self.assertEqual((fixture.target / "staged.txt").read_text(), "staged WIP\n")
        self.assertEqual((fixture.target / "target.txt").read_text(), "target WIP\n")
        self.assertEqual(
            (fixture.target / "untracked.txt").read_text(), "untracked WIP\n"
        )
        self.assertFalse((fixture.current / "temp").exists())

    def test_dry_run_reuses_a_marked_probe_repository(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()

        first = fixture.invoke("A", "--dry-run")

        self.assertEqual(first.returncode, 0, first.stderr)
        probe = fixture.probe_repository()
        self.assertTrue(probe.is_dir())
        self.assertEqual(git(probe, "config", "--get", "loom.branchSyncProbe"), "true")
        git_dir_inode = (probe / ".git").stat().st_ino

        second = fixture.invoke("A", "--dry-run")

        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual((probe / ".git").stat().st_ino, git_dir_inode)

    def test_unmarked_probe_directory_is_never_cleaned(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        probe = fixture.probe_repository()
        probe.mkdir()
        sentinel = probe / "sentinel.txt"
        fixture.write(sentinel, "preserve me\n")

        completed = fixture.invoke("A", "--dry-run")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unrecognized probe repository", completed.stderr)
        self.assertEqual(sentinel.read_text(), "preserve me\n")

    def test_dry_run_reports_phase_and_total_timings(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()

        completed = fixture.invoke("A", "--dry-run")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertRegex(
            completed.stdout,
            re.compile(
                r"^timing: resolve=\d+\.\d{3}s fingerprint=\d+\.\d{3}s "
                r"preflight=\d+\.\d{3}s total=\d+\.\d{3}s$",
                re.MULTILINE,
            ),
        )

    def test_dry_run_skips_fetch_when_remote_commit_is_already_available(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        wrapper = self.git_wrapper(
            "fetch",
            "exit 73",
            require_inject_worktree=False,
        )

        completed = fixture.invoke(
            "A",
            "--dry-run",
            env={"PATH": f"{wrapper.parent}:{os.environ['PATH']}"},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_ignored_scan_is_scoped_to_synchronized_paths(self) -> None:
        fixture = self.fixture
        fixture.commit_current("nested/current.txt", "from B\n", "Current change")
        fixture.commit_target("nested/target.txt", "from A\n", "Target change")
        fixture.push_branches()
        wrapper = self.git_wrapper(
            "ls-files",
            "found_separator=; found_path=; "
            'for argument in "$@"; do '
            "  if [ \"$argument\" = ':(literal)nested' ]; then exit 75; fi; "
            '  if [ "$found_separator" = yes ]; then found_path=yes; break; fi; '
            '  if [ "$argument" = -- ]; then found_separator=yes; fi; '
            "done; "
            'if [ "$found_path" != yes ]; then exit 74; fi',
            condition='[ "$1" = "ls-files" ] && '
            'case " $* " in *" --ignored "*) true;; *) false;; esac',
            require_inject_worktree=False,
        )

        completed = fixture.invoke(
            "A",
            "--dry-run",
            env={"PATH": f"{wrapper.parent}:{os.environ['PATH']}"},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_dry_run_reports_commit_conflict_without_changing_real_repo(self) -> None:
        fixture = self.fixture
        fixture.commit_current("base.txt", "from B\n", "Conflicting current change")
        fixture.commit_target("base.txt", "from A\n", "Conflicting target change")
        fixture.push_branches()

        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")

        completed = fixture.invoke("A", "--dry-run")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("warning:", completed.stderr.lower())
        self.assertIn("commit-rebase", completed.stderr)
        self.assertIn("base.txt", completed.stderr)
        self.assertIn("Conflicting target change", completed.stderr)
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)
        self.assertEqual(git(fixture.target, "status", "--porcelain=v1"), "")

    def test_dry_run_reports_wip_restore_conflict_without_stashing_real_wip(
        self,
    ) -> None:
        fixture = self.fixture
        fixture.commit_current("base.txt", "from B\n", "Current base change")
        fixture.commit_target("target.txt", "from A\n", "Target side change")
        fixture.push_branches()
        fixture.write(fixture.target / "base.txt", "target WIP\n")

        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")
        status_before = git(fixture.target, "status", "--porcelain=v1")

        completed = fixture.invoke("A", "--dry-run")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("warning:", completed.stderr.lower())
        self.assertIn("wip-restore", completed.stderr)
        self.assertIn("base.txt", completed.stderr)
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)
        self.assertEqual(git(fixture.target, "status", "--porcelain=v1"), status_before)
        self.assertEqual((fixture.target / "base.txt").read_text(), "target WIP\n")

    def test_sync_rebases_target_fast_forwards_current_and_restores_wip(
        self,
    ) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()

        original_target = git(fixture.target, "rev-parse", "A")
        remote_current = fixture.remote_oid("B")
        fixture.write(fixture.target / "existing-stash.txt", "existing stash\n")
        git(
            fixture.target,
            "stash",
            "push",
            "--include-untracked",
            "--message",
            "Existing stash",
        )
        stash_before = git(fixture.target, "stash", "list", "--format=%H %s")
        fixture.write(fixture.target / "staged.txt", "staged version\n")
        git(fixture.target, "add", "staged.txt")
        fixture.write(fixture.target / "staged.txt", "worktree version\n")
        fixture.write(fixture.target / "untracked.txt", "untracked WIP\n")

        completed = fixture.invoke("A")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("synchronized", completed.stdout.lower())
        target = git(fixture.target, "rev-parse", "A")
        current = git(fixture.current, "rev-parse", "B")
        self.assertEqual(target, current)
        self.assertNotEqual(target, original_target)
        self.assertEqual(fixture.remote_oid("A"), target)
        self.assertEqual(fixture.remote_oid("B"), remote_current)
        base = git(fixture.target, "merge-base", "A~2", "A")
        self.assertEqual(
            git(
                fixture.target,
                "log",
                "--reverse",
                "--format=%s",
                f"{base}..A",
            ).splitlines(),
            ["Current change", "Target change"],
        )
        self.assertEqual(git(fixture.target, "show", ":staged.txt"), "staged version")
        self.assertEqual(
            (fixture.target / "staged.txt").read_text(), "worktree version\n"
        )
        self.assertEqual(
            (fixture.target / "untracked.txt").read_text(), "untracked WIP\n"
        )
        self.assertEqual(
            git(fixture.target, "status", "--porcelain=v1").splitlines(),
            ["AM staged.txt", "?? untracked.txt"],
        )
        self.assertEqual(
            git(fixture.target, "stash", "list", "--format=%H %s"),
            stash_before,
        )
        self.assertEqual(git(fixture.current, "status", "--porcelain=v1"), "")
        self.assertEqual((fixture.current / "target.txt").read_text(), "from A\n")

    def test_actual_sync_queries_remote_oid_only_once(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        call_log = fixture.root / "ls-remote.log"
        wrapper = self.git_wrapper(
            "ls-remote",
            "printf 'call\\n' >> \"$CALL_LOG\"",
            require_inject_worktree=False,
        )

        completed = fixture.invoke(
            "A",
            env={
                "PATH": f"{wrapper.parent}:{os.environ['PATH']}",
                "CALL_LOG": str(call_log),
            },
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(call_log.read_text().splitlines(), ["call"])

    def test_exact_lease_refuses_concurrent_remote_update(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        expected_remote = fixture.remote_oid("A")
        fixture.write(fixture.target / "target.txt", "target WIP\n")
        fixture.write(fixture.target / "untracked.txt", "untracked WIP\n")

        attacker = fixture.root / "attacker"
        run(
            fixture.root,
            "git",
            "clone",
            "--quiet",
            "--branch",
            "A",
            str(fixture.remote),
            str(attacker),
        )
        git(attacker, "config", "user.name", "Remote Writer")
        git(attacker, "config", "user.email", "remote-writer@example.com")
        fixture.write(attacker / "remote.txt", "concurrent remote change\n")
        git(attacker, "add", "remote.txt")
        git(attacker, "commit", "-m", "Concurrent remote change")
        attacker_oid = git(attacker, "rev-parse", "HEAD")

        real_git = shutil.which("git")
        self.assertIsNotNone(real_git)
        wrapper = self.git_wrapper(
            "push",
            "unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX; "
            f"{shlex.quote(real_git or 'git')} -c core.hooksPath=/dev/null "
            f"-C {shlex.quote(str(attacker))} push --quiet origin HEAD:A",
            condition='[ "$1" = "-c" ] && [ "$3" = "push" ]',
        )

        completed = fixture.invoke(
            "A",
            env={
                "PATH": f"{wrapper.parent}:{os.environ['PATH']}",
                "INJECT_WORKTREE": str(fixture.current),
            },
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("force-with-lease rejected", completed.stderr)
        self.assertIn(expected_remote, completed.stderr)
        self.assertIn(attacker_oid, completed.stderr)
        self.assertEqual(fixture.remote_oid("A"), attacker_oid)
        self.assertEqual(
            git(fixture.target, "rev-parse", "A"),
            git(fixture.current, "rev-parse", "B"),
        )
        self.assertEqual((fixture.target / "target.txt").read_text(), "target WIP\n")
        self.assertEqual(
            (fixture.target / "untracked.txt").read_text(), "untracked WIP\n"
        )

    def test_dirty_current_worktree_is_rejected_before_preflight(self) -> None:
        fixture = self.fixture
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        fixture.write(fixture.current / "dirty.txt", "dirty current worktree\n")

        completed = fixture.invoke("A", "--dry-run")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("current branch B worktree is not clean", completed.stderr)
        self.assertEqual(list(fixture.current.glob("temp/branch-sync-probe-*")), [])

    def test_remote_only_target_commit_is_never_overwritten(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()

        remote_writer = fixture.root / "remote-writer"
        run(
            fixture.root,
            "git",
            "clone",
            "--quiet",
            "--branch",
            "A",
            str(fixture.remote),
            str(remote_writer),
        )
        git(remote_writer, "config", "user.name", "Remote Writer")
        git(remote_writer, "config", "user.email", "remote-writer@example.com")
        fixture.write(remote_writer / "remote-only.txt", "remote only\n")
        git(remote_writer, "add", "remote-only.txt")
        git(remote_writer, "commit", "-m", "Remote only change")
        git(remote_writer, "push", "origin", "A")
        remote_oid = fixture.remote_oid("A")

        completed = fixture.invoke("A", "--dry-run")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "contains commits that are not in the local target", completed.stderr
        )
        self.assertEqual(fixture.remote_oid("A"), remote_oid)

    def test_ignored_wip_collision_is_rejected_without_changing_real_repo(
        self,
    ) -> None:
        fixture = self.fixture
        fixture.commit_target(".gitignore", "ignored.txt\n", "Ignore local data")
        git(fixture.target, "merge", "--ff-only", "A")
        fixture.write(fixture.target / "ignored.txt", "precious target WIP\n")
        fixture.write(fixture.current / "ignored.txt", "tracked by current branch\n")
        git(fixture.current, "add", "--force", "ignored.txt")
        git(fixture.current, "commit", "-m", "Track ignored path")
        fixture.push_branches()

        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")

        completed = fixture.invoke("A")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ignored WIP", completed.stderr)
        self.assertIn("ignored.txt", completed.stderr)
        self.assertEqual(
            (fixture.target / "ignored.txt").read_text(), "precious target WIP\n"
        )
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)

    def test_ignored_wip_in_current_worktree_blocks_fast_forward(self) -> None:
        fixture = self.fixture
        fixture.commit_current(".gitignore", "target-added.txt\n", "Ignore local data")
        fixture.write(fixture.current / "target-added.txt", "precious current WIP\n")
        fixture.commit_target(
            "target-added.txt", "tracked by target branch\n", "Add target data"
        )
        fixture.push_branches()

        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")

        completed = fixture.invoke("A")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ignored WIP in B", completed.stderr)
        self.assertIn("target-added.txt", completed.stderr)
        self.assertEqual(
            (fixture.current / "target-added.txt").read_text(),
            "precious current WIP\n",
        )
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)

    def test_ignored_file_blocking_a_candidate_directory_is_rejected(self) -> None:
        fixture = self.fixture
        fixture.commit_target(".gitignore", "blocked\n", "Ignore local data")
        fixture.write(fixture.target / "blocked", "precious target WIP\n")
        fixture.commit_current(
            "blocked/child.txt", "tracked by current branch\n", "Add nested data"
        )
        fixture.push_branches()
        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")

        completed = fixture.invoke("A", "--dry-run")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ignored WIP", completed.stderr)
        self.assertIn("blocked", completed.stderr)
        self.assertEqual(
            (fixture.target / "blocked").read_text(), "precious target WIP\n"
        )
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)

    def test_target_ancestor_fast_forwards_without_inventing_commits(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.push_branches()
        fixture.write(fixture.target / "untracked.txt", "target WIP\n")

        completed = fixture.invoke("A")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        target = git(fixture.target, "rev-parse", "A")
        self.assertEqual(target, git(fixture.current, "rev-parse", "B"))
        self.assertEqual(target, fixture.remote_oid("A"))
        self.assertEqual(
            git(fixture.target, "show", "-s", "--format=%s", "A"), "Current change"
        )
        self.assertEqual((fixture.target / "untracked.txt").read_text(), "target WIP\n")

    def test_actual_sync_uses_the_exact_preflight_candidate(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()

        hooks = fixture.root / "rewrite-hooks"
        hooks.mkdir()
        hook = hooks / "post-rewrite"
        hook.write_text(
            "#!/bin/sh\n"
            "printf 'hook mutation\\n' > hook-mutation.txt\n"
            "git add hook-mutation.txt\n"
            "git commit --no-verify -m 'Hook mutation' >/dev/null\n"
        )
        hook.chmod(0o755)
        git(fixture.target, "config", "core.hooksPath", str(hooks))

        completed = fixture.invoke("A")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        subjects = git(fixture.target, "log", "--format=%s", "-3", "A").splitlines()
        self.assertEqual(subjects, ["Target change", "Current change", "Base"])
        self.assertFalse((fixture.target / "hook-mutation.txt").exists())
        self.assertEqual(fixture.remote_oid("A"), git(fixture.target, "rev-parse", "A"))

    def test_push_does_not_follow_annotated_tags(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        git(fixture.current, "tag", "-a", "local-release", "-m", "Local release")
        git(fixture.current, "config", "push.followTags", "true")

        completed = fixture.invoke("A")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        remote_tags = git(
            fixture.current,
            "ls-remote",
            "--tags",
            str(fixture.remote),
            "refs/tags/local-release",
        )
        self.assertEqual(remote_tags, "")

    def test_push_does_not_run_repository_pre_push_hooks(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        remote_current = fixture.remote_oid("B")
        marker = fixture.root / "pre-push-ran"
        hooks = fixture.root / "push-hooks"
        hooks.mkdir()
        hook = hooks / "pre-push"
        hook.write_text(
            "#!/bin/sh\n"
            "unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX\n"
            f"touch {shlex.quote(str(marker))}\n"
            "git -c core.hooksPath=/dev/null push --quiet origin B\n"
        )
        hook.chmod(0o755)
        git(fixture.current, "config", "core.hooksPath", str(hooks))

        completed = fixture.invoke("A")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(marker.exists())
        self.assertEqual(fixture.remote_oid("B"), remote_current)

    def test_preflight_and_push_use_the_same_push_url(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()

        push_remote = fixture.root / "push-remote.git"
        run(fixture.root, "git", "init", "--bare", str(push_remote))
        git(fixture.target, "push", str(push_remote), "A", "B")
        original_remote_a = fixture.remote_oid("A")
        git(fixture.target, "config", "remote.origin.pushurl", str(push_remote))

        completed = fixture.invoke("A")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        synchronized = git(fixture.target, "rev-parse", "A")
        pushed = git(
            fixture.target,
            "ls-remote",
            "--heads",
            str(push_remote),
            "refs/heads/A",
        ).split()[0]
        self.assertEqual(pushed, synchronized)
        self.assertEqual(fixture.remote_oid("A"), original_remote_a)

    def test_multiple_push_urls_are_rejected_before_preflight(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        git(
            fixture.target,
            "config",
            "--add",
            "remote.origin.pushurl",
            str(fixture.remote),
        )
        git(
            fixture.target,
            "config",
            "--add",
            "remote.origin.pushurl",
            str(fixture.root / "other.git"),
        )

        completed = fixture.invoke("A", "--dry-run")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("exactly one push URL", completed.stderr)
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)

    def test_patch_equivalent_target_commit_is_preserved(self) -> None:
        fixture = self.fixture
        fixture.commit_current("equivalent.txt", "same patch\n", "Current equivalent")
        fixture.commit_target("equivalent.txt", "same patch\n", "Target equivalent")
        fixture.push_branches()

        completed = fixture.invoke("A")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        subjects = git(
            fixture.target,
            "log",
            "--reverse",
            "--format=%s",
            "A~2..A",
        ).splitlines()
        self.assertEqual(subjects, ["Current equivalent", "Target equivalent"])

    def test_candidate_uses_repository_committer_identity(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()

        completed = fixture.invoke("A")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        committer = git(fixture.target, "show", "-s", "--format=%cn%n%ce", "A")
        self.assertEqual(
            committer.splitlines(), ["Branch Sync Test", "branch-sync@example.com"]
        )

    def test_push_never_recurses_into_submodules(self) -> None:
        fixture = self.fixture
        submodule_remote = fixture.root / "submodule.git"
        run(fixture.root, "git", "init", "--bare", str(submodule_remote))
        submodule_seed = fixture.root / "submodule-seed"
        run(
            fixture.root,
            "git",
            "clone",
            "--quiet",
            str(submodule_remote),
            str(submodule_seed),
        )
        git(submodule_seed, "config", "user.name", "Submodule Test")
        git(submodule_seed, "config", "user.email", "submodule@example.com")
        git(submodule_seed, "switch", "-c", "main")
        fixture.write(submodule_seed / "data.txt", "published\n")
        git(submodule_seed, "add", "data.txt")
        git(submodule_seed, "commit", "-m", "Published submodule commit")
        git(submodule_seed, "push", "-u", "origin", "main")
        published = git(submodule_seed, "rev-parse", "HEAD")

        run(
            fixture.target,
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(submodule_remote),
            "submodule",
        )
        git(fixture.target / "submodule", "config", "user.name", "Submodule Test")
        git(
            fixture.target / "submodule",
            "config",
            "user.email",
            "submodule@example.com",
        )
        fixture.write(fixture.target / "submodule" / "data.txt", "unpublished\n")
        git(fixture.target / "submodule", "add", "data.txt")
        git(
            fixture.target / "submodule", "commit", "-m", "Unpublished submodule commit"
        )
        unpublished = git(fixture.target / "submodule", "rev-parse", "HEAD")
        git(fixture.target, "add", ".gitmodules", "submodule")
        git(fixture.target, "commit", "-m", "Add submodule update")
        fixture.commit_current("current.txt", "from B\n", "Current change")
        git(fixture.current, "push", "-u", "origin", "B")
        git(fixture.target, "config", "push.recurseSubmodules", "on-demand")

        completed = fixture.invoke("A")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        remote_head = git(
            fixture.target,
            "ls-remote",
            "--heads",
            str(submodule_remote),
            "refs/heads/main",
        ).split()[0]
        self.assertEqual(remote_head, published)
        self.assertNotEqual(remote_head, unpublished)

    def test_late_untracked_wip_is_not_overwritten(self) -> None:
        fixture = self.fixture
        fixture.commit_current("late.txt", "candidate content\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")
        wrapper = self.git_wrapper(
            "apply",
            "printf 'late WIP\\n' > \"$INJECT_PATH\"",
        )

        completed = fixture.invoke(
            "A",
            env={
                "PATH": f"{wrapper.parent}:{os.environ['PATH']}",
                "INJECT_WORKTREE": str(fixture.target),
                "INJECT_PATH": str(fixture.target / "late.txt"),
            },
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual((fixture.target / "late.txt").read_text(), "late WIP\n")
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)

    def test_concurrent_wip_change_reports_the_changed_path(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        fixture.write(fixture.target / "target.txt", "initial WIP\n")
        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")
        wrapper = self.git_wrapper(
            "push",
            "printf 'concurrent WIP\\n' > \"$INJECT_PATH\"",
            condition='[ "$1" = "-c" ] && [ "$3" = "push" ] && '
            'case " $* " in *" --dry-run "*) true;; *) false;; esac',
            require_inject_worktree=False,
        )

        completed = fixture.invoke(
            "A",
            env={
                "PATH": f"{wrapper.parent}:{os.environ['PATH']}",
                "INJECT_PATH": str(fixture.target / "target.txt"),
            },
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("target WIP changed during isolated preflight", completed.stderr)
        self.assertIn("target.txt", completed.stderr)
        self.assertEqual(
            (fixture.target / "target.txt").read_text(), "concurrent WIP\n"
        )
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)

    def test_late_ignored_wip_is_not_overwritten(self) -> None:
        fixture = self.fixture
        fixture.commit_target(".gitignore", "late.txt\n", "Ignore local data")
        fixture.commit_current("late.txt", "candidate content\n", "Current change")
        fixture.push_branches()
        target_before = git(fixture.target, "rev-parse", "A")
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")
        wrapper = self.git_wrapper(
            "apply",
            "printf 'late ignored WIP\\n' > \"$INJECT_PATH\"",
        )

        completed = fixture.invoke(
            "A",
            env={
                "PATH": f"{wrapper.parent}:{os.environ['PATH']}",
                "INJECT_WORKTREE": str(fixture.target),
                "INJECT_PATH": str(fixture.target / "late.txt"),
            },
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(
            (fixture.target / "late.txt").read_text(), "late ignored WIP\n"
        )
        self.assertEqual(git(fixture.target, "rev-parse", "A"), target_before)
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)

    def test_target_ref_compare_and_swap_preserves_concurrent_commit(self) -> None:
        fixture = self.fixture
        fixture.commit_current("current.txt", "from B\n", "Current change")
        fixture.commit_target("target.txt", "from A\n", "Target change")
        fixture.push_branches()
        current_before = git(fixture.current, "rev-parse", "B")
        remote_before = fixture.remote_oid("A")
        real_git = shutil.which("git")
        self.assertIsNotNone(real_git)
        wrapper = self.git_wrapper(
            "update-ref",
            f"{shlex.quote(real_git or 'git')} -C "
            f"{shlex.quote(str(fixture.target))} commit --no-verify -m "
            "'Concurrent local commit' >/dev/null",
        )

        completed = fixture.invoke(
            "A",
            env={
                "PATH": f"{wrapper.parent}:{os.environ['PATH']}",
                "INJECT_WORKTREE": str(fixture.current),
            },
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(
            git(fixture.target, "show", "-s", "--format=%s", "A"),
            "Concurrent local commit",
        )
        self.assertEqual(git(fixture.current, "rev-parse", "B"), current_before)
        self.assertEqual(fixture.remote_oid("A"), remote_before)

    def git_wrapper(
        self,
        intercepted_command: str,
        injection: str,
        *,
        condition: str | None = None,
        require_inject_worktree: bool = True,
    ) -> Path:
        real_git = shutil.which("git")
        self.assertIsNotNone(real_git)
        directory = self.fixture.root / f"git-wrapper-{intercepted_command}"
        directory.mkdir()
        wrapper = directory / "git"
        predicate = condition or f'[ "$1" = {shlex.quote(intercepted_command)} ]'
        worktree_predicate = (
            '[ "$PWD" = "$INJECT_WORKTREE" ]' if require_inject_worktree else "true"
        )
        wrapper.write_text(
            "#!/bin/sh\n"
            f"if {predicate} && {worktree_predicate}; then\n"
            f"  {injection}\n"
            "fi\n"
            f'exec {shlex.quote(real_git or "git")} "$@"\n'
        )
        wrapper.chmod(0o755)
        return wrapper


if __name__ == "__main__":
    unittest.main()
