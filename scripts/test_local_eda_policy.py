#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGED_CHECK = REPO_ROOT / "scripts" / "check_staged_local_paths.py"
ROOT_RESOLVER = REPO_ROOT / "scripts" / "resolve_experiment_root.py"
HOOK = REPO_ROOT / ".githooks" / "pre-commit"
TEST_ROOT = REPO_ROOT / "build" / "local-policy-tests"
TOP_LEVEL_TEMP_DIRECTORY = "temp"
DiskUsage = namedtuple("DiskUsage", "total used free")


def run(
    cwd: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def git(cwd: Path, *arguments: str, check: bool = True) -> str:
    return run(cwd, "git", *arguments, check=check).stdout.strip()


def write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def load_resolver():
    if not ROOT_RESOLVER.is_file():
        raise AssertionError(f"missing experiment-root resolver: {ROOT_RESOLVER}")
    spec = importlib.util.spec_from_file_location(
        "resolve_experiment_root", ROOT_RESOLVER
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load experiment-root resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepositoryFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        run(root, "git", "init", "--initial-branch=main")
        git(root, "config", "user.name", "Local Policy Test")
        git(root, "config", "user.email", "local-policy@example.com")
        git(root, "config", "core.excludesFile", "/dev/null")
        write(
            root / ".gitignore",
            "/build/\n/local-experiments/\n/loom-local-config.json\n",
        )
        write(root / "tracked.txt", "base\n")
        git(root, "add", ".gitignore", "tracked.txt")
        git(root, "commit", "-m", "Base")

    def invoke_staged_check(
        self, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        return run(
            cwd or self.root,
            "python3",
            str(STAGED_CHECK),
            check=False,
        )


class StagedPathPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="local-policy-test-", dir=TEST_ROOT
        )
        self.fixture = RepositoryFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rejects_staged_local_output_roots_and_config(self) -> None:
        for relative in (
            "build/result.log",
            "loom-local-config.json",
        ):
            with self.subTest(relative=relative):
                git(self.fixture.root, "reset", "--hard", "HEAD")
                git(self.fixture.root, "clean", "-fdx")
                write(self.fixture.root / relative, "private\n")
                git(self.fixture.root, "add", "--force", relative)

                completed = self.fixture.invoke_staged_check()

                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertIn(repr(relative), completed.stderr)

    def test_rejects_staged_top_level_temp_paths(self) -> None:
        relative = f"{TOP_LEVEL_TEMP_DIRECTORY}/forbidden-ledger.md"
        write(self.fixture.root / relative, "private\n")
        git(self.fixture.root, "add", "--force", relative)

        completed = self.fixture.invoke_staged_check()

        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn(repr(relative), completed.stderr)

    def test_rejects_any_tracked_ignored_file(self) -> None:
        relative = "private-output.txt"
        write(self.fixture.root / ".gitignore", "/private-output.txt\n")
        git(self.fixture.root, "add", ".gitignore")
        git(self.fixture.root, "commit", "-m", "Ignore private output")
        write(self.fixture.root / relative, "private\n")
        git(self.fixture.root, "add", "--force", relative)

        completed = self.fixture.invoke_staged_check()

        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn(repr(relative), completed.stderr)

    def test_allows_nested_lookalikes_and_forbidden_path_deletions(self) -> None:
        write(self.fixture.root / "nested/build/result.txt", "authored\n")
        git(self.fixture.root, "add", "nested")
        allowed = self.fixture.invoke_staged_check(self.fixture.root / "nested")
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

        write(self.fixture.root / "build/legacy.txt", "legacy\n")
        git(self.fixture.root, "add", "--force", "build/legacy.txt")
        git(self.fixture.root, "commit", "--no-verify", "-m", "Legacy output")
        git(self.fixture.root, "rm", "build/legacy.txt")
        deletion = self.fixture.invoke_staged_check()
        self.assertEqual(deletion.returncode, 0, deletion.stderr)

    def test_tracked_hook_delegates_to_the_staged_checker(self) -> None:
        self.assertTrue(HOOK.is_file(), f"missing hook: {HOOK}")
        completed = run(REPO_ROOT, "sh", "-n", str(HOOK), check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("check_staged_local_paths.py", HOOK.read_text())


class ExperimentRootPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="experiment-root-test-", dir=TEST_ROOT
        )
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.scratch = self.root / "scratch"
        self.cache = self.root / "cache" / "loom"
        self.temporary_root = self.root / "tmp"
        RepositoryFixture(self.repository)
        self.scratch.mkdir()
        self.temporary_root.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def resolve(self, *, configured_root: Path | None = None, cache: bool = True):
        module = load_resolver()
        return module.resolve_experiment_root(
            repository=self.repository,
            configured_root=configured_root,
            scratch_root=self.scratch,
            cache_root=self.cache if cache else None,
            temporary_root=self.temporary_root,
        )

    def test_explicit_external_root_has_highest_priority(self) -> None:
        configured = self.root / "configured"
        selected = self.resolve(configured_root=configured)
        self.assertEqual(selected, configured.resolve())
        self.assertTrue(selected.is_dir())

    def test_repository_build_must_exist_and_be_ignored(self) -> None:
        repository_build = self.repository / "build"
        repository_build.mkdir()
        with mock.patch.object(
            shutil, "disk_usage", return_value=DiskUsage(0, 0, 200 << 30)
        ):
            self.assertEqual(self.resolve(), repository_build.resolve())

        shutil.rmtree(repository_build)
        write(self.repository / ".gitignore", "/local-experiments/\n")
        repository_build.mkdir()
        with mock.patch.object(
            shutil, "disk_usage", return_value=DiskUsage(0, 0, 200 << 30)
        ):
            selected = self.resolve()
        self.assertEqual(selected, (self.scratch / f"loom-{os.getuid()}").resolve())

    def test_large_scratch_precedes_cache(self) -> None:
        with mock.patch.object(
            shutil, "disk_usage", return_value=DiskUsage(0, 0, (100 << 30) + 1)
        ):
            selected = self.resolve()
        self.assertEqual(selected, (self.scratch / f"loom-{os.getuid()}").resolve())

    def test_cache_and_temporary_fallbacks_are_created(self) -> None:
        with mock.patch.object(
            shutil, "disk_usage", return_value=DiskUsage(0, 0, 100 << 30)
        ):
            self.assertEqual(self.resolve(), self.cache.resolve())
        shutil.rmtree(self.cache)
        with mock.patch.object(shutil, "disk_usage", return_value=DiskUsage(0, 0, 0)):
            selected = self.resolve(cache=False)
        self.assertEqual(
            selected, (self.temporary_root / f"loom-{os.getuid()}").resolve()
        )

    def test_explicit_ignored_repository_path_is_allowed(self) -> None:
        configured = self.repository / "local-experiments" / "run"
        selected = self.resolve(configured_root=configured)
        self.assertEqual(selected, configured.resolve())
        self.assertTrue(selected.is_dir())

    def test_explicit_unignored_repository_path_is_rejected(self) -> None:
        module = load_resolver()
        with self.assertRaisesRegex(
            module.ExperimentRootError, "must be Git-ignored"
        ):
            module.resolve_experiment_root(
                repository=self.repository,
                configured_root=self.repository / "other",
                scratch_root=self.scratch,
                cache_root=self.cache,
                temporary_root=self.temporary_root,
            )

    def test_external_tool_cache_selection_and_cleanup_are_exact(self) -> None:
        module = load_resolver()
        repository_build = self.repository / "build"
        repository_build.mkdir()
        selected = module.resolve_external_tool_cache_root(
            repository=self.repository,
            configured_root=None,
            environment={},
            scratch_root=self.scratch,
            cache_root=self.cache,
            temporary_root=self.temporary_root,
        )
        self.assertEqual(selected, repository_build / "external-tool-cache")
        self.assertFalse(selected.exists())

        explicit = self.root / "explicit-cache"
        overridden = module.resolve_external_tool_cache_root(
            repository=self.repository,
            configured_root=None,
            environment={module.EXTERNAL_TOOL_CACHE_ROOT_ENVIRONMENT: str(explicit)},
            scratch_root=self.scratch,
            cache_root=self.cache,
            temporary_root=self.temporary_root,
        )
        self.assertEqual(overridden, explicit)

        explicit.mkdir()
        write(
            explicit / module.EXTERNAL_TOOL_CACHE_MARKER,
            module.EXTERNAL_TOOL_CACHE_MARKER_CONTENTS,
        )
        write(explicit / "entries" / "cached", "payload\n")
        (explicit / "locks").mkdir()
        self.assertTrue(module.remove_external_tool_cache_root(explicit))
        self.assertFalse(explicit.exists())

        unmarked = self.root / "unmarked-cache"
        write(unmarked / "keep", "owned elsewhere\n")
        with self.assertRaisesRegex(module.ExperimentRootError, "unmarked"):
            module.remove_external_tool_cache_root(unmarked)
        self.assertTrue((unmarked / "keep").is_file())

        foreign = self.root / "marked-cache-with-foreign-member"
        write(
            foreign / module.EXTERNAL_TOOL_CACHE_MARKER,
            module.EXTERNAL_TOOL_CACHE_MARKER_CONTENTS,
        )
        write(foreign / "unrelated", "keep\n")
        with self.assertRaisesRegex(module.ExperimentRootError, "foreign"):
            module.remove_external_tool_cache_root(foreign)
        self.assertTrue((foreign / "unrelated").is_file())


if __name__ == "__main__":
    unittest.main()
