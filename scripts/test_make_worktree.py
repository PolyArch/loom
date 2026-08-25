#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import io
import json
import multiprocessing
import os
import select
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from contextlib import ExitStack, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from resolve_experiment_root import (
    EXTERNAL_TOOL_CACHE_MARKER,
    EXTERNAL_TOOL_CACHE_MARKER_CONTENTS,
)

SCRIPT = Path(__file__).with_name("make-worktree.py")
REPO_ROOT = SCRIPT.parents[1]
REPO_TEMP_ROOT = REPO_ROOT / "build" / "test-runs"
UNSET = object()


def load_dispatcher():
    spec = importlib.util.spec_from_file_location("make_worktree", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    git(path, "config", "user.name", "Loom Test")
    git(path, "config", "user.email", "loom-test@example.com")


def commit_file(repo: Path, name: str, content: str) -> str:
    (repo / name).write_text(content)
    git(repo, "add", name)
    git(repo, "commit", "-qm", f"Update {name}")
    return git(repo, "rev-parse", "HEAD")


class GitTopology:
    def __init__(self, root: Path):
        self.llvm_repo = root / "llvm-repo"
        init_repo(self.llvm_repo)
        self.llvm_pin = commit_file(self.llvm_repo, "llvm.txt", "llvm-a\n")
        commit_file(self.llvm_repo, "llvm.txt", "llvm-b\n")

        self.circt_repo = root / "circt-repo"
        init_repo(self.circt_repo)
        (self.circt_repo / ".gitmodules").write_text(f'[submodule "llvm"]\n\tpath = llvm\n\turl = {self.llvm_repo}\n')
        (self.circt_repo / "circt.txt").write_text("circt\n")
        git(self.circt_repo, "add", ".gitmodules", "circt.txt")
        git(
            self.circt_repo,
            "update-index",
            "--add",
            "--cacheinfo",
            "160000",
            self.llvm_pin,
            "llvm",
        )
        git(self.circt_repo, "commit", "-qm", "Pin LLVM")
        self.circt_pin = git(self.circt_repo, "rev-parse", "HEAD")

        self.or_tools_repo = root / "or-tools-repo"
        init_repo(self.or_tools_repo)
        self.or_tools_pin = commit_file(self.or_tools_repo, "or-tools.txt", "or-tools\n")

        self.main = root / "main"
        init_repo(self.main)
        for source, path in (
            (self.circt_repo, "externals/circt"),
            (self.llvm_repo, "externals/llvm"),
            (self.or_tools_repo, "externals/or-tools"),
        ):
            git(
                self.main,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "-q",
                str(source),
                path,
            )
        git(self.main / "externals/llvm", "checkout", "-q", self.llvm_pin)
        git(
            self.main,
            "add",
            ".gitmodules",
            "externals/circt",
            "externals/llvm",
            "externals/or-tools",
        )
        git(self.main, "commit", "-qm", "Pin dependencies")

        self.linked = root / "linked"
        git(
            self.main,
            "worktree",
            "add",
            "-q",
            "-b",
            "linked",
            str(self.linked),
        )


def build_paths(root: Path):
    externals = root / "externals"
    llvm_root = externals / "llvm"
    llvm_build = llvm_root / "build"
    circt_root = externals / "circt"
    circt_build = circt_root / "build"
    or_tools_root = externals / "or-tools"
    or_tools_build = or_tools_root / "build"
    or_tools_install = or_tools_root / "install"
    return SimpleNamespace(
        root=root,
        main=root,
        is_main=True,
        externals_root=externals,
        llvm_root=llvm_root,
        llvm_src=llvm_root / "llvm",
        llvm_build=llvm_build,
        llvm_lock=externals / ".loom-build.llvm.lock",
        llvm_lock_turnstile=externals / ".loom-build.llvm.turnstile.lock",
        llvm_stamp=externals / ".loom-build.llvm.stamp",
        mlir_dir=llvm_build / "lib" / "cmake" / "mlir",
        cmake_llvm_dir=llvm_build / "lib" / "cmake" / "llvm",
        cmake_clang_dir=llvm_build / "lib" / "cmake" / "clang",
        cmake_polly_dir=(
            llvm_build / "tools" / "polly" / "lib" / "cmake" / "polly"
        ),
        llvm_lit=llvm_build / "bin" / "llvm-lit",
        circt_root=circt_root,
        circt_build=circt_build,
        circt_stamp=externals / ".loom-build.circt.stamp",
        circt_cmake_dir=circt_build / "lib" / "cmake" / "circt",
        or_tools_root=or_tools_root,
        or_tools_build=or_tools_build,
        or_tools_install=or_tools_install,
        or_tools_lock=externals / ".loom-build.or-tools.lock",
        or_tools_lock_turnstile=(externals / ".loom-build.or-tools.turnstile.lock"),
        or_tools_stamp=externals / ".loom-build.or-tools.stamp",
        or_tools_cmake_dir=(or_tools_install / "lib" / "cmake" / "ortools"),
        loom_build=root / "build",
    )


class MakeWorktreeTest(unittest.TestCase):
    def setUp(self):
        self.module = load_dispatcher()
        self.state = self.module.DependencyState("circt-pin", "llvm-pin", "or-tools-pin")
        self.compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        self.loom_compilers = (
            ("/clang", "clang 21.1.8"),
            ("/clang++", "clang 21.1.8"),
        )
        self.args = Namespace(jobs=1, lock_timeout=1.0)
        self.llvm_identity = self.module.llvm_build_identity(self.state.llvm_commit, self.compilers)
        self.circt_identity = self.module.circt_build_identity(self.llvm_identity, self.state.circt_commit)
        self.or_tools_identity = self.module.or_tools_build_identity(self.state.or_tools_commit, self.loom_compilers)
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    def test_parallelism_reserves_development_capacity_and_caps_workers(self) -> None:
        self.assertEqual(self.module.bounded_job_count(999, cpu_count=32), 28)
        self.assertEqual(self.module.bounded_job_count(999, cpu_count=256), 120)
        self.assertEqual(self.module.bounded_job_count(999, cpu_count=4), 1)
        self.assertEqual(self.module.bounded_job_count(8, cpu_count=32), 8)

    def test_shared_llvm_build_includes_corpus_targets(self) -> None:
        paths = build_paths(REPO_TEMP_ROOT / "polly-layout")
        self.assertEqual(
            paths.cmake_polly_dir.relative_to(paths.llvm_build),
            Path("tools/polly/lib/cmake/polly"),
        )
        projects = next(
            argument
            for argument in self.module.LLVM_SEMANTIC_CMAKE_ARGS
            if argument.startswith("-DLLVM_ENABLE_PROJECTS=")
        )
        self.assertIn(
            "polly",
            projects.removeprefix("-DLLVM_ENABLE_PROJECTS=").split(";"),
        )
        targets = next(
            argument
            for argument in self.module.LLVM_SEMANTIC_CMAKE_ARGS
            if argument.startswith("-DLLVM_TARGETS_TO_BUILD=")
        )
        self.assertGreaterEqual(
            set(targets.removeprefix("-DLLVM_TARGETS_TO_BUILD=").split(";")),
            {"host", "RISCV", "ARM", "AArch64"},
        )

    def build_environment(self, module, run=UNSET):
        stack = ExitStack()
        stack.enter_context(patch.object(module, "check_git_version"))
        stack.enter_context(
            patch.object(
                module,
                "check_loom_compilers",
                return_value=self.loom_compilers,
            )
        )
        stack.enter_context(patch.object(module, "is_nfs", return_value=False))
        stack.enter_context(
            patch.object(
                module,
                "check_dependency_pins",
                return_value=self.state,
            )
        )
        stack.enter_context(
            patch.object(
                module,
                "check_llvm_compilers",
                return_value=self.compilers,
            )
        )
        if run is not UNSET:
            stack.enter_context(patch.object(module, "run", side_effect=run))
        return stack

    def write_llvm_artifacts(self, paths) -> None:
        for directory, name in (
            (paths.mlir_dir, "MLIRConfig.cmake"),
            (paths.cmake_llvm_dir, "LLVMConfig.cmake"),
            (paths.cmake_clang_dir, "ClangConfig.cmake"),
            (paths.cmake_polly_dir, "PollyConfig.cmake"),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text("ready\n")
        (paths.llvm_build / "build.ninja").write_text("ninja\n")
        (paths.llvm_build / "lib" / "libPollyISL.a").write_text("archive\n")
        paths.llvm_lit.parent.mkdir(parents=True, exist_ok=True)
        paths.llvm_lit.write_text("#!/bin/sh\nexit 0\n")
        paths.llvm_lit.chmod(0o755)

    def ready_llvm(self, paths, stamp: str | None = None) -> None:
        self.write_llvm_artifacts(paths)
        paths.llvm_stamp.parent.mkdir(parents=True, exist_ok=True)
        paths.llvm_stamp.write_text((stamp or self.llvm_identity) + "\n")
        self.ready_or_tools(paths)

    def ready_circt(self, paths, stamp: str | None = None) -> None:
        paths.circt_cmake_dir.mkdir(parents=True, exist_ok=True)
        (paths.circt_cmake_dir / "CIRCTConfig.cmake").write_text("ready\n")
        (paths.circt_build / "build.ninja").write_text("ninja\n")
        paths.circt_stamp.write_text((stamp or self.circt_identity) + "\n")

    def ready_or_tools(self, paths, stamp: str | None = None) -> None:
        paths.or_tools_build.mkdir(parents=True, exist_ok=True)
        paths.or_tools_cmake_dir.mkdir(parents=True, exist_ok=True)
        (paths.or_tools_cmake_dir / "ortoolsConfig.cmake").write_text("set(ortools_VERSION 9.15)\n")
        (paths.or_tools_cmake_dir / "ortoolsTargets.cmake").write_text(
            "add_library(ortools::ortools INTERFACE IMPORTED)\n"
        )
        (paths.or_tools_cmake_dir / "loom-source-commit.txt").write_text(self.state.or_tools_commit + "\n")
        (paths.or_tools_build / "build.ninja").write_text("ninja\n")
        paths.or_tools_stamp.write_text((stamp or self.or_tools_identity) + "\n")

    def loom_consumer(self, shared, root: Path, configured: bool = True):
        paths = SimpleNamespace(**vars(shared))
        paths.root = root
        paths.is_main = False
        paths.loom_build = root / "build"
        if configured:
            paths.loom_build.mkdir(parents=True)
            (paths.loom_build / "build.ninja").write_text("ninja\n")
            (paths.loom_build / "CMakeCache.txt").write_text(
                "CMAKE_C_COMPILER:FILEPATH=/usr/bin/clang\n"
                "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++\n"
                f"Polly_DIR:PATH={shared.cmake_polly_dir}\n"
                f"ortools_DIR:PATH={shared.or_tools_cmake_dir}\n"
                "LOOM_ORTOOLS_SOURCE_COMMIT:STRING="
                f"{self.state.or_tools_commit}\n"
                f"LOOM_EXTERNAL_SOURCE_DIR:PATH={shared.externals_root}\n"
            )
        return paths

    def join_processes(self, processes) -> None:
        for process in processes:
            if process.pid is None:
                continue
            process.join(2.0)
            if process.is_alive():
                process.terminate()
                process.join(2.0)

    def wait_for_turnstile_records(
        self,
        path: Path,
        expected: int,
        timeout: float = 2.0,
    ) -> None:
        expected_size = expected * self.module._TURNSTILE_RECORD_SIZE
        deadline = time.monotonic() + timeout
        while path.stat().st_size != expected_size:
            if time.monotonic() >= deadline:
                self.fail(f"turnstile holds {path.stat().st_size} bytes, expected {expected_size}")
            select.select((), (), (), 0.005)

    def wait_for_lock_holders(
        self,
        path: Path,
        expected_pids: set[int],
        timeout: float = 1.0,
    ) -> None:
        self.wait_for_lock_processes(
            path,
            expected_holders=expected_pids,
            timeout=timeout,
        )

    def wait_for_lock_processes(
        self,
        path: Path,
        expected_holders: set[int] | None = None,
        expected_waiters: set[int] | None = None,
        expected_processes: set[int] | None = None,
        timeout: float = 1.0,
    ) -> None:
        stat = path.stat()
        key = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
        expected_holders = expected_holders or set()
        expected_waiters = expected_waiters or set()
        expected_processes = expected_processes or set()
        deadline = time.monotonic() + timeout
        while True:
            holders = set()
            waiters = set()
            for line in Path("/proc/locks").read_text().splitlines():
                fields = line.split()
                if len(fields) >= 6 and fields[1] in ("FLOCK", "POSIX") and fields[5] == key:
                    holders.add(int(fields[4]))
                elif len(fields) >= 7 and fields[1] == "->" and fields[2] in ("FLOCK", "POSIX") and fields[6] == key:
                    waiters.add(int(fields[5]))
            if expected_holders <= holders and expected_waiters <= waiters and expected_processes <= holders | waiters:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.fail(
                    f"flock holders {holders} and waiters {waiters} did not "
                    f"include holders {expected_holders}, waiters "
                    f"{expected_waiters}, and processes {expected_processes}"
                )
            select.select((), (), (), min(0.005, remaining))

    def capture_die(self, fn, *args, **kwargs) -> str:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            fn(*args, **kwargs)
        return stderr.getvalue()

    def test_linked_worktree_routes_to_primary_artifact_owner(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            paths = self.module.Paths(topology.linked)

            self.assertEqual(paths.main, topology.main)
            self.assertEqual(paths.externals_root, topology.main / "externals")
            self.assertEqual(
                self.module.gitlinks_at_head(topology.linked),
                (
                    "externals/circt",
                    "externals/llvm",
                    "externals/or-tools",
                ),
            )
            linked_status = git(topology.linked, "submodule", "status")
            self.assertTrue(all(line.startswith("-") for line in linked_status.splitlines()))
            nested_status = git(
                topology.main / "externals/circt",
                "submodule",
                "status",
                "llvm",
            )
            self.assertTrue(nested_status.startswith("-"))

            state = self.module.check_dependency_pins(paths)
            self.assertEqual(state.circt_commit, topology.circt_pin)
            self.assertEqual(state.llvm_commit, topology.llvm_pin)
            self.assertEqual(state.or_tools_commit, topology.or_tools_pin)

    def test_primary_owner_runtime_state_uses_canonical_ignore_policy(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            (topology.main / ".gitignore").write_text((REPO_ROOT / ".gitignore").read_text())
            runtime_state = (
                "externals/.loom-build.llvm.lock",
                "externals/.loom-build.llvm.turnstile.lock",
                "externals/.loom-build.llvm.stamp",
                "externals/.loom-build.circt.stamp",
                "externals/.loom-build.or-tools.lock",
                "externals/.loom-build.or-tools.turnstile.lock",
                "externals/.loom-build.or-tools.stamp",
            )
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(topology.main),
                    "-c",
                    "core.excludesFile=/dev/null",
                    "check-ignore",
                    "--",
                    *runtime_state,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(result.stdout.split()), set(runtime_state))

    def test_dependency_identities_have_one_owner_each(self):
        same_llvm = self.module.llvm_build_identity(self.state.llvm_commit, self.compilers)
        changed_circt = self.module.DependencyState(
            "other-circt",
            self.state.llvm_commit,
            self.state.or_tools_commit,
        )
        self.assertEqual(
            same_llvm,
            self.module.llvm_build_identity(changed_circt.llvm_commit, self.compilers),
        )
        self.assertNotEqual(
            self.module.circt_build_identity(same_llvm, self.state.circt_commit),
            self.module.circt_build_identity(same_llvm, changed_circt.circt_commit),
        )

        payload = json.loads(same_llvm)
        self.assertEqual(payload["dependencies"], {"llvm": "llvm-pin"})
        circt_payload = json.loads(self.circt_identity)
        self.assertNotIn("circt_build_targets", circt_payload)
        expected = ("-DCIRCT_INCLUDE_TOOLS=OFF",
                    "-DCIRCT_SLANG_FRONTEND_ENABLED=ON",
                    "-DCIRCT_SLANG_BUILD_FROM_SOURCE=ON")
        self.assertEqual(self.module.CIRCT_SEMANTIC_CMAKE_ARGS[-3:], expected)
        self.assertEqual(tuple(arg for arg in circt_payload["circt_semantic_cmake_args"] if arg in expected), expected)
        or_tools_payload = json.loads(self.or_tools_identity)
        self.assertEqual(
            or_tools_payload["dependencies"],
            {"or_tools": self.state.or_tools_commit},
        )
        self.assertEqual(or_tools_payload["compilers"]["cxx"]["path"], "/clang++")
        self.assertIn("-DUSE_GUROBI=OFF", or_tools_payload["semantic_cmake_args"])
        self.assertIn("-DUSE_XPRESS=OFF", or_tools_payload["semantic_cmake_args"])

        legacy_payload = json.loads(same_llvm)
        legacy_payload["dependencies"]["circt"] = "old-circt"
        legacy_stamp = json.dumps(legacy_payload, sort_keys=True)
        self.assertTrue(self.module.llvm_stamp_matches(legacy_stamp, same_llvm))
        self.assertFalse(
            self.module.llvm_stamp_matches(
                legacy_stamp,
                self.module.llvm_build_identity("other-llvm", self.compilers),
            )
        )

    def test_circt_pin_change_keeps_public_llvm_fast_path(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            legacy_payload = json.loads(self.llvm_identity)
            legacy_payload["dependencies"]["circt"] = "old-circt"
            legacy_stamp = json.dumps(legacy_payload, sort_keys=True)
            self.ready_llvm(shared, legacy_stamp)
            consumer = self.loom_consumer(shared, Path(td) / "consumer")
            calls = []

            def capture_run(cmd, **kwargs):
                calls.append(cmd)

            with self.build_environment(self.module, run=capture_run):
                self.module.cmd_build_loom(consumer, self.args)

            self.assertEqual(
                calls,
                [
                    [
                        "cmake",
                        "--build",
                        str(consumer.loom_build),
                        "-j1",
                    ]
                ],
            )
            self.assertEqual(self.module.read_stamp(shared.llvm_stamp), legacy_stamp)

    def test_ready_public_loom_readers_overlap(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            self.ready_llvm(shared)
            readers = {name: self.loom_consumer(shared, Path(td) / name) for name in ("reader-a", "reader-b")}
            context = multiprocessing.get_context("fork")
            active = {name: context.Event() for name in readers}
            release = {name: context.Event() for name in readers}
            by_build = {str(paths.loom_build): name for name, paths in readers.items()}

            def controlled_run(cmd, **kwargs):
                name = by_build.get(cmd[2]) if cmd[:2] == ["cmake", "--build"] else None
                if name is None:
                    raise AssertionError(f"unexpected command: {cmd}")
                active[name].set()
                if not release[name].wait(2.0):
                    raise RuntimeError(f"{name} release timed out")

            def consume(name):
                self.module.cmd_build_loom(readers[name], self.args)

            processes = {name: context.Process(target=consume, args=(name,), name=name) for name in readers}
            try:
                with self.build_environment(self.module, run=controlled_run):
                    processes["reader-a"].start()
                    self.assertTrue(active["reader-a"].wait(1.0))
                    processes["reader-b"].start()
                    self.assertTrue(
                        active["reader-b"].wait(0.5),
                        "ready Loom readers did not overlap",
                    )
            finally:
                for event in release.values():
                    event.set()
                self.join_processes(processes.values())

            self.assertEqual(
                {name: process.exitcode for name, process in processes.items()},
                {name: 0 for name in processes},
            )

    def test_queued_reader_precedes_later_writers(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            self.ready_llvm(shared)
            readers = {
                name: self.loom_consumer(shared, Path(td) / name) for name in ("holding-reader", "queued-reader")
            }
            context = multiprocessing.get_context("fork")
            later_writers = tuple(f"later-writer-{index}" for index in range(8))
            names = (
                "holding-reader",
                "first-writer",
                "queued-reader",
                *later_writers,
            )
            active = {name: context.Event() for name in names}
            release = {name: context.Event() for name in names}
            started = {name: context.Event() for name in names if name != "holding-reader"}
            order = context.Queue()
            by_build = {str(paths.loom_build): name for name, paths in readers.items()}
            args = Namespace(jobs=1, lock_timeout=3.0)

            def controlled_run(cmd, **kwargs):
                if cmd[:2] != ["cmake", "--build"]:
                    raise AssertionError(f"unexpected command: {cmd}")
                name = by_build.get(cmd[2])
                if cmd[2] == str(shared.llvm_build):
                    name = multiprocessing.current_process().name
                if name is None:
                    raise AssertionError(f"unexpected command: {cmd}")
                order.put(name)
                active[name].set()
                if not release[name].wait(2.0):
                    raise RuntimeError(f"{name} release timed out")

            def consume(name):
                if name == "queued-reader":
                    started[name].set()
                self.module.cmd_build_loom(readers[name], args)

            def write(name):
                started[name].set()
                self.module.cmd_build_llvm(shared, args)

            processes = {
                "holding-reader": context.Process(
                    target=consume,
                    args=("holding-reader",),
                    name="holding-reader",
                ),
                "queued-reader": context.Process(
                    target=consume,
                    args=("queued-reader",),
                    name="queued-reader",
                ),
                "first-writer": context.Process(
                    target=write,
                    args=("first-writer",),
                    name="first-writer",
                ),
            }
            processes.update({name: context.Process(target=write, args=(name,), name=name) for name in later_writers})
            try:
                with self.build_environment(self.module, run=controlled_run):
                    processes["holding-reader"].start()
                    self.assertTrue(active["holding-reader"].wait(1.0))
                    self.assertEqual(order.get(timeout=1.0), "holding-reader")

                    processes["first-writer"].start()
                    self.assertTrue(started["first-writer"].wait(0.5))
                    self.wait_for_lock_holders(
                        shared.llvm_lock_turnstile,
                        {processes["first-writer"].pid},
                    )

                    processes["queued-reader"].start()
                    self.assertTrue(started["queued-reader"].wait(0.5))
                    self.wait_for_lock_processes(
                        shared.llvm_lock_turnstile,
                        expected_waiters={processes["queued-reader"].pid},
                    )

                    for name in later_writers:
                        processes[name].start()
                        self.assertTrue(started[name].wait(0.5))
                        self.wait_for_lock_processes(
                            shared.llvm_lock_turnstile,
                            expected_processes={processes[name].pid},
                        )

                    release["holding-reader"].set()
                    self.assertEqual(order.get(timeout=1.0), "first-writer")
                    release["first-writer"].set()
                    self.assertEqual(order.get(timeout=1.0), "queued-reader")
                    release["queued-reader"].set()
            finally:
                for event in release.values():
                    event.set()
                self.join_processes(processes.values())
                order.close()
                order.join_thread()

            self.assertEqual(
                {name: process.exitcode for name, process in processes.items()},
                {name: 0 for name in processes},
            )

    def test_public_test_holds_llvm_lease_through_lit(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            self.ready_llvm(shared)
            consumer = self.loom_consumer(shared, Path(td) / "consumer")
            context = multiprocessing.get_context("fork")
            lit_active = context.Event()
            release_lit = context.Event()
            writer_active = context.Event()
            release_writer = context.Event()

            def controlled_run(cmd, **kwargs):
                if cmd and cmd[0] == str(shared.llvm_lit):
                    lit_active.set()
                    if not release_lit.wait(2.0):
                        raise RuntimeError("lit release timed out")
                    Path(cmd[cmd.index("--output") + 1]).write_text('{"tests": []}')
                    return
                if cmd[:2] == ["cmake", "--build"]:
                    if cmd[2] == str(consumer.loom_build):
                        return
                    if cmd[2] == str(shared.llvm_build):
                        writer_active.set()
                        if not release_writer.wait(2.0):
                            raise RuntimeError("writer release timed out")
                        return
                raise AssertionError(f"unexpected command: {cmd}")

            test_process = context.Process(
                target=self.module.cmd_test,
                args=(consumer, self.args),
                name="loom-test",
            )
            writer_process = context.Process(
                target=self.module.cmd_build_llvm,
                args=(shared, self.args),
                name="llvm-writer",
            )
            try:
                with self.build_environment(self.module, run=controlled_run):
                    test_process.start()
                    self.assertTrue(lit_active.wait(1.0))
                    writer_process.start()
                    self.wait_for_lock_holders(
                        shared.llvm_lock_turnstile,
                        {writer_process.pid},
                    )
                    self.assertFalse(writer_active.is_set())
                    release_lit.set()
                    self.assertTrue(writer_active.wait(1.0))
                    release_writer.set()
            finally:
                release_lit.set()
                release_writer.set()
                self.join_processes((test_process, writer_process))

            self.assertEqual(test_process.exitcode, 0)
            self.assertEqual(writer_process.exitcode, 0)

    def test_dispatcher_death_keeps_lease_until_mutator_stops(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            root = Path(td)
            shared = build_paths(root / "shared")
            self.ready_llvm(shared)
            consumer = self.loom_consumer(shared, root / "consumer")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_cmake = fake_bin / "cmake"
            heartbeat = root / "heartbeat"
            fake_cmake.write_text(
                "#!/usr/bin/env python3\n"
                "import os\n"
                "import socket\n"
                "import time\n"
                "with open(os.environ['LOOM_TEST_HEARTBEAT'], 'ab', "
                "buffering=0) as output:\n"
                "    with socket.socket() as channel:\n"
                "        channel.connect(('127.0.0.1', "
                "int(os.environ['LOOM_TEST_PORT'])))\n"
                "        channel.sendall("
                "f'{os.getpid()} {os.getppid()}\\n'.encode())\n"
                "    while True:\n"
                "        output.write(b'x')\n"
                "        time.sleep(0.001)\n"
            )
            fake_cmake.chmod(0o755)
            server = socket.socket()
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(1.0)
            server_port = server.getsockname()[1]
            context = multiprocessing.get_context("fork")
            writer_active = context.Event()
            release_writer = context.Event()

            def run_dispatcher():
                os.environ["PATH"] = f"{fake_bin}{os.pathsep}{os.environ['PATH']}"
                os.environ["LOOM_TEST_PORT"] = str(server_port)
                os.environ["LOOM_TEST_HEARTBEAT"] = str(heartbeat)
                self.module.cmd_build_loom(consumer, self.args)

            def compete():
                with self.module.SharedProductLock(
                    shared.llvm_lock,
                    shared.llvm_lock_turnstile,
                    2.0,
                    shared=False,
                ):
                    writer_active.set()
                    release_writer.wait(2.0)

            dispatcher = context.Process(target=run_dispatcher, name="loom-dispatcher")
            writer = context.Process(target=compete, name="llvm-writer")
            command_pid = None
            supervisor_pid = None
            child_pidfds = []

            def process_exists(pid):
                try:
                    os.kill(pid, 0)
                    return True
                except ProcessLookupError:
                    return False

            try:
                with self.build_environment(self.module):
                    dispatcher.start()
                    connection, _ = server.accept()
                    with connection:
                        command_pid, supervisor_pid = tuple(int(pid) for pid in connection.recv(128).decode().split())
                    child_pidfds = [os.pidfd_open(pid) for pid in (command_pid, supervisor_pid)]
                    self.assertEqual(os.getpgid(command_pid), os.getpgid(supervisor_pid))
                    lock_stat = shared.llvm_lock.stat()
                    command_files = {
                        (fd.stat().st_dev, fd.stat().st_ino) for fd in Path(f"/proc/{command_pid}/fd").iterdir()
                    }
                    self.assertIn(
                        (lock_stat.st_dev, lock_stat.st_ino),
                        command_files,
                        "the mutating command must own the product lease",
                    )
                    turnstile_stat = shared.llvm_lock_turnstile.stat()
                    self.assertNotIn(
                        (turnstile_stat.st_dev, turnstile_stat.st_ino),
                        command_files,
                    )
                    writer.start()
                    self.wait_for_lock_holders(
                        shared.llvm_lock_turnstile,
                        {writer.pid},
                    )

                    os.kill(supervisor_pid, signal.SIGSTOP)
                    deadline = time.monotonic() + 1.0
                    while True:
                        status = Path(f"/proc/{supervisor_pid}/status").read_text()
                        if "\nState:\tT" in status:
                            break
                        if time.monotonic() >= deadline:
                            self.fail("supervisor did not stop")
                        select.select((), (), (), 0.005)

                    before = heartbeat.stat().st_size
                    dispatcher.kill()
                    dispatcher.join(1.0)
                    self.assertFalse(dispatcher.is_alive())
                    self.assertFalse(
                        writer_active.wait(0.1),
                        "competing writer acquired while a mutating child "
                        f"remained active ({heartbeat.stat().st_size - before} "
                        "post-lease writes)",
                    )
                    os.kill(supervisor_pid, signal.SIGCONT)
                    self.assertTrue(writer_active.wait(1.0))
                    command_ready, _, _ = select.select([child_pidfds[0]], (), (), 0)
                    self.assertEqual(
                        command_ready,
                        [child_pidfds[0]],
                        "competing writer acquired before the mutator died",
                    )
                    final_size = heartbeat.stat().st_size
                    select.select((), (), (), 0.02)
                    self.assertEqual(
                        heartbeat.stat().st_size,
                        final_size,
                        "mutator wrote after its product lease ended",
                    )
                    release_writer.set()

                    ready, _, _ = select.select(child_pidfds, (), (), 1.0)
                    self.assertEqual(
                        set(ready),
                        set(child_pidfds),
                        "build process group survived dispatcher death",
                    )
            finally:
                release_writer.set()
                if supervisor_pid is not None and process_exists(supervisor_pid):
                    os.kill(supervisor_pid, signal.SIGCONT)
                if dispatcher.is_alive():
                    dispatcher.kill()
                    dispatcher.join(1.0)
                self.join_processes((writer,))
                for pid in (command_pid, supervisor_pid):
                    if pid is None:
                        continue
                    if process_exists(pid):
                        os.kill(pid, signal.SIGTERM)
                for pidfd in child_pidfds:
                    os.close(pidfd)
                server.close()

    def test_failed_automatic_repair_revokes_all_dependent_readiness(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            self.ready_llvm(paths)
            self.ready_circt(paths)
            consumer = self.loom_consumer(paths, Path(td) / "consumer")
            paths.llvm_lit.unlink()
            real_rmtree = shutil.rmtree

            def checked_removal(*args, **kwargs):
                self.assertFalse(paths.llvm_stamp.exists())
                self.assertFalse(paths.circt_stamp.exists())
                return real_rmtree(*args, **kwargs)

            def fail_configure(*args, **kwargs):
                self.assertFalse(paths.llvm_stamp.exists())
                self.assertFalse(paths.circt_stamp.exists())
                self.write_llvm_artifacts(paths)
                raise RuntimeError("injected configure failure")

            with (
                self.build_environment(
                    self.module,
                    run=lambda *args, **kwargs: None,
                ),
                patch.object(
                    self.module.shutil,
                    "rmtree",
                    side_effect=checked_removal,
                ),
                patch.object(
                    self.module,
                    "configure_llvm",
                    side_effect=fail_configure,
                ),
                self.assertRaisesRegex(RuntimeError, "injected configure failure"),
            ):
                self.module.cmd_build_loom(consumer, self.args)

            self.assertFalse(paths.llvm_stamp.exists())
            self.assertFalse(paths.circt_stamp.exists())

            with (
                self.build_environment(
                    self.module,
                    run=lambda *args, **kwargs: None,
                ),
                patch.object(
                    self.module,
                    "configure_llvm",
                    side_effect=RuntimeError("repair retried"),
                ),
                self.assertRaisesRegex(RuntimeError, "repair retried"),
            ):
                self.module.cmd_build_loom(consumer, self.args)

    def test_public_writer_preflights_revoke_owned_readiness(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            llvm_paths = build_paths(Path(td) / "llvm")
            self.ready_llvm(llvm_paths)
            self.ready_circt(llvm_paths)

            def fail_llvm_preflight():
                self.assertFalse(llvm_paths.llvm_stamp.exists())
                self.assertFalse(llvm_paths.circt_stamp.exists())
                raise RuntimeError("LLVM preflight failed")

            with (
                patch.object(self.module, "is_nfs", return_value=False),
                patch.object(
                    self.module,
                    "check_git_version",
                    side_effect=fail_llvm_preflight,
                ),
                self.assertRaisesRegex(RuntimeError, "LLVM preflight failed"),
            ):
                self.module.cmd_build_llvm(llvm_paths, self.args)

            circt_paths = build_paths(Path(td) / "circt")
            self.ready_llvm(circt_paths)
            self.ready_circt(circt_paths)

            def fail_circt_preflight():
                self.assertEqual(
                    self.module.read_stamp(circt_paths.llvm_stamp),
                    self.llvm_identity,
                )
                self.assertFalse(circt_paths.circt_stamp.exists())
                raise RuntimeError("CIRCT preflight failed")

            with (
                patch.object(self.module, "is_nfs", return_value=False),
                patch.object(
                    self.module,
                    "check_git_version",
                    side_effect=fail_circt_preflight,
                ),
                self.assertRaisesRegex(RuntimeError, "CIRCT preflight failed"),
            ):
                self.module.cmd_build_circt(circt_paths, self.args)

    def test_explicit_llvm_build_restores_only_validated_llvm_readiness(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            self.ready_llvm(paths)
            self.ready_circt(paths)

            def fail_build(cmd, **kwargs):
                raise subprocess.CalledProcessError(1, cmd)

            with (
                self.build_environment(self.module, run=fail_build),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                self.module.cmd_build_llvm(paths, self.args)
            self.assertFalse(paths.llvm_stamp.exists())
            self.assertFalse(paths.circt_stamp.exists())

            with (
                self.build_environment(self.module, run=lambda *args, **kwargs: None),
                patch.object(
                    self.module,
                    "configure_llvm",
                    side_effect=lambda *args: self.write_llvm_artifacts(paths),
                ),
            ):
                self.module.cmd_build_llvm(paths, self.args)
            self.assertEqual(
                self.module.read_stamp(paths.llvm_stamp),
                self.llvm_identity,
            )
            self.assertFalse(paths.circt_stamp.exists())

    def test_distclean_revokes_before_delete_and_linked_preserves(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            main = build_paths(Path(td) / "main")
            self.ready_llvm(main)
            self.ready_circt(main)

            def fail_removal(*args, **kwargs):
                self.assertFalse(main.llvm_stamp.exists())
                self.assertFalse(main.circt_stamp.exists())
                self.assertFalse(main.or_tools_stamp.exists())
                raise RuntimeError("deletion interrupted")

            with (
                patch.object(
                    self.module.shutil,
                    "rmtree",
                    side_effect=fail_removal,
                ),
                self.assertRaisesRegex(RuntimeError, "deletion interrupted"),
            ):
                self.module.cmd_distclean(main, self.args)
            self.assertFalse(main.llvm_stamp.exists())
            self.assertFalse(main.circt_stamp.exists())
            self.assertFalse(main.or_tools_stamp.exists())

            linked = build_paths(Path(td) / "linked")
            linked.is_main = False
            self.ready_llvm(linked)
            self.ready_circt(linked)
            linked.loom_build.mkdir(parents=True, exist_ok=True)
            self.module.cmd_distclean(linked, self.args)
            self.assertTrue(linked.llvm_build.exists())
            self.assertTrue(linked.circt_build.exists())
            self.assertTrue(linked.or_tools_build.exists())
            self.assertTrue(linked.or_tools_install.exists())
            self.assertTrue(linked.llvm_stamp.exists())
            self.assertTrue(linked.circt_stamp.exists())
            self.assertTrue(linked.or_tools_stamp.exists())

    def test_clean_preserves_only_a_marked_external_tool_cache(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            paths.loom_build.mkdir(parents=True)
            cache = (
                paths.loom_build / self.module.EXTERNAL_TOOL_CACHE_DIRECTORY
            )
            cache.mkdir()
            (cache / EXTERNAL_TOOL_CACHE_MARKER).write_text(
                EXTERNAL_TOOL_CACHE_MARKER_CONTENTS
            )
            (cache / "entries").mkdir()
            (cache / "locks").mkdir()
            (cache / "entries" / "result").write_text("cached\n")
            (paths.loom_build / "build.ninja").write_text("generated\n")
            (paths.loom_build / "lib").mkdir()
            (paths.loom_build / "lib" / "generated.a").write_text("generated\n")

            self.module.cmd_clean(paths, self.args)

            self.assertTrue((cache / "entries" / "result").is_file())
            self.assertFalse((paths.loom_build / "build.ninja").exists())
            self.assertFalse((paths.loom_build / "lib").exists())

            paths.is_main = False
            self.module.cmd_distclean(paths, self.args)
            self.assertFalse(paths.loom_build.exists())

            paths.loom_build.mkdir()
            cache.mkdir()
            (cache / "foreign").write_text("not a Loom cache\n")
            self.module.cmd_clean(paths, self.args)
            self.assertFalse(cache.exists())

    def test_explicit_circt_build_uses_package_readiness_only(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            self.ready_llvm(paths)
            self.ready_circt(paths)
            calls = []

            def capture_run(cmd, **kwargs):
                calls.append(cmd)

            with self.build_environment(self.module, run=capture_run):
                self.module.cmd_build_circt(paths, self.args)

            self.assertEqual(
                calls,
                [
                    [
                        "cmake",
                        "--build",
                        str(paths.circt_build),
                        "-j1",
                    ]
                ],
            )
            self.assertEqual(
                self.module.available_circt_dir(
                    paths,
                    self.llvm_identity,
                    self.state.circt_commit,
                ),
                str(paths.circt_cmake_dir),
            )

            def fail_build(cmd, **kwargs):
                raise subprocess.CalledProcessError(1, cmd)

            with (
                self.build_environment(self.module, run=fail_build),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                self.module.cmd_build_circt(paths, self.args)
            self.assertFalse(paths.circt_stamp.exists())
            self.assertIsNone(
                self.module.available_circt_dir(
                    paths,
                    self.llvm_identity,
                    self.state.circt_commit,
                )
            )

    def test_explicit_or_tools_build_publishes_exact_package(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            self.ready_or_tools(paths)
            calls = []

            def capture_run(cmd, **kwargs):
                calls.append(cmd)

            with self.build_environment(self.module, run=capture_run):
                self.module.cmd_build_or_tools(paths, self.args)

            self.assertEqual(
                calls,
                [
                    [
                        "cmake",
                        "--build",
                        str(paths.or_tools_build),
                        "--target",
                        "install",
                        "-j1",
                    ]
                ],
            )
            self.assertEqual(
                self.module.available_or_tools_dir(
                    paths,
                    self.or_tools_identity,
                    self.state.or_tools_commit,
                ),
                str(paths.or_tools_cmake_dir),
            )

            (paths.or_tools_cmake_dir / "loom-source-commit.txt").write_text("foreign-commit\n")
            self.assertIsNone(
                self.module.available_or_tools_dir(
                    paths,
                    self.or_tools_identity,
                    self.state.or_tools_commit,
                )
            )

    def test_ordinary_build_repairs_and_routes_required_or_tools(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            self.ready_llvm(shared)
            shared.or_tools_stamp.unlink()
            shutil.rmtree(shared.or_tools_build)
            shutil.rmtree(shared.or_tools_install)
            consumer = self.loom_consumer(shared, Path(td) / "consumer", configured=False)
            offered = []
            commands = []

            def capture_configure(paths, circt_dir, or_tools_dir, or_tools_commit):
                offered.append((circt_dir, or_tools_dir))
                self.assertEqual(or_tools_commit, self.state.or_tools_commit)
                paths.loom_build.mkdir(parents=True)
                (paths.loom_build / "build.ninja").write_text("ninja\n")

            def materialize(cmd, **kwargs):
                commands.append(cmd)
                if cmd[:2] == ["cmake", "--build"] and (str(shared.or_tools_build) in cmd):
                    self.ready_or_tools(shared)

            with (
                self.build_environment(self.module, run=materialize),
                patch.object(
                    self.module,
                    "configure_or_tools",
                    side_effect=lambda *args: (
                        shared.or_tools_build.mkdir(parents=True),
                        (shared.or_tools_build / "build.ninja").write_text("ninja\n"),
                    ),
                ),
                patch.object(
                    self.module,
                    "configure_loom",
                    side_effect=capture_configure,
                ),
            ):
                self.module.cmd_build_loom(consumer, self.args)

            self.assertEqual(
                offered,
                [(None, str(shared.or_tools_cmake_dir))],
            )
            self.assertIn(
                [
                    "cmake",
                    "--build",
                    str(shared.or_tools_build),
                    "--target",
                    "install",
                    "-j1",
                ],
                commands,
            )

    def test_ordinary_build_routes_ready_circt_without_building_it(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            self.ready_llvm(shared)
            self.ready_circt(shared)
            consumer = self.loom_consumer(shared, Path(td) / "consumer", configured=False)
            offered = []
            commands = []

            def capture_configure(paths, circt_dir, or_tools_dir, or_tools_commit):
                offered.append((circt_dir, or_tools_dir))
                self.assertEqual(or_tools_commit, self.state.or_tools_commit)
                paths.loom_build.mkdir(parents=True)
                (paths.loom_build / "build.ninja").write_text("ninja\n")

            with (
                self.build_environment(
                    self.module,
                    run=lambda cmd, **kwargs: commands.append(cmd),
                ),
                patch.object(
                    self.module,
                    "configure_loom",
                    side_effect=capture_configure,
                ),
            ):
                self.module.cmd_build_loom(consumer, self.args)

            self.assertEqual(
                offered,
                [
                    (
                        str(shared.circt_cmake_dir),
                        str(shared.or_tools_cmake_dir),
                    )
                ],
            )
            self.assertEqual(
                commands,
                [
                    [
                        "cmake",
                        "--build",
                        str(consumer.loom_build),
                        "-j1",
                    ]
                ],
            )

    def test_blocking_lock_timeout_is_bounded_and_cold(self):
        context = multiprocessing.get_context("fork")
        held = context.Event()
        release = context.Event()
        temp_dir = tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT)
        self.addCleanup(temp_dir.cleanup)
        paths = build_paths(Path(temp_dir.name))

        def hold_writer():
            with self.module.SharedProductLock(
                paths.llvm_lock,
                paths.llvm_lock_turnstile,
                1.0,
                shared=False,
            ):
                held.set()
                release.wait(2.0)

        holder = context.Process(target=hold_writer, name="lock-holder")
        try:
            holder.start()
            self.assertTrue(held.wait(1.0))
            started = time.monotonic()
            cpu_started = time.process_time()
            stderr = io.StringIO()
            with (
                redirect_stderr(stderr),
                self.assertRaises(SystemExit),
                self.module.SharedProductLock(
                    paths.llvm_lock,
                    paths.llvm_lock_turnstile,
                    0.05,
                    shared=True,
                ),
            ):
                pass
            elapsed = time.monotonic() - started
            cpu_elapsed = time.process_time() - cpu_started
        finally:
            release.set()
            self.join_processes((holder,))

        self.assertEqual(holder.exitcode, 0)
        self.assertGreaterEqual(elapsed, 0.04)
        self.assertLess(elapsed, 0.20)
        self.assertLess(cpu_elapsed, 0.02)
        self.assertIn("timed out after 0.05s", stderr.getvalue())
        with self.module.SharedProductLock(
            paths.llvm_lock,
            paths.llvm_lock_turnstile,
            0.1,
            shared=True,
        ):
            pass

        self.assertEqual(self.module.validate_lock_timeout(0), 0)
        self.assertEqual(
            self.module.validate_lock_timeout(self.module.MAX_LOCK_TIMEOUT),
            self.module.MAX_LOCK_TIMEOUT,
        )
        with self.assertRaises(ValueError):
            self.module.validate_lock_timeout(float("nan"))
        with self.assertRaises(ValueError):
            self.module.validate_lock_timeout(float("inf"))
        with self.assertRaises(ValueError):
            self.module.validate_lock_timeout(self.module.MAX_LOCK_TIMEOUT + 0.1)

        cli = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--lock-timeout",
                "nan",
                "doctor",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(cli.returncode, 2)
        self.assertIn("--lock-timeout", cli.stderr)

    def test_completed_acquisitions_leave_bounded_turnstile_state(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            record_size = self.module._TURNSTILE_RECORD_SIZE
            acquisitions = 64
            sizes = set()
            for _ in range(acquisitions):
                with self.module.SharedProductLock(
                    paths.llvm_lock,
                    paths.llvm_lock_turnstile,
                    1.0,
                    shared=False,
                ):
                    pass
                sizes.add(paths.llvm_lock_turnstile.stat().st_size)

            # A completed acquisition leaves no participant behind, so the
            # turnstile keeps only its header and one reusable residency
            # slot however many acquisitions have run through it. Nothing
            # accumulates for a later acquisition to scan.
            self.assertEqual(sizes, {2 * record_size})

            self.module.cmd_distclean(paths, Namespace(jobs=1, lock_timeout=1.0))
            self.assertEqual(paths.llvm_lock_turnstile.stat().st_size, 2 * record_size)

            # The slot is genuinely reused rather than the protocol having
            # stopped tracking arrival order.
            issued = int(paths.llvm_lock_turnstile.read_bytes()[:16], 16)
            self.assertGreater(issued, acquisitions)

    def test_drained_cohort_leaves_constant_turnstile_state(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            record_size = self.module._TURNSTILE_RECORD_SIZE
            context = multiprocessing.get_context("fork")
            held = context.Event()
            release = context.Event()
            cohort = 32

            def hold_writer():
                with self.module.SharedProductLock(
                    paths.llvm_lock,
                    paths.llvm_lock_turnstile,
                    10.0,
                    shared=False,
                ):
                    held.set()
                    release.wait(10.0)

            def queue_writer():
                with self.module.SharedProductLock(
                    paths.llvm_lock,
                    paths.llvm_lock_turnstile,
                    20.0,
                    shared=False,
                ):
                    pass

            holder = context.Process(target=hold_writer, name="holder")
            queued = [context.Process(target=queue_writer, name=f"queued-{index}") for index in range(cohort)]
            try:
                holder.start()
                self.assertTrue(held.wait(2.0))
                for index, process in enumerate(queued):
                    process.start()
                    self.wait_for_turnstile_records(paths.llvm_lock_turnstile, index + 2)
                peak = paths.llvm_lock_turnstile.stat().st_size
                release.set()
                self.join_processes(queued)
            finally:
                release.set()
                self.join_processes((holder, *queued))

            self.assertEqual(holder.exitcode, 0)
            self.assertEqual({process.exitcode for process in queued}, {0})
            self.assertEqual(peak, (cohort + 1) * record_size)
            # The whole cohort drained normally and nothing has acquired
            # since. What the burst leaves behind must be the header and the
            # single record of the participant that left last, never the peak
            # the turnstile once carried: residency outlives no participant,
            # so the next acquisition scans a constant table.
            self.assertEqual(paths.llvm_lock_turnstile.stat().st_size, 2 * record_size)

    def test_crashed_participants_are_reclaimed_without_replay(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            record_size = self.module._TURNSTILE_RECORD_SIZE
            context = multiprocessing.get_context("fork")
            held = context.Event()
            release = context.Event()
            order = context.Queue()

            def hold_writer():
                with self.module.SharedProductLock(
                    paths.llvm_lock,
                    paths.llvm_lock_turnstile,
                    10.0,
                    shared=False,
                ):
                    held.set()
                    release.wait(10.0)

            def queue_writer(name):
                with self.module.SharedProductLock(
                    paths.llvm_lock,
                    paths.llvm_lock_turnstile,
                    20.0,
                    shared=False,
                ):
                    order.put(name)

            holder = context.Process(target=hold_writer, name="holder")
            queued = [
                context.Process(
                    target=queue_writer,
                    args=(f"queued-{index}",),
                    name=f"queued-{index}",
                )
                for index in range(64)
            ]
            newcomer = context.Process(target=queue_writer, args=("newcomer",), name="newcomer")
            retained = queued[-1]
            try:
                holder.start()
                self.assertTrue(held.wait(1.0))
                for index, process in enumerate(queued):
                    process.start()
                    # Queued participants are the only residency the
                    # turnstile carries: one record each, plus the header.
                    self.wait_for_turnstile_records(paths.llvm_lock_turnstile, index + 2)

                # Leave only the participant holding the last record, so
                # every reclaimable record sits below a live one.
                for process in queued[:-1]:
                    process.kill()
                self.join_processes(queued[:-1])

                newcomer.start()
                # Registering rewrites the table as the live tickets alone,
                # so a sparse live set cannot preserve peak-concurrency
                # state for a later acquisition to scan.
                self.wait_for_turnstile_records(paths.llvm_lock_turnstile, 3)
                self.assertTrue(retained.is_alive(), "compaction dropped a live ticket")

                release.set()
                # Compaction moves records but never reorders tickets.
                self.assertEqual(order.get(timeout=5.0), retained.name)
                self.assertEqual(order.get(timeout=5.0), "newcomer")
            finally:
                release.set()
                self.join_processes((holder, newcomer, *queued))
                order.close()
                order.join_thread()

            self.assertEqual(
                (holder.exitcode, retained.exitcode, newcomer.exitcode),
                (0, 0, 0),
            )
            # A participant that died while queued is skipped rather than
            # waited on, and its record is reclaimed by the next arrival.
            with self.module.SharedProductLock(
                paths.llvm_lock,
                paths.llvm_lock_turnstile,
                0.5,
                shared=False,
            ):
                pass
            self.assertEqual(paths.llvm_lock_turnstile.stat().st_size, 2 * record_size)

    def test_dirty_or_drifted_dependency_pins_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            paths = self.module.Paths(topology.main)

            # A tracked modification in a shared dependency checkout is a
            # divergence from the pinned upstream commit, not a build input.
            (paths.circt_root / "circt.txt").write_text("dirty\n")
            dirty = self.capture_die(self.module.check_dependency_pins, paths)
            self.assertIn("tracked modifications", dirty)
            git(paths.circt_root, "checkout", "--", "circt.txt")

            (paths.or_tools_root / "or-tools.txt").write_text("dirty\n")
            dirty = self.capture_die(self.module.check_dependency_pins, paths)
            self.assertIn("externals/or-tools", dirty)
            self.assertIn("tracked modifications", dirty)
            self.assertIn("update the corresponding parent dependency gitlink", dirty)
            git(paths.or_tools_root, "checkout", "--", "or-tools.txt")

            # A clean shared checkout moved off the parent gitlink is drift.
            llvm_other = git(topology.llvm_repo, "rev-parse", "HEAD")
            git(paths.llvm_root, "checkout", "-q", llvm_other)
            drift = self.capture_die(self.module.check_dependency_pins, paths)
            self.assertIn("checkout drift", drift)
            self.assertIn(llvm_other, drift)

    def test_uninitialized_submodule_contracts_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            main_paths = self.module.Paths(topology.main)

            # A linked worktree must not initialize any submodule; it
            # consumes the primary-owned shared checkouts instead.
            linked_paths = self.module.Paths(topology.linked)
            git(
                topology.linked,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "update",
                "--init",
                "--",
                "externals/llvm",
            )
            hygiene = self.capture_die(self.module.check_linked_submodule_hygiene, linked_paths)
            self.assertIn("externals/llvm", hygiene)
            self.assertIn("must not initialize submodules", hygiene)

            # The CIRCT nested LLVM submodule must stay uninitialized so the
            # shared sibling externals/llvm is the only LLVM in play.
            git(
                main_paths.circt_root,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "update",
                "--init",
                "--",
                "llvm",
            )
            nested = self.capture_die(self.module.check_dependency_pins, main_paths)
            self.assertIn("must remain uninitialized", nested)

    def test_incompatible_parent_gitlinks_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            paths = self.module.Paths(topology.linked)
            # externals/llvm is repinned in the invoking worktree to a commit
            # the pinned CIRCT's nested LLVM does not agree with. The shared
            # LLVM checkout is moved to match, so this is a genuine
            # parent-gitlink inconsistency rather than checkout drift.
            llvm_other = git(topology.llvm_repo, "rev-parse", "HEAD")
            git(
                topology.linked,
                "update-index",
                "--cacheinfo",
                "160000",
                llvm_other,
                "externals/llvm",
            )
            git(topology.linked, "commit", "-qm", "Repoint externals/llvm")
            git(paths.llvm_root, "checkout", "-q", llvm_other)

            error = self.capture_die(self.module.check_dependency_pins, paths)
            self.assertIn("parent gitlinks", error)
            self.assertIn("atomically", error)

    def test_compiler_floor_and_ccache_resolution(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            root = Path(td)

            # Both public compiler gates reject the second compiler when it
            # falls just below the configured family floor.
            old = root / "old"
            old.mkdir()
            for tool, version in (
                ("gcc", "7.4.0"),
                ("g++", "7.3.0"),
                ("clang", "21.1.8"),
                ("clang++", "21.1.7"),
            ):
                compiler = old / tool
                compiler.write_text(f"#!/bin/sh\necho '{tool} (fake) {version}'\n")
                compiler.chmod(0o755)
            old_path = f"{old}{os.pathsep}{os.environ['PATH']}"
            llvm_floor = io.StringIO()
            with (
                patch.dict(os.environ, {"PATH": old_path}),
                redirect_stderr(llvm_floor),
                self.assertRaises(SystemExit),
            ):
                self.module.check_llvm_compilers()
            self.assertIn("must be at least 7.4", llvm_floor.getvalue())
            loom_floor = io.StringIO()
            with (
                patch.dict(os.environ, {"PATH": old_path}),
                redirect_stderr(loom_floor),
                self.assertRaises(SystemExit),
            ):
                self.module.check_loom_compilers()
            self.assertIn("must be at least 21.1.8", loom_floor.getvalue())

            # A ccache launcher symlink is skipped in favour of the real
            # compiler behind it on PATH.
            ccache_bin = root / "ccache-bin"
            real_bin = root / "real-bin"
            ccache_bin.mkdir()
            real_bin.mkdir()
            ccache = ccache_bin / "ccache"
            ccache.write_text("#!/bin/sh\nexit 0\n")
            ccache.chmod(0o755)
            (ccache_bin / "gcc").symlink_to(ccache)
            real_gcc = real_bin / "gcc"
            real_gcc.write_text("#!/bin/sh\nexit 0\n")
            real_gcc.chmod(0o755)
            with patch.dict(os.environ, {"PATH": f"{ccache_bin}{os.pathsep}{real_bin}"}):
                resolved = self.module.resolve_compiler_executable("gcc")
            self.assertEqual(Path(resolved).resolve(), real_gcc.resolve())

            # A language-driver symlink is semantically significant. Resolving
            # clang++ to the clang binary would silently switch C++ links to
            # C driver mode and omit the C++ runtime.
            clang_bin = root / "clang-bin"
            clang_bin.mkdir()
            clang = clang_bin / "clang"
            clang.write_text("#!/bin/sh\nexit 0\n")
            clang.chmod(0o755)
            clang_cxx = clang_bin / "clang++"
            clang_cxx.symlink_to(clang)
            with patch.dict(os.environ, {"PATH": str(clang_bin)}):
                resolved = self.module.resolve_compiler_executable("clang++")
            self.assertEqual(Path(resolved), clang_cxx)

    def test_explicit_circt_dir_is_exact_package_directory(self):
        cmake = shutil.which("cmake")
        self.assertIsNotNone(cmake)
        ninja = shutil.which("ninja")
        self.assertIsNotNone(ninja)
        paths = self.module.Paths(REPO_ROOT)
        packages = (
            ("MLIR", paths.mlir_dir),
            ("LLVM", paths.cmake_llvm_dir),
            ("CIRCT", paths.circt_cmake_dir),
        )
        missing = [
            f"{name}_DIR={directory}"
            for name, directory in packages
            if not (directory / f"{name}Config.cmake").is_file()
        ]
        if missing:
            self.skipTest("shared build products absent: " + ", ".join(missing))

        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(circt_consumer LANGUAGES C CXX)\n"
                "find_package(CIRCT REQUIRED CONFIG NO_DEFAULT_PATH)\n"
                "if(NOT TARGET CIRCTSupport)\n"
                '  message(FATAL_ERROR "CIRCTSupport target missing")\n'
                "endif()\n"
                'message(STATUS "Using CIRCTConfig.cmake in: '
                '${CIRCT_DIR}")\n'
                "add_executable(circt-consumer main.cpp)\n"
                "target_link_libraries(circt-consumer PRIVATE CIRCTSupport)\n"
            )
            (source / "main.cpp").write_text("int main() { return 0; }\n")

            def configure(circt_dir, label):
                return subprocess.run(
                    [
                        cmake,
                        "-S",
                        str(source),
                        "-B",
                        str(root / label),
                        "-G",
                        "Ninja",
                        "-DCMAKE_BUILD_TYPE=Release",
                        f"-DMLIR_DIR={paths.mlir_dir}",
                        f"-DLLVM_DIR={paths.cmake_llvm_dir}",
                        f"-DCIRCT_DIR={circt_dir}",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

            # A parent of the package directory must not satisfy CIRCT_DIR:
            # find_package refuses it instead of descending into lib/cmake.
            parent = configure(paths.circt_build, "parent")
            self.assertNotEqual(parent.returncode, 0, parent.stdout)
            self.assertIn('provided by "CIRCT"', parent.stdout)

            # The exact package directory must configure and expose a linkable
            # imported target to an external consumer.
            exact = configure(paths.circt_cmake_dir, "exact")
            self.assertEqual(exact.returncode, 0, exact.stdout)
            self.assertIn(
                f"Using CIRCTConfig.cmake in: {paths.circt_cmake_dir}",
                exact.stdout,
            )
            self.assertNotIn('provided by "CIRCT"', exact.stdout)
            build = subprocess.run(
                [
                    cmake,
                    "--build",
                    str(root / "exact"),
                    "--parallel",
                    "1",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(build.returncode, 0, build.stdout)
            self.assertTrue((root / "exact" / "circt-consumer").is_file())

    def test_installed_or_tools_package_solves_cp_sat_model(self):
        cmake = shutil.which("cmake")
        self.assertIsNotNone(cmake)
        ninja = shutil.which("ninja")
        self.assertIsNotNone(ninja)
        paths = self.module.Paths(REPO_ROOT)
        config = paths.or_tools_cmake_dir / "ortoolsConfig.cmake"
        if not config.is_file():
            self.skipTest(f"shared OR-Tools package absent: {config}")

        expected_commit = git(REPO_ROOT, "rev-parse", "HEAD:externals/or-tools")
        self.assertEqual(
            self.module.read_stamp(self.module.or_tools_commit_projection(paths)),
            expected_commit,
        )

        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\n"
                "project(ortools_consumer LANGUAGES CXX)\n"
                "find_package(ortools REQUIRED CONFIG NO_DEFAULT_PATH)\n"
                "if(NOT TARGET ortools::ortools)\n"
                '  message(FATAL_ERROR "ortools target missing")\n'
                "endif()\n"
                "add_executable(cp-sat-probe main.cpp)\n"
                "target_link_libraries(cp-sat-probe PRIVATE "
                "ortools::ortools)\n"
            )
            (source / "main.cpp").write_text(
                '#include "ortools/sat/cp_model.h"\n'
                "int main() {\n"
                "  using namespace operations_research::sat;\n"
                "  CpModelBuilder model;\n"
                "  const IntVar x = model.NewIntVar(\n"
                "      operations_research::Domain(0, 10));\n"
                "  model.AddEquality(x, 7);\n"
                "  const CpSolverResponse response = Solve(model.Build());\n"
                "  return SolutionIntegerValue(response, x) == 7 ? 0 : 1;\n"
                "}\n"
            )

            def configure(or_tools_dir, label):
                return subprocess.run(
                    [
                        cmake,
                        "-S",
                        str(source),
                        "-B",
                        str(root / label),
                        "-G",
                        "Ninja",
                        f"-Dortools_DIR={or_tools_dir}",
                        f"-DCMAKE_PREFIX_PATH={paths.or_tools_install}",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

            parent = configure(paths.or_tools_install, "parent")
            self.assertNotEqual(parent.returncode, 0, parent.stdout)

            exact = configure(paths.or_tools_cmake_dir, "exact")
            self.assertEqual(exact.returncode, 0, exact.stdout)
            build = subprocess.run(
                [cmake, "--build", str(root / "exact"), "--parallel", "1"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(build.returncode, 0, build.stdout)
            probe = subprocess.run(
                [str(root / "exact" / "cp-sat-probe")],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(probe.returncode, 0, probe.stdout)


if __name__ == "__main__":
    unittest.main()
