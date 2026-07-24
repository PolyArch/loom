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
        (self.circt_repo / ".gitmodules").write_text(
            "[submodule \"llvm\"]\n"
            "\tpath = llvm\n"
            f"\turl = {self.llvm_repo}\n"
        )
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

        self.main = root / "main"
        init_repo(self.main)
        for source, path in (
            (self.circt_repo, "externals/circt"),
            (self.llvm_repo, "externals/llvm"),
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
        llvm_lit=llvm_build / "bin" / "llvm-lit",
        circt_root=circt_root,
        circt_build=circt_build,
        circt_stamp=externals / ".loom-build.circt.stamp",
        circt_cmake_dir=circt_build / "lib" / "cmake" / "circt",
        loom_build=root / "build",
    )


class MakeWorktreeTest(unittest.TestCase):
    def setUp(self):
        self.module = load_dispatcher()
        self.state = self.module.DependencyState("circt-pin", "llvm-pin")
        self.compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        self.args = Namespace(jobs=1, lock_timeout=1.0)
        self.llvm_identity = self.module.llvm_build_identity(
            self.state.llvm_commit, self.compilers
        )
        self.circt_identity = self.module.circt_build_identity(
            self.llvm_identity, self.state.circt_commit
        )
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    def build_environment(self, module, run=UNSET):
        stack = ExitStack()
        stack.enter_context(patch.object(module, "check_git_version"))
        stack.enter_context(patch.object(module, "check_loom_compilers"))
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
        ):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text("ready\n")
        (paths.llvm_build / "build.ninja").write_text("ninja\n")
        paths.llvm_lit.parent.mkdir(parents=True, exist_ok=True)
        paths.llvm_lit.write_text("#!/bin/sh\nexit 0\n")
        paths.llvm_lit.chmod(0o755)

    def ready_llvm(self, paths, stamp: str | None = None) -> None:
        self.write_llvm_artifacts(paths)
        paths.llvm_stamp.parent.mkdir(parents=True, exist_ok=True)
        paths.llvm_stamp.write_text((stamp or self.llvm_identity) + "\n")

    def ready_circt(self, paths, stamp: str | None = None) -> None:
        paths.circt_cmake_dir.mkdir(parents=True, exist_ok=True)
        (paths.circt_cmake_dir / "CIRCTConfig.cmake").write_text("ready\n")
        (paths.circt_build / "build.ninja").write_text("ninja\n")
        paths.circt_stamp.write_text((stamp or self.circt_identity) + "\n")

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

    def wait_for_flock_holders(
        self,
        path: Path,
        expected_pids: set[int],
        timeout: float = 1.0,
    ) -> None:
        stat = path.stat()
        key = (
            f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:"
            f"{stat.st_ino}"
        )
        deadline = time.monotonic() + timeout
        while True:
            holders = set()
            for line in Path("/proc/locks").read_text().splitlines():
                fields = line.split()
                if (
                    len(fields) >= 6
                    and fields[1] == "FLOCK"
                    and fields[3] == "READ"
                    and fields[5] == key
                ):
                    holders.add(int(fields[4]))
            if expected_pids <= holders:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.fail(
                    f"flock holders {holders} did not include "
                    f"{expected_pids}"
                )
            select.select((), (), (), min(0.005, remaining))

    def test_linked_worktree_routes_to_primary_artifact_owner(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            paths = self.module.Paths(topology.linked)

            self.assertEqual(paths.main, topology.main)
            self.assertEqual(paths.externals_root, topology.main / "externals")
            self.assertEqual(
                self.module.gitlinks_at_head(topology.linked),
                ("externals/circt", "externals/llvm"),
            )
            linked_status = git(topology.linked, "submodule", "status")
            self.assertTrue(
                all(line.startswith("-") for line in linked_status.splitlines())
            )
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

    def test_primary_owner_runtime_state_uses_canonical_ignore_policy(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            (topology.main / ".gitignore").write_text(
                (REPO_ROOT / ".gitignore").read_text()
            )
            runtime_state = (
                "externals/.loom-build.llvm.lock",
                "externals/.loom-build.llvm.turnstile.lock",
                "externals/.loom-build.llvm.stamp",
                "externals/.loom-build.circt.stamp",
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
        same_llvm = self.module.llvm_build_identity(
            self.state.llvm_commit, self.compilers
        )
        changed_circt = self.module.DependencyState(
            "other-circt", self.state.llvm_commit
        )
        self.assertEqual(
            same_llvm,
            self.module.llvm_build_identity(
                changed_circt.llvm_commit, self.compilers
            ),
        )
        self.assertNotEqual(
            self.module.circt_build_identity(
                same_llvm, self.state.circt_commit
            ),
            self.module.circt_build_identity(
                same_llvm, changed_circt.circt_commit
            ),
        )

        payload = json.loads(same_llvm)
        self.assertEqual(payload["dependencies"], {"llvm": "llvm-pin"})
        circt_payload = json.loads(self.circt_identity)
        self.assertNotIn("circt_build_targets", circt_payload)

        legacy_payload = json.loads(same_llvm)
        legacy_payload["dependencies"]["circt"] = "old-circt"
        legacy_stamp = json.dumps(legacy_payload, sort_keys=True)
        self.assertTrue(
            self.module.llvm_stamp_matches(legacy_stamp, same_llvm)
        )
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
                [[
                    "cmake",
                    "--build",
                    str(consumer.loom_build),
                    "-j1",
                ]],
            )
            self.assertEqual(
                self.module.read_stamp(shared.llvm_stamp), legacy_stamp
            )

    def test_ready_public_loom_readers_overlap(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            self.ready_llvm(shared)
            readers = {
                name: self.loom_consumer(shared, Path(td) / name)
                for name in ("reader-a", "reader-b")
            }
            context = multiprocessing.get_context("fork")
            active = {name: context.Event() for name in readers}
            release = {name: context.Event() for name in readers}
            by_build = {
                str(paths.loom_build): name for name, paths in readers.items()
            }

            def controlled_run(cmd, **kwargs):
                name = by_build.get(cmd[2]) if cmd[:2] == ["cmake", "--build"] else None
                if name is None:
                    raise AssertionError(f"unexpected command: {cmd}")
                active[name].set()
                if not release[name].wait(2.0):
                    raise RuntimeError(f"{name} release timed out")

            def consume(name):
                self.module.cmd_build_loom(readers[name], self.args)

            processes = {
                name: context.Process(target=consume, args=(name,), name=name)
                for name in readers
            }
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

    def test_queued_public_writers_precede_late_reader(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            self.ready_llvm(shared)
            readers = {
                name: self.loom_consumer(shared, Path(td) / name)
                for name in ("holding-reader", "late-reader")
            }
            context = multiprocessing.get_context("fork")
            names = (
                "holding-reader",
                "writer-a",
                "writer-b",
                "late-reader",
            )
            active = {name: context.Event() for name in names}
            release = {name: context.Event() for name in names}
            started = {
                name: context.Event()
                for name in ("writer-a", "writer-b", "late-reader")
            }
            order = context.Queue()
            by_build = {
                str(paths.loom_build): name for name, paths in readers.items()
            }

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
                if name == "late-reader":
                    started[name].set()
                self.module.cmd_build_loom(readers[name], self.args)

            def write(name):
                started[name].set()
                self.module.cmd_build_llvm(shared, self.args)

            processes = {
                "holding-reader": context.Process(
                    target=consume,
                    args=("holding-reader",),
                    name="holding-reader",
                ),
                "late-reader": context.Process(
                    target=consume,
                    args=("late-reader",),
                    name="late-reader",
                ),
                "writer-a": context.Process(
                    target=write,
                    args=("writer-a",),
                    name="writer-a",
                ),
                "writer-b": context.Process(
                    target=write,
                    args=("writer-b",),
                    name="writer-b",
                ),
            }
            try:
                with self.build_environment(self.module, run=controlled_run):
                    processes["holding-reader"].start()
                    self.assertTrue(active["holding-reader"].wait(1.0))
                    self.assertEqual(order.get(timeout=1.0), "holding-reader")
                    for name in ("writer-a", "writer-b", "late-reader"):
                        if name == "late-reader":
                            self.wait_for_flock_holders(
                                shared.llvm_lock_turnstile,
                                {
                                    processes["writer-a"].pid,
                                    processes["writer-b"].pid,
                                },
                            )
                        processes[name].start()
                        self.assertTrue(started[name].wait(0.5))

                    release["holding-reader"].set()
                    first = order.get(timeout=1.0)
                    self.assertIn(first, {"writer-a", "writer-b"})
                    release[first].set()
                    second = order.get(timeout=1.0)
                    self.assertEqual(
                        {first, second}, {"writer-a", "writer-b"}
                    )
                    release[second].set()
                    self.assertEqual(
                        order.get(timeout=1.0), "late-reader"
                    )
                    release["late-reader"].set()
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
                    self.wait_for_flock_holders(
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

    def test_dispatcher_death_contains_build_process_group(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            root = Path(td)
            shared = build_paths(root / "shared")
            self.ready_llvm(shared)
            consumer = self.loom_consumer(shared, root / "consumer")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_cmake = fake_bin / "cmake"
            fake_cmake.write_text(
                "#!/usr/bin/env python3\n"
                "import os\n"
                "import signal\n"
                "import socket\n"
                "child = os.fork()\n"
                "if child == 0:\n"
                "    signal.pause()\n"
                "    os._exit(0)\n"
                "with socket.socket() as channel:\n"
                "    channel.connect(('127.0.0.1', "
                "int(os.environ['LOOM_TEST_PORT'])))\n"
                "    channel.sendall(f'{os.getpid()} {child}\\n'.encode())\n"
                "signal.pause()\n"
            )
            fake_cmake.chmod(0o755)
            server = socket.socket()
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(1.0)
            server_port = server.getsockname()[1]
            context = multiprocessing.get_context("fork")

            def run_dispatcher():
                os.environ["PATH"] = (
                    f"{fake_bin}{os.pathsep}{os.environ['PATH']}"
                )
                os.environ["LOOM_TEST_PORT"] = str(server_port)
                self.module.cmd_build_loom(consumer, self.args)

            dispatcher = context.Process(
                target=run_dispatcher, name="loom-dispatcher"
            )
            child_pids = ()
            child_pidfds = ()

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
                        child_pids = tuple(
                            int(pid)
                            for pid in connection.recv(128).decode().split()
                        )
                    self.assertEqual(len(child_pids), 2)
                    child_pidfds = tuple(
                        os.pidfd_open(pid) for pid in child_pids
                    )
                    self.assertEqual(
                        len({os.getpgid(pid) for pid in child_pids}), 1
                    )
                    lock_files = {
                        (path.stat().st_dev, path.stat().st_ino)
                        for path in (
                            shared.llvm_lock,
                            shared.llvm_lock_turnstile,
                        )
                    }
                    for pid in child_pids:
                        inherited = set()
                        for fd in Path(f"/proc/{pid}/fd").iterdir():
                            try:
                                stat = fd.stat()
                            except FileNotFoundError:
                                continue
                            inherited.add((stat.st_dev, stat.st_ino))
                        self.assertTrue(lock_files.isdisjoint(inherited))
                    dispatcher.kill()
                    dispatcher.join(1.0)
                    self.assertFalse(dispatcher.is_alive())

                    pending = set(child_pidfds)
                    deadline = time.monotonic() + 1.0
                    while pending:
                        ready, _, _ = select.select(
                            tuple(pending),
                            (),
                            (),
                            max(0, deadline - time.monotonic()),
                        )
                        if not ready:
                            break
                        pending.difference_update(ready)
                    self.assertFalse(
                        pending,
                        "build process group survived dispatcher death",
                    )
            finally:
                if dispatcher.is_alive():
                    dispatcher.kill()
                    dispatcher.join(1.0)
                for pid in child_pids:
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
                self.assertRaisesRegex(
                    RuntimeError, "injected configure failure"
                ),
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
                self.assertRaisesRegex(
                    RuntimeError, "LLVM preflight failed"
                ),
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
                self.assertRaisesRegex(
                    RuntimeError, "CIRCT preflight failed"
                ),
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
                self.build_environment(
                    self.module, run=lambda *args, **kwargs: None
                ),
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
                raise RuntimeError("deletion interrupted")

            with (
                patch.object(
                    self.module.shutil,
                    "rmtree",
                    side_effect=fail_removal,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "deletion interrupted"
                ),
            ):
                self.module.cmd_distclean(main, self.args)
            self.assertFalse(main.llvm_stamp.exists())
            self.assertFalse(main.circt_stamp.exists())

            linked = build_paths(Path(td) / "linked")
            linked.is_main = False
            self.ready_llvm(linked)
            self.ready_circt(linked)
            linked.loom_build.mkdir(parents=True, exist_ok=True)
            self.module.cmd_distclean(linked, self.args)
            self.assertTrue(linked.llvm_build.exists())
            self.assertTrue(linked.circt_build.exists())
            self.assertTrue(linked.llvm_stamp.exists())
            self.assertTrue(linked.circt_stamp.exists())

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
                [[
                    "cmake",
                    "--build",
                    str(paths.circt_build),
                    "-j1",
                ]],
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

    def test_ordinary_build_routes_ready_circt_without_building_it(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            self.ready_llvm(shared)
            self.ready_circt(shared)
            consumer = self.loom_consumer(
                shared, Path(td) / "consumer", configured=False
            )
            offered = []
            commands = []

            def capture_configure(paths, circt_dir):
                offered.append(circt_dir)
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

            self.assertEqual(offered, [str(shared.circt_cmake_dir)])
            self.assertEqual(
                commands,
                [[
                    "cmake",
                    "--build",
                    str(consumer.loom_build),
                    "-j1",
                ]],
            )

    def test_blocking_lock_timeout_is_bounded_and_cold(self):
        context = multiprocessing.get_context("fork")
        held = context.Event()
        release = context.Event()
        paths = build_paths(REPO_TEMP_ROOT / "lock-timeout")

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

        self.assertEqual(self.module.validate_lock_timeout(0), 0)
        self.assertEqual(
            self.module.validate_lock_timeout(
                self.module.MAX_LOCK_TIMEOUT
            ),
            self.module.MAX_LOCK_TIMEOUT,
        )
        with self.assertRaises(ValueError):
            self.module.validate_lock_timeout(float("nan"))
        with self.assertRaises(ValueError):
            self.module.validate_lock_timeout(float("inf"))
        with self.assertRaises(ValueError):
            self.module.validate_lock_timeout(
                self.module.MAX_LOCK_TIMEOUT + 0.1
            )

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

    def test_explicit_circt_dir_is_exact_package_directory(self):
        cmake = shutil.which("cmake")
        self.assertIsNotNone(cmake)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            root = Path(td)
            repo_root = root / "repo"
            repo_root.mkdir()
            shutil.copy(REPO_ROOT / "CMakeLists.txt", repo_root / "CMakeLists.txt")
            for subdir in ("include", "lib", "tools", "test"):
                directory = repo_root / subdir
                directory.mkdir()
                (directory / "CMakeLists.txt").write_text("")

            build_root = root / "packages"
            for name, subdir in (
                ("LLVM", "llvm"),
                ("MLIR", "mlir"),
                ("Clang", "clang"),
                ("CIRCT", "circt"),
            ):
                config_dir = build_root / "lib" / "cmake" / subdir
                config_dir.mkdir(parents=True)
                (config_dir / f"{name}Config.cmake").write_text(
                    f"set({name}_FOUND TRUE)\n"
                )
            circt_package = build_root / "lib" / "cmake" / "circt"

            def configure(circt_dir):
                return subprocess.run(
                    [
                        cmake,
                        "-S",
                        str(repo_root),
                        "-B",
                        str(root / f"build-{circt_dir.name}"),
                        f"-DCIRCT_DIR={circt_dir}",
                        f"-DCMAKE_PREFIX_PATH={build_root}",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

            nested = configure(build_root)
            self.assertNotEqual(nested.returncode, 0, nested.stdout)
            self.assertIn('provided by "CIRCT"', nested.stdout)

            exact = configure(circt_package)
            self.assertIn(
                f"Using CIRCTConfig.cmake in: {circt_package}",
                exact.stdout,
            )
            self.assertNotIn('provided by "CIRCT"', exact.stdout)


if __name__ == "__main__":
    unittest.main()
