#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("make-worktree.py")
REPO_TEMP_ROOT = SCRIPT.parents[1] / "build" / "test-runs"


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
        self.llvm_other = commit_file(self.llvm_repo, "llvm.txt", "llvm-b\n")

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

        self.cmsis_repo = root / "cmsis-repo"
        init_repo(self.cmsis_repo)
        (self.cmsis_repo / "Source").mkdir()
        commit_file(
            self.cmsis_repo, "Source/kernel.c", "int kernel(void) { return 0; }\n"
        )

        self.main = root / "super"
        init_repo(self.main)
        git(
            self.main,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(self.circt_repo),
            "externals/circt",
        )
        git(
            self.main,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(self.llvm_repo),
            "externals/llvm",
        )
        git(
            self.main,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(self.cmsis_repo),
            "externals/cmsis",
        )
        git(self.main / "externals/llvm", "checkout", "-q", self.llvm_pin)
        git(
            self.main,
            "add",
            ".gitmodules",
            "externals/circt",
            "externals/llvm",
            "externals/cmsis",
        )
        git(self.main, "commit", "-qm", "Pin dependencies")

        self.linked = root / "linked"
        git(self.main, "worktree", "add", "-q", "-b", "linked", str(self.linked))

    def conflict(self, worktree: Path, path: str) -> None:
        git(worktree, "update-index", "--force-remove", path)
        entries = (
            f"160000 {self.circt_pin} 1\t{path}\n"
            f"160000 {self.llvm_pin} 2\t{path}\n"
            f"160000 {self.llvm_other} 3\t{path}\n"
        )
        git(worktree, "update-index", "--index-info", input_text=entries)

    def reset_index(self, worktree: Path, path: str) -> None:
        git(worktree, "reset", "-q", "HEAD", "--", path)


def build_paths(root: Path):
    llvm_root = root / "llvm"
    llvm_build = llvm_root / "build"
    circt_root = root / "externals" / "circt"
    circt_build = circt_root / "build"
    return SimpleNamespace(
        root=root,
        main=root,
        is_main=True,
        externals_root=root / "externals",
        llvm_root=llvm_root,
        llvm_src=llvm_root / "llvm",
        llvm_build=llvm_build,
        llvm_lock=root / ".llvm.lock",
        llvm_stamp=root / ".llvm.stamp",
        mlir_dir=llvm_build / "lib" / "cmake" / "mlir",
        cmake_llvm_dir=llvm_build / "lib" / "cmake" / "llvm",
        cmake_clang_dir=llvm_build / "lib" / "cmake" / "clang",
        llvm_lit=llvm_build / "bin" / "llvm-lit",
        circt_root=circt_root,
        circt_build=circt_build,
        circt_stamp=root / "externals" / ".loom-build.circt.stamp",
        circt_cmake_dir=circt_build / "lib" / "cmake" / "circt",
        circt_required_lib=circt_build / "lib" / "libCIRCTExportVerilog.a",
        loom_build=root / "loom-build",
    )


class MakeWorktreeTest(unittest.TestCase):
    def setUp(self):
        self.module = load_dispatcher()
        self.args = Namespace(jobs=1, lock_timeout=1.0)

    def gate_error(self, paths) -> str:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.module.check_dependency_pins(paths)
        return stderr.getvalue()

    def hygiene_error(self, paths) -> str:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.module.check_linked_submodule_hygiene(paths)
        return stderr.getvalue()

    def test_primary_worktree_resolution_fails_outside_git_topology(self):
        with tempfile.TemporaryDirectory() as td:
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                self.module.resolve_main_worktree(Path(td))
            self.assertIn("could not resolve primary worktree", stderr.getvalue())

    def test_linked_worktree_uses_primary_externals_without_submodules(self):
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            paths = self.module.Paths(topology.linked)

            self.assertEqual(paths.externals_root, topology.main / "externals")
            self.module.check_linked_submodule_hygiene(paths)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.module.cmd_externals_root(paths, self.args)
            self.assertEqual(
                stdout.getvalue().strip(), str(topology.main / "externals")
            )

            git(
                topology.linked,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "update",
                "--init",
                "--",
                "externals/cmsis",
            )
            error = self.hygiene_error(paths)
            self.assertIn("externals/cmsis", error)
            self.assertIn("must not initialize submodules", error)
            self.assertIn(str(topology.main / "externals"), error)
            self.assertNotIn("submodule deinit", error)
            self.assertIn("externals/cmsis", self.gate_error(paths))

            shutil.rmtree(topology.linked / "externals" / "cmsis")
            residual_error = self.hygiene_error(paths)
            self.assertIn("administrative state", residual_error)

    def test_dependency_gate_with_real_linked_worktree(self):
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            paths = self.module.Paths(topology.linked)
            expected = self.module.DependencyState(
                topology.circt_pin, topology.llvm_pin
            )

            for repo in (paths.circt_root, paths.llvm_root):
                output = repo / "build" / "artifact.o"
                output.parent.mkdir()
                output.write_text("untracked\n")
            self.assertEqual(self.module.check_dependency_pins(paths), expected)

            topology.conflict(topology.main, "externals/circt")
            self.assertEqual(self.module.check_dependency_pins(paths), expected)
            topology.reset_index(topology.main, "externals/circt")

            topology.conflict(topology.linked, "externals/circt")
            self.assertIn("unmerged gitlink", self.gate_error(paths))
            topology.reset_index(topology.linked, "externals/circt")

            git(paths.llvm_root, "checkout", "-q", topology.llvm_other)
            self.assertIn("checkout drift", self.gate_error(paths))
            git(paths.llvm_root, "checkout", "-q", topology.llvm_pin)

            (paths.circt_root / "circt.txt").write_text("dirty\n")
            dirty_error = self.gate_error(paths)
            self.assertIn("tracked modifications", dirty_error)
            self.assertNotIn("restore or commit", dirty_error)
            self.assertIn("upstream commit", dirty_error)
            git(paths.circt_root, "checkout", "--", "circt.txt")

            (paths.llvm_root / "llvm.txt").write_text("dirty\n")
            git(paths.llvm_root, "add", "llvm.txt")
            self.assertIn("tracked modifications", self.gate_error(paths))
            git(paths.llvm_root, "reset", "-q", "HEAD", "--", "llvm.txt")
            git(paths.llvm_root, "checkout", "--", "llvm.txt")

            git(
                paths.circt_root,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "update",
                "--init",
                "--",
                "llvm",
            )
            self.assertIn("must remain uninitialized", self.gate_error(paths))

            git(paths.llvm_root, "checkout", "-q", topology.llvm_other)
            git(
                topology.linked,
                "update-index",
                "--cacheinfo",
                "160000",
                topology.llvm_other,
                "externals/llvm",
            )
            git(topology.linked, "commit", "-qm", "Pin incompatible LLVM")
            pair_error = self.gate_error(paths)
            self.assertIn("parent gitlinks", pair_error)
            self.assertIn("updated atomically", pair_error)

            missing_pin = commit_file(
                topology.circt_repo, "circt.txt", "circt-new\n"
            )
            git(
                topology.linked,
                "update-index",
                "--cacheinfo",
                "160000",
                missing_pin,
                "externals/circt",
            )
            git(topology.linked, "commit", "-qm", "Advance CIRCT pin")
            missing_error = self.gate_error(paths)
            self.assertIn("checkout drift", missing_error)
            self.assertIn(missing_pin, missing_error)
            self.assertIn("fetch origin", missing_error)
            self.assertNotIn("could not inspect", missing_error)

    def test_build_identity_is_deterministic_and_configures_exact_compilers(self):
        state = self.module.DependencyState("circt", "llvm")
        compilers = (
            ("/toolchain/gcc", "gcc 14.3.1"),
            ("/toolchain/g++", "g++ 14.3.1"),
        )
        paths = build_paths(Path("/tmp/loom-build-test"))
        calls = []
        with patch.object(self.module, "run", side_effect=calls.append):
            self.module.configure_llvm(paths, compilers)

        identity = self.module.llvm_build_identity(state, compilers)
        self.assertEqual(identity, self.module.llvm_build_identity(state, compilers))
        payload = json.loads(identity)
        self.assertEqual(payload["dependencies"], {"circt": "circt", "llvm": "llvm"})
        self.assertEqual(payload["compilers"]["c"]["path"], "/toolchain/gcc")
        self.assertEqual(
            payload["semantic_cmake_args"],
            list(self.module.LLVM_SEMANTIC_CMAKE_ARGS),
        )
        self.assertIn("-DCMAKE_C_COMPILER=/toolchain/gcc", calls[0])
        self.assertIn("-DCMAKE_CXX_COMPILER=/toolchain/g++", calls[0])

    def test_unknown_or_changed_identity_replaces_build_under_lock(self):
        state = self.module.DependencyState("circt", "llvm")
        current = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        old = (("/gcc", "gcc 13"), ("/g++", "g++ 13"))
        current_identity = self.module.llvm_build_identity(state, current)
        semantic_payload = json.loads(current_identity)
        semantic_payload["semantic_cmake_args"] = ["-DOLD_OPTION=ON"]
        prior_values = (
            (current_identity, False),
            (None, True),
            ("git:legacy", True),
            ("{malformed", True),
            (self.module.llvm_build_identity(state, old), True),
            (
                self.module.llvm_build_identity(
                    self.module.DependencyState("other-circt", "llvm"), current
                ),
                True,
            ),
            (json.dumps(semantic_payload, sort_keys=True), True),
        )

        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            for index, (prior, should_replace) in enumerate(prior_values):
                paths = build_paths(Path(td) / str(index))
                paths.mlir_dir.mkdir(parents=True)
                paths.cmake_clang_dir.mkdir(parents=True)
                old_object = paths.llvm_build / "old.o"
                old_object.write_text("reusable\n")
                (paths.llvm_build / "build.ninja").write_text("ready\n")
                (paths.mlir_dir / "MLIRConfig.cmake").write_text("ready\n")
                (paths.cmake_clang_dir / "ClangConfig.cmake").write_text(
                    "ready\n"
                )
                if prior is not None:
                    paths.llvm_stamp.write_text(prior + "\n")

                locked = {"value": False}

                class TrackingLock:
                    def __init__(self, *args):
                        pass

                    def __enter__(self):
                        locked["value"] = True

                    def __exit__(self, *args):
                        locked["value"] = False

                def dependency_state(_paths):
                    self.assertTrue(locked["value"])
                    return state

                def compiler_identities():
                    self.assertTrue(locked["value"])
                    return current

                def configure(configure_paths, compilers):
                    self.assertFalse(old_object.exists())
                    self.assertEqual(compilers, current)
                    configure_paths.mlir_dir.mkdir(parents=True)
                    configure_paths.cmake_clang_dir.mkdir(parents=True)
                    (configure_paths.llvm_build / "build.ninja").write_text("new\n")

                with (
                    patch.object(self.module, "FileLock", TrackingLock),
                    patch.object(
                        self.module,
                        "check_dependency_pins",
                        side_effect=dependency_state,
                    ),
                    patch.object(
                        self.module,
                        "check_llvm_compilers",
                        side_effect=compiler_identities,
                    ),
                    patch.object(self.module, "configure_llvm", side_effect=configure),
                    patch.object(self.module, "is_nfs", return_value=False),
                    patch.object(self.module, "run"),
                ):
                    self.module.ensure_shared_llvm(paths, self.args)

                self.assertFalse(locked["value"])
                self.assertEqual(old_object.exists(), not should_replace)
                self.assertEqual(
                    paths.llvm_stamp.read_text().strip(), current_identity
                )

    def test_compiler_minimums_are_preserved(self):
        with (
            patch.object(
                self.module,
                "compiler_version",
                side_effect=(
                    ((14, 3, 1), "gcc 14.3.1"),
                    ((7, 3, 0), "g++ 7.3.0"),
                ),
            ),
            patch.object(self.module.shutil, "which", return_value="/usr/bin/compiler"),
            self.assertRaises(SystemExit),
        ):
            self.module.check_llvm_compilers()

        with (
            patch.object(
                self.module,
                "compiler_version",
                side_effect=(
                    ((21, 1, 8), "clang 21.1.8"),
                    ((21, 1, 7), "clang++ 21.1.7"),
                ),
            ),
            patch.object(self.module.shutil, "which", return_value="/usr/bin/compiler"),
            self.assertRaises(SystemExit),
        ):
            self.module.check_loom_compilers()

    def test_compiler_resolution_skips_ccache_frontend_symlinks(self):
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as temp_dir:
            root = Path(temp_dir)
            launcher_dir = root / "launcher"
            compiler_dir = root / "compiler"
            launcher_dir.mkdir()
            compiler_dir.mkdir()

            ccache = launcher_dir / "ccache"
            ccache.write_text("#!/bin/sh\nexit 0\n")
            ccache.chmod(0o755)
            (launcher_dir / "gcc").symlink_to(ccache)

            compiler = compiler_dir / "gcc"
            compiler.write_text("#!/bin/sh\nexit 0\n")
            compiler.chmod(0o755)

            path = os.pathsep.join((str(launcher_dir), str(compiler_dir)))
            with patch.dict(os.environ, {"PATH": path}, clear=False):
                self.assertEqual(
                    self.module.resolve_compiler_executable("gcc"),
                    str(compiler.resolve()),
                )

    def test_doctor_and_test_commands_preserve_reports_and_lit_invocation(self):
        state = self.module.DependencyState("circt", "llvm")
        paths = build_paths(Path("/repo"))
        stdout = io.StringIO()
        with (
            patch.object(self.module, "check_git_version"),
            patch.object(self.module, "check_dependency_pins", return_value=state),
            patch.object(self.module.shutil, "which", return_value="/bin/tool"),
            patch.object(self.module, "is_nfs", return_value=False),
            patch.object(self.module, "read_stamp", return_value="identity"),
            patch.object(self.module, "compiler_status", return_value="ok"),
            patch.object(self.module, "loom_build_is_stale", return_value=False),
            redirect_stdout(stdout),
        ):
            self.module.cmd_doctor(paths, self.args)
        report = stdout.getvalue()
        self.assertIn("circt_commit    circt", report)
        self.assertIn("circt_llvm_pin  llvm", report)
        self.assertIn("nested_llvm     uninitialized", report)
        self.assertNotIn("circt_ready", report)
        self.assertIn("circt_config    False", report)
        self.assertIn("circt_lib       False", report)

        self.args.jobs = 7
        calls = []
        with (
            patch.dict("os.environ", {"LIT_OPTS": "--show-all"}, clear=True),
            patch.object(self.module, "check_git_version"),
            patch.object(self.module, "check_loom_compilers"),
            patch.object(self.module, "build_loom"),
            patch.object(
                self.module,
                "run",
                side_effect=lambda cmd, **kwargs: calls.append((cmd, kwargs)),
            ),
        ):
            self.module.cmd_test(paths, self.args)
        command, kwargs = calls[0]
        self.assertIn("-j7", command)
        self.assertIn("--show-all", command)
        self.assertEqual(kwargs["env"]["LOOM_TEST_JOBS"], "7")

    def test_circt_identity_availability_and_stamp_invariant(self):
        module = self.module
        state = module.DependencyState("circt-pin", "llvm-pin")
        compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        llvm_identity = module.llvm_build_identity(state, compilers)
        circt_identity = module.circt_build_identity(llvm_identity)

        # Identity is deterministic and embeds the full LLVM build identity,
        # so a semantic LLVM change invalidates CIRCT even when CIRCT's own
        # args and target are unchanged.
        self.assertEqual(
            circt_identity, module.circt_build_identity(llvm_identity)
        )
        payload = json.loads(circt_identity)
        self.assertEqual(
            payload["llvm_build_identity"], json.loads(llvm_identity)
        )
        self.assertEqual(
            payload["circt_build_targets"], list(module.CIRCT_BUILD_TARGETS)
        )
        self.assertEqual(
            payload["circt_semantic_cmake_args"],
            list(module.CIRCT_SEMANTIC_CMAKE_ARGS),
        )
        drifted = json.loads(llvm_identity)
        drifted["semantic_cmake_args"] = ["-DCHANGED_LLVM=ON"]
        self.assertNotEqual(
            circt_identity,
            module.circt_build_identity(json.dumps(drifted, sort_keys=True)),
        )

        # Availability requires the package config, the concrete
        # CIRCTExportVerilog library, and a stamp matching the current
        # LLVM identity. Any one missing leaves CIRCT unavailable.
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            paths.circt_cmake_dir.mkdir(parents=True)
            config = paths.circt_cmake_dir / "CIRCTConfig.cmake"

            self.assertIsNone(
                module.available_circt_dir(paths, llvm_identity)
            )
            config.write_text("x\n")
            # Config plus a matching stamp, but the library artifact is
            # still missing: must remain unavailable.
            paths.circt_stamp.write_text(circt_identity + "\n")
            self.assertIsNone(
                module.available_circt_dir(paths, llvm_identity)
            )
            paths.circt_required_lib.parent.mkdir(parents=True, exist_ok=True)
            paths.circt_required_lib.write_text("ar\n")
            self.assertEqual(
                module.available_circt_dir(paths, llvm_identity),
                str(paths.circt_cmake_dir),
            )
            stale = module.llvm_build_identity(
                module.DependencyState("circt-pin", "llvm-other"), compilers
            )
            self.assertIsNone(module.available_circt_dir(paths, stale))

            # A CIRCT build that does not produce both artifacts fails and is
            # left neither advertised nor stamped.
            incomplete = build_paths(Path(td) / "incomplete")
            incomplete.circt_build.mkdir(parents=True)
            (incomplete.circt_build / "build.ninja").write_text("ninja\n")
            with (
                patch.object(module, "configure_circt"),
                patch.object(module, "run"),
            ):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    module._sync_circt_locked(
                        incomplete, self.args, compilers, llvm_identity, True
                    )
            self.assertIsNone(
                module.available_circt_dir(incomplete, llvm_identity)
            )
            self.assertEqual(module.read_stamp(incomplete.circt_stamp), "")

    def test_configure_circt_shared_llvm_single_lock_no_nested(self):
        module = self.module
        paths = build_paths(Path("/repo"))
        compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))

        # configure_circt points at the shared sibling LLVM and emits the
        # CIRCT semantic args; the nested externals/circt/llvm never appears.
        calls = []
        with patch.object(module, "run", side_effect=calls.append):
            module.configure_circt(paths, compilers)
        self.assertEqual(len(calls), 1)
        cmd = calls[0]
        joined = " ".join(str(part) for part in cmd)
        self.assertIn(f"-DMLIR_DIR={paths.mlir_dir}", cmd)
        self.assertIn(f"-DLLVM_DIR={paths.cmake_llvm_dir}", cmd)
        self.assertIn(str(paths.circt_root), cmd)
        self.assertIn(str(paths.circt_build), cmd)
        for arg in module.CIRCT_SEMANTIC_CMAKE_ARGS:
            self.assertIn(arg, cmd)
        self.assertNotIn("circt/llvm", joined)

        # sync_shared_circt holds the single LLVM lock across the LLVM
        # ensure phase and the CIRCT phase (in that order) and never opens
        # a CIRCT-only lock.
        state = module.DependencyState("circt-pin", "llvm-pin")
        llvm_identity = module.llvm_build_identity(state, compilers)
        events: list = []

        class TrackingLock:
            def __init__(self, path, timeout):
                events.append(("lock-open", str(path)))
                self.path = path

            def __enter__(self):
                events.append(("lock-enter", str(self.path)))
                return self

            def __exit__(self, *exc):
                events.append(("lock-exit", str(self.path)))

        def llvm_phase(*args, **kwargs):
            events.append("llvm-sync")
            return llvm_identity

        def circt_phase(*args, **kwargs):
            events.append("circt-sync")

        with (
            patch.object(module, "FileLock", TrackingLock),
            patch.object(
                module, "check_dependency_pins", return_value=state
            ),
            patch.object(
                module, "check_llvm_compilers", return_value=compilers
            ),
            patch.object(module, "_sync_llvm_locked", side_effect=llvm_phase),
            patch.object(module, "_sync_circt_locked", side_effect=circt_phase),
            patch.object(module, "is_nfs", return_value=False),
        ):
            module.sync_shared_circt(paths, self.args, True)

        self.assertEqual(
            [e[1] for e in events if e[0] == "lock-open"],
            [str(paths.llvm_lock)],
        )
        enter = events.index(("lock-enter", str(paths.llvm_lock)))
        exit_idx = events.index(("lock-exit", str(paths.llvm_lock)))
        self.assertLess(enter, events.index("llvm-sync"))
        self.assertLess(events.index("llvm-sync"), events.index("circt-sync"))
        self.assertLess(events.index("circt-sync"), exit_idx)

    def test_linked_routing_and_no_implicit_circt_build(self):
        module = self.module
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            topology = GitTopology(Path(td))
            linked = module.Paths(topology.linked)
            main_externals = topology.main / "externals"

            # All CIRCT build state is owned by the main worktree's
            # externals; a linked worktree holds none of its own.
            self.assertEqual(
                linked.circt_build, main_externals / "circt" / "build"
            )
            self.assertEqual(
                linked.circt_stamp,
                main_externals / ".loom-build.circt.stamp",
            )
            self.assertEqual(
                linked.circt_required_lib,
                main_externals / "circt" / "build" / "lib"
                / "libCIRCTExportVerilog.a",
            )
            for owned in (
                linked.circt_build,
                linked.circt_stamp,
                linked.circt_required_lib,
            ):
                owned_str = str(owned)
                self.assertTrue(owned_str.startswith(str(main_externals)))
                self.assertFalse(owned_str.startswith(str(topology.linked)))

            # `make loom` never builds CIRCT; it only offers an
            # already-built, stamped CIRCT matching the ensured LLVM.
            state = module.DependencyState(
                topology.circt_pin, topology.llvm_pin
            )
            compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
            llvm_identity = module.llvm_build_identity(state, compilers)
            circt_identity = module.circt_build_identity(llvm_identity)

            linked.circt_cmake_dir.mkdir(parents=True)
            (linked.circt_cmake_dir / "CIRCTConfig.cmake").write_text("x\n")
            linked.circt_required_lib.parent.mkdir(parents=True, exist_ok=True)
            linked.circt_required_lib.write_text("ar\n")
            linked.circt_stamp.write_text(circt_identity + "\n")

            circt_build_calls: list = []
            offered = {}

            def capture_configure(configure_paths, circt_dir):
                offered["circt_dir"] = circt_dir

            with (
                patch.object(
                    module, "ensure_shared_llvm", return_value=llvm_identity
                ),
                patch.object(
                    module, "configure_loom", side_effect=capture_configure
                ),
                patch.object(module, "run"),
                patch.object(
                    module,
                    "sync_shared_circt",
                    side_effect=lambda *a, **k: circt_build_calls.append(1),
                ),
                patch.object(
                    module, "build_circt",
                    side_effect=lambda *a, **k: circt_build_calls.append(1),
                ),
                patch.object(
                    module, "configure_circt",
                    side_effect=lambda *a, **k: circt_build_calls.append(1),
                ),
            ):
                module.build_loom(linked, self.args)

            self.assertEqual(circt_build_calls, [])
            self.assertEqual(offered["circt_dir"], str(linked.circt_cmake_dir))

    def test_main_and_linked_distclean_ownership(self):
        module = self.module
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            # Main: both shared builds are removed under the single LLVM lock.
            main_paths = build_paths(Path(td) / "main")
            main_paths.circt_cmake_dir.mkdir(parents=True)
            (main_paths.circt_cmake_dir / "CIRCTConfig.cmake").write_text("x\n")
            main_paths.circt_required_lib.parent.mkdir(parents=True, exist_ok=True)
            main_paths.circt_required_lib.write_text("ar\n")
            main_paths.circt_stamp.write_text("id\n")
            main_paths.llvm_build.mkdir(parents=True)
            main_paths.llvm_stamp.write_text("id\n")

            main_locks: list = []

            class TrackingLock:
                def __init__(self, path, timeout):
                    main_locks.append(str(path))

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            with patch.object(module, "FileLock", TrackingLock):
                module.cmd_distclean(main_paths, self.args)
            self.assertEqual(main_locks, [str(main_paths.llvm_lock)])
            self.assertFalse(main_paths.circt_build.exists())
            self.assertFalse(main_paths.circt_stamp.exists())
            self.assertFalse(main_paths.llvm_build.exists())
            self.assertFalse(main_paths.llvm_stamp.exists())

            # Linked: shared builds are preserved and no lock is taken.
            linked_paths = build_paths(Path(td) / "linked")
            linked_paths.main = Path(td) / "main"
            linked_paths.is_main = False
            linked_paths.circt_cmake_dir.mkdir(parents=True)
            (linked_paths.circt_cmake_dir / "CIRCTConfig.cmake").write_text("x\n")
            linked_paths.circt_required_lib.parent.mkdir(parents=True, exist_ok=True)
            linked_paths.circt_required_lib.write_text("ar\n")
            linked_paths.circt_stamp.write_text("id\n")

            opened = {"value": False}

            class NoLock:
                def __init__(self, *args):
                    opened["value"] = True

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            with patch.object(module, "FileLock", NoLock):
                module.cmd_distclean(linked_paths, self.args)
            self.assertFalse(opened["value"])
            self.assertTrue(linked_paths.circt_build.exists())
            self.assertTrue(linked_paths.circt_stamp.exists())

    def test_shared_build_state_is_git_ignored(self):
        """The shared LLVM and CIRCT build state under externals/ is local
        and must be ignored. Verified through git's actual ignore behavior
        against the repository's own .gitignore, not a source-text match, so
        the policy may be expressed by any equivalent pattern."""
        gitignore = (SCRIPT.parents[1] / ".gitignore").read_text()
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            sandbox = Path(td) / "repo"
            init_repo(sandbox)
            (sandbox / ".gitignore").write_text(gitignore)
            shared_state = (
                "externals/.loom-build.llvm.lock",
                "externals/.loom-build.llvm.stamp",
                "externals/.loom-build.circt.stamp",
            )
            result = subprocess.run(
                [
                    "git", "-C", str(sandbox),
                    "-c", "core.excludesFile=/dev/null",
                    "check-ignore", "--", *shared_state,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            ignored = set(result.stdout.split())
            for path in shared_state:
                self.assertIn(path, ignored, f"{path} is not git-ignored")

    def test_failed_explicit_circt_rebuild_revokes_availability(self):
        """An explicit CIRCT rebuild that fails must revoke availability even
        when a valid, matching stamp and complete artifacts already exist.
        The stamp is the single readiness authority: a build attempt
        invalidates it up front, so a failed build cannot keep advertising
        the old or a partially rebuilt CIRCT."""
        module = self.module
        state = module.DependencyState("circt-pin", "llvm-pin")
        compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        llvm_identity = module.llvm_build_identity(state, compilers)
        circt_identity = module.circt_build_identity(llvm_identity)

        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            # A complete, stamped, identity-matching prior build.
            paths.circt_cmake_dir.mkdir(parents=True)
            (paths.circt_cmake_dir / "CIRCTConfig.cmake").write_text("x\n")
            paths.circt_required_lib.parent.mkdir(parents=True, exist_ok=True)
            paths.circt_required_lib.write_text("ar\n")
            (paths.circt_build / "build.ninja").write_text("ninja\n")
            paths.circt_stamp.write_text(circt_identity + "\n")

            # Precondition: the prior build is advertised as available.
            self.assertEqual(
                module.available_circt_dir(paths, llvm_identity),
                str(paths.circt_cmake_dir),
            )

            # An explicit rebuild whose cmake --build fails. The prior
            # artifacts stay on disk (the build failed before overwriting
            # them), so only stamp invalidation can revoke availability.
            def failing_build(cmd, **kwargs):
                raise subprocess.CalledProcessError(1, cmd)

            with (
                patch.object(module, "configure_circt"),
                patch.object(module, "run", side_effect=failing_build),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                module._sync_circt_locked(
                    paths, self.args, compilers, llvm_identity, True
                )

            # After the failed rebuild the stale build must not be advertised.
            self.assertIsNone(
                module.available_circt_dir(paths, llvm_identity)
            )

    def test_explicit_circt_dir_is_exact_without_search_path_fallback(self):
        """An explicit CIRCT_DIR is loaded exactly. An invalid directory must
        fail the CIRCT package lookup rather than fall back to a valid CIRCT
        reachable through CMAKE_PREFIX_PATH, verified by really configuring the
        repository's top-level CMakeLists.txt."""
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("cmake not available")

        repo_root = SCRIPT.parents[1]
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            root = Path(td)
            # One prefix reachable via CMAKE_PREFIX_PATH: minimal MLIR and
            # Clang configs to satisfy the two find_package calls before
            # CIRCT, plus a valid decoy CIRCT the exact lookup must ignore.
            prefix = root / "prefix"
            for name, subdir in (
                ("MLIR", "mlir"), ("Clang", "clang"), ("CIRCT", "circt"),
            ):
                cfg_dir = prefix / "lib" / "cmake" / subdir
                cfg_dir.mkdir(parents=True)
                (cfg_dir / f"{name}Config.cmake").write_text(
                    f"set({name}_FOUND TRUE)\n"
                )
            decoy_circt_dir = prefix / "lib" / "cmake" / "circt"
            # An existing directory that holds no CIRCT package.
            bad_circt_dir = root / "bad-circt"
            bad_circt_dir.mkdir()

            result = subprocess.run(
                [
                    cmake,
                    "-S", str(repo_root),
                    "-B", str(root / "build"),
                    f"-DCIRCT_DIR={bad_circt_dir}",
                    f"-DCMAKE_PREFIX_PATH={prefix}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Configuration fails at the explicit CIRCT lookup and never
            # reports using the decoy reachable through CMAKE_PREFIX_PATH.
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn('provided by "CIRCT"', result.stdout)
            self.assertNotIn(str(decoy_circt_dir), result.stdout)


if __name__ == "__main__":
    unittest.main()
