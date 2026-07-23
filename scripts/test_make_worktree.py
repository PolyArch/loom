#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import io
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
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
        llvm_lock_turnstile=root / ".llvm.turnstile.lock",
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
                paths.cmake_llvm_dir.mkdir(parents=True)
                (paths.cmake_llvm_dir / "LLVMConfig.cmake").write_text(
                    "ready\n"
                )
                (paths.cmake_clang_dir / "ClangConfig.cmake").write_text(
                    "ready\n"
                )
                if prior is not None:
                    paths.llvm_stamp.write_text(prior + "\n")

                locked = {"value": False}

                class TrackingLock:
                    def __init__(self, *args, **kwargs):
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
                    configure_paths.cmake_llvm_dir.mkdir(parents=True)
                    configure_paths.cmake_clang_dir.mkdir(parents=True)
                    (configure_paths.llvm_build / "build.ninja").write_text("new\n")
                    (
                        configure_paths.mlir_dir / "MLIRConfig.cmake"
                    ).write_text("ready\n")
                    (
                        configure_paths.cmake_llvm_dir / "LLVMConfig.cmake"
                    ).write_text("ready\n")
                    (
                        configure_paths.cmake_clang_dir / "ClangConfig.cmake"
                    ).write_text("ready\n")

                with (
                    patch.object(
                        self.module, "SharedProductLock", TrackingLock
                    ),
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
                        incomplete, self.args, compilers, llvm_identity, True, ""
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

        # sync_shared_circt holds one exclusive shared-product transaction
        # across the LLVM ensure and CIRCT phases.
        state = module.DependencyState("circt-pin", "llvm-pin")
        llvm_identity = module.llvm_build_identity(state, compilers)
        events: list = []

        class TrackingLock:
            def __init__(self, product_path, turnstile_path, timeout, shared):
                events.append(
                    (
                        "lock-open",
                        str(product_path),
                        str(turnstile_path),
                        shared,
                    )
                )
                self.path = product_path

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
            patch.object(module, "SharedProductLock", TrackingLock),
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
        self.assertEqual(
            [e[2] for e in events if e[0] == "lock-open"],
            [str(paths.llvm_lock_turnstile)],
        )
        self.assertEqual(
            [e[3] for e in events if e[0] == "lock-open"],
            [False],
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
            self._stamped_llvm_build(linked, llvm_identity)

            circt_build_calls: list = []
            offered = {}

            def capture_configure(configure_paths, circt_dir):
                offered["circt_dir"] = circt_dir

            with (
                patch.object(
                    module, "ensure_shared_llvm", return_value=llvm_identity
                ),
                patch.object(
                    module, "check_dependency_pins", return_value=state
                ),
                patch.object(
                    module, "check_llvm_compilers", return_value=compilers
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
            # Main: both shared builds are removed in one writer transaction.
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
                def __init__(
                    self, product_path, turnstile_path, timeout, shared
                ):
                    main_locks.append(
                        (str(product_path), str(turnstile_path), shared)
                    )

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            with patch.object(module, "SharedProductLock", TrackingLock):
                module.cmd_distclean(main_paths, self.args)
            self.assertEqual(
                main_locks,
                [(
                    str(main_paths.llvm_lock),
                    str(main_paths.llvm_lock_turnstile),
                    False,
                )],
            )
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
                def __init__(self, *args, **kwargs):
                    opened["value"] = True

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            with patch.object(module, "SharedProductLock", NoLock):
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
                "externals/.loom-build.llvm.turnstile.lock",
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

    def _stamped_circt_build(self, paths, circt_identity):
        """Set up a complete, stamped, identity-matching prior CIRCT build."""
        paths.circt_cmake_dir.mkdir(parents=True)
        (paths.circt_cmake_dir / "CIRCTConfig.cmake").write_text("x\n")
        paths.circt_required_lib.parent.mkdir(parents=True, exist_ok=True)
        paths.circt_required_lib.write_text("ar\n")
        (paths.circt_build / "build.ninja").write_text("ninja\n")
        paths.circt_stamp.write_text(circt_identity + "\n")

    def _stamped_llvm_build(self, paths, llvm_identity):
        """Set up a complete, stamped, identity-matching prior LLVM build."""
        for config_dir, config_name in (
            (paths.mlir_dir, "MLIRConfig.cmake"),
            (paths.cmake_llvm_dir, "LLVMConfig.cmake"),
            (paths.cmake_clang_dir, "ClangConfig.cmake"),
        ):
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / config_name).write_text("ready\n")
        (paths.llvm_build / "build.ninja").write_text("ninja\n")
        paths.llvm_stamp.write_text(llvm_identity + "\n")

    def test_cmd_build_llvm_invalidates_stamps_before_git_preflight(self):
        """The public LLVM command revokes LLVM and dependent CIRCT readiness
        before its failure-capable compatibility preflight."""
        module = self.module
        state = module.DependencyState("circt-pin", "llvm-pin")
        compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        llvm_identity = module.llvm_build_identity(state, compilers)
        circt_identity = module.circt_build_identity(llvm_identity)

        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            self._stamped_llvm_build(paths, llvm_identity)
            self._stamped_circt_build(paths, circt_identity)

            def fail_preflight():
                self.assertFalse(paths.llvm_stamp.exists())
                self.assertFalse(paths.circt_stamp.exists())
                raise RuntimeError("injected git preflight failure")

            with (
                patch.object(module, "is_nfs", return_value=False),
                patch.object(
                    module,
                    "check_git_version",
                    side_effect=fail_preflight,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "injected git preflight failure"
                ),
            ):
                module.cmd_build_llvm(paths, self.args)

            self.assertTrue((paths.llvm_build / "build.ninja").exists())
            self.assertTrue(
                (paths.circt_cmake_dir / "CIRCTConfig.cmake").exists()
            )
            self.assertFalse(paths.llvm_stamp.exists())
            self.assertFalse(paths.circt_stamp.exists())
            self.assertIsNone(module.available_circt_dir(paths, llvm_identity))

    def test_failed_explicit_llvm_build_revokes_both_readiness_stamps(self):
        """A matching incremental LLVM build that fails through the public
        entry point leaves neither LLVM nor dependent CIRCT advertised."""
        module = self.module
        state = module.DependencyState("circt-pin", "llvm-pin")
        compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        llvm_identity = module.llvm_build_identity(state, compilers)
        circt_identity = module.circt_build_identity(llvm_identity)

        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            self._stamped_llvm_build(paths, llvm_identity)
            self._stamped_circt_build(paths, circt_identity)

            def fail_build(cmd, **kwargs):
                raise subprocess.CalledProcessError(1, cmd)

            with (
                patch.object(module, "is_nfs", return_value=False),
                patch.object(module, "check_dependency_pins", return_value=state),
                patch.object(
                    module, "check_llvm_compilers", return_value=compilers
                ),
                patch.object(module, "check_git_version"),
                patch.object(module, "configure_llvm") as configure_llvm,
                patch.object(module, "run", side_effect=fail_build),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                module.cmd_build_llvm(paths, self.args)

            configure_llvm.assert_not_called()
            self.assertTrue((paths.llvm_build / "build.ninja").exists())
            self.assertFalse(paths.llvm_stamp.exists())
            self.assertFalse(paths.circt_stamp.exists())
            self.assertIsNone(module.available_circt_dir(paths, llvm_identity))

    def test_successful_explicit_llvm_build_restores_only_llvm_readiness(self):
        """Validated LLVM success restores its own readiness but keeps CIRCT
        revoked until a later explicit CIRCT build validates that dependency."""
        module = self.module
        state = module.DependencyState("circt-pin", "llvm-pin")
        compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        llvm_identity = module.llvm_build_identity(state, compilers)
        circt_identity = module.circt_build_identity(llvm_identity)

        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            self._stamped_llvm_build(paths, llvm_identity)
            self._stamped_circt_build(paths, circt_identity)

            with (
                patch.object(module, "is_nfs", return_value=False),
                patch.object(module, "check_dependency_pins", return_value=state),
                patch.object(
                    module, "check_llvm_compilers", return_value=compilers
                ),
                patch.object(module, "check_git_version"),
                patch.object(module, "configure_llvm") as configure_llvm,
                patch.object(module, "run"),
            ):
                module.cmd_build_llvm(paths, self.args)

            configure_llvm.assert_not_called()
            self.assertEqual(module.read_stamp(paths.llvm_stamp), llvm_identity)
            self.assertFalse(paths.circt_stamp.exists())
            self.assertIsNone(module.available_circt_dir(paths, llvm_identity))

    def test_public_writer_precedes_late_reader_across_processes(self):
        """Two Loom readers coexist, then a queued public LLVM writer enters
        before a later Loom reader in separate processes."""
        module = self.module
        state = module.DependencyState("circt-pin", "llvm-pin")
        compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        llvm_identity = module.llvm_build_identity(state, compilers)

        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            shared = build_paths(Path(td) / "shared")
            self._stamped_llvm_build(shared, llvm_identity)

            readers = {}
            for name in ("reader-a", "reader-b", "late-reader"):
                paths = SimpleNamespace(**vars(shared))
                paths.root = Path(td) / name
                paths.loom_build = paths.root / "build"
                paths.loom_build.mkdir(parents=True)
                (paths.loom_build / "build.ninja").write_text("ninja\n")
                (paths.loom_build / "CMakeCache.txt").write_text(
                    "CMAKE_C_COMPILER:FILEPATH=/usr/bin/clang\n"
                    "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++\n"
                )
                readers[name] = paths

            context = multiprocessing.get_context("fork")
            active = {
                name: context.Event()
                for name in ("reader-a", "reader-b", "late-reader", "writer")
            }
            release = {
                name: context.Event()
                for name in ("reader-a", "reader-b", "late-reader", "writer")
            }
            writer_waiting = context.Event()
            late_reader_waiting = context.Event()
            reader_by_build = {
                str(paths.loom_build): name for name, paths in readers.items()
            }

            def controlled_run(cmd, **kwargs):
                if cmd[:2] == ["cmake", "--build"]:
                    if cmd[2] == str(shared.llvm_build):
                        active["writer"].set()
                        if not release["writer"].wait(5.0):
                            raise RuntimeError("writer release timed out")
                        return
                    name = reader_by_build.get(cmd[2])
                    if name is not None:
                        active[name].set()
                        if not release[name].wait(5.0):
                            raise RuntimeError(f"{name} release timed out")
                        return
                raise AssertionError(f"unexpected command: {cmd}")

            def capture_info(message):
                process = multiprocessing.current_process().name
                if process == "queued-writer" and "product lock" in message:
                    writer_waiting.set()
                if process == "late-reader" and "turnstile" in message:
                    late_reader_waiting.set()

            def run_reader(name):
                module.cmd_build_loom(readers[name], self.args)

            def run_writer():
                module.cmd_build_llvm(shared, self.args)

            processes = {
                name: context.Process(
                    target=run_reader,
                    args=(name,),
                    name=name,
                )
                for name in readers
            }
            processes["writer"] = context.Process(
                target=run_writer,
                name="queued-writer",
            )
            try:
                with (
                    patch.object(
                        module,
                        "ensure_shared_llvm",
                        return_value=llvm_identity,
                    ),
                    patch.object(
                        module, "check_dependency_pins", return_value=state
                    ),
                    patch.object(
                        module, "check_llvm_compilers", return_value=compilers
                    ),
                    patch.object(module, "check_git_version"),
                    patch.object(module, "check_loom_compilers"),
                    patch.object(module, "is_nfs", return_value=False),
                    patch.object(module, "run", side_effect=controlled_run),
                    patch.object(module, "info", side_effect=capture_info),
                ):
                    processes["reader-a"].start()
                    self.assertTrue(active["reader-a"].wait(2.0))
                    processes["reader-b"].start()
                    self.assertTrue(
                        active["reader-b"].wait(2.0),
                        "public Loom readers did not coexist",
                    )
                    processes["writer"].start()
                    self.assertTrue(
                        writer_waiting.wait(2.0),
                        "public LLVM writer did not wait for Loom readers",
                    )
                    processes["late-reader"].start()
                    self.assertTrue(
                        late_reader_waiting.wait(2.0),
                        "late Loom reader did not wait at the turnstile",
                    )
                    self.assertFalse(active["late-reader"].is_set())

                    release["reader-a"].set()
                    release["reader-b"].set()
                    self.assertTrue(
                        active["writer"].wait(2.0),
                        "queued writer did not acquire the product lock",
                    )
                    self.assertFalse(active["late-reader"].is_set())

                    release["writer"].set()
                    self.assertTrue(
                        active["late-reader"].wait(2.0),
                        "late reader did not enter after the writer",
                    )
                    release["late-reader"].set()
            finally:
                for event in release.values():
                    event.set()
                for process in processes.values():
                    if process.pid is None:
                        continue
                    process.join(5.0)
                    if process.is_alive():
                        process.terminate()
                        process.join(5.0)

            self.assertEqual(
                {name: process.exitcode for name, process in processes.items()},
                {name: 0 for name in processes},
            )
            self.assertEqual(module.read_stamp(shared.llvm_stamp), llvm_identity)
            self.assertFalse(shared.circt_stamp.exists())

    def test_public_loom_reader_honors_short_lock_timeout(self):
        """A public Loom reader contending with a real writer observes one
        monotonic 0.05 second deadline instead of a fixed one-second sleep."""
        module = self.module
        context = multiprocessing.get_context("fork")
        held = context.Event()
        release = context.Event()

        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))

            def hold_writer():
                with module.SharedProductLock(
                    paths.llvm_lock,
                    paths.llvm_lock_turnstile,
                    2.0,
                    shared=False,
                ):
                    held.set()
                    if not release.wait(5.0):
                        raise RuntimeError("holder release timed out")

            holder = context.Process(target=hold_writer, name="lock-holder")
            try:
                holder.start()
                self.assertTrue(held.wait(2.0), "writer did not acquire lock")
                args = Namespace(jobs=1, lock_timeout=0.05)
                stderr = io.StringIO()
                started = time.monotonic()
                with (
                    patch.object(
                        module, "ensure_shared_llvm", return_value="identity"
                    ),
                    redirect_stderr(stderr),
                    self.assertRaises(SystemExit),
                ):
                    module.build_loom(paths, args)
                elapsed = time.monotonic() - started
            finally:
                release.set()
                holder.join(5.0)
                if holder.is_alive():
                    holder.terminate()
                    holder.join(5.0)

            self.assertEqual(holder.exitcode, 0)
            self.assertGreaterEqual(elapsed, 0.04)
            self.assertLess(elapsed, 0.30)
            self.assertIn("timed out after 0.05s", stderr.getvalue())

    def test_cmd_build_circt_invalidates_stamp_before_git_preflight(self):
        """The public CIRCT command revokes its readiness before its
        failure-capable compatibility preflight while preserving valid LLVM."""
        module = self.module
        state = module.DependencyState("circt-pin", "llvm-pin")
        compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        llvm_identity = module.llvm_build_identity(state, compilers)
        circt_identity = module.circt_build_identity(llvm_identity)

        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            self._stamped_llvm_build(paths, llvm_identity)
            self._stamped_circt_build(paths, circt_identity)
            self.assertEqual(
                module.available_circt_dir(paths, llvm_identity),
                str(paths.circt_cmake_dir),
            )

            def fail_preflight():
                self.assertEqual(
                    module.read_stamp(paths.llvm_stamp), llvm_identity
                )
                self.assertFalse(paths.circt_stamp.exists())
                raise RuntimeError("injected git preflight failure")

            with (
                patch.object(module, "is_nfs", return_value=False),
                patch.object(
                    module,
                    "check_git_version",
                    side_effect=fail_preflight,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "injected git preflight failure"
                ),
            ):
                module.cmd_build_circt(paths, self.args)

            self.assertEqual(module.read_stamp(paths.llvm_stamp), llvm_identity)
            self.assertTrue(
                (paths.circt_cmake_dir / "CIRCTConfig.cmake").exists()
            )
            self.assertTrue(paths.circt_required_lib.exists())
            self.assertFalse(paths.circt_stamp.exists())
            self.assertIsNone(module.available_circt_dir(paths, llvm_identity))

    def test_failed_explicit_circt_build_revokes_availability(self):
        """A failed explicit CIRCT build through the public `make circt`
        pipeline leaves no CIRCT advertised. A still-matching complete build is
        reused incrementally: configure_circt is not called and the prior
        artifacts and build.ninja survive the failed cmake build, yet the
        up-front stamp invalidation still leaves the stamp absent and
        availability None."""
        module = self.module
        state = module.DependencyState("circt-pin", "llvm-pin")
        compilers = (("/gcc", "gcc 14"), ("/g++", "g++ 14"))
        llvm_identity = module.llvm_build_identity(state, compilers)
        circt_identity = module.circt_build_identity(llvm_identity)

        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            paths = build_paths(Path(td))
            self._stamped_circt_build(paths, circt_identity)
            self.assertEqual(
                module.available_circt_dir(paths, llvm_identity),
                str(paths.circt_cmake_dir),
            )

            def failing_build(cmd, **kwargs):
                raise subprocess.CalledProcessError(1, cmd)

            with (
                patch.object(module, "is_nfs", return_value=False),
                patch.object(module, "check_dependency_pins", return_value=state),
                patch.object(
                    module, "check_llvm_compilers", return_value=compilers
                ),
                patch.object(
                    module, "_sync_llvm_locked", return_value=llvm_identity
                ),
                patch.object(module, "check_git_version"),
                patch.object(module, "configure_circt") as configure_circt,
                patch.object(module, "run", side_effect=failing_build),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                module.cmd_build_circt(paths, self.args)

            # The matching complete build is reused, not wiped: no reconfigure,
            # and the prior build tree survives the failed cmake build.
            configure_circt.assert_not_called()
            self.assertTrue(
                (paths.circt_cmake_dir / "CIRCTConfig.cmake").exists()
            )
            self.assertTrue(paths.circt_required_lib.exists())
            self.assertTrue((paths.circt_build / "build.ninja").exists())
            # The up-front invalidation still revokes readiness.
            self.assertFalse(paths.circt_stamp.exists())
            self.assertIsNone(module.available_circt_dir(paths, llvm_identity))

    def test_explicit_circt_dir_is_exact_package_directory(self):
        """CIRCT_DIR names the exact directory holding CIRCTConfig.cmake, with
        no prefix expansion and no search fallback. Supplying a build root
        whose CIRCT package is nested at lib/cmake/circt must fail the CIRCT
        lookup; supplying that exact package directory must pass it. Verified
        by really configuring the repository's top-level CMakeLists.txt."""
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("cmake not available")

        repo_root = SCRIPT.parents[1]
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT) as td:
            root = Path(td)
            # One fixture: a build-root prefix on CMAKE_PREFIX_PATH holding
            # minimal MLIR and Clang configs for the two find_package calls
            # before CIRCT, plus a valid CIRCT package nested at the standard
            # lib/cmake/circt location.
            build_root = root / "fixture"
            for name, subdir in (
                ("MLIR", "mlir"), ("Clang", "clang"), ("CIRCT", "circt"),
            ):
                cfg_dir = build_root / "lib" / "cmake" / subdir
                cfg_dir.mkdir(parents=True)
                (cfg_dir / f"{name}Config.cmake").write_text(
                    f"set({name}_FOUND TRUE)\n"
                )
            circt_pkg_dir = build_root / "lib" / "cmake" / "circt"

            def configure(circt_dir):
                return subprocess.run(
                    [
                        cmake,
                        "-S", str(repo_root),
                        "-B", str(root / f"build-{circt_dir.name}"),
                        f"-DCIRCT_DIR={circt_dir}",
                        f"-DCMAKE_PREFIX_PATH={build_root}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            # Build root: the CIRCT package is only nested, so the exact
            # lookup fails (no prefix expansion, no CMAKE_PREFIX_PATH fallback).
            nested = configure(build_root)
            self.assertNotEqual(nested.returncode, 0, nested.stdout)
            self.assertIn('provided by "CIRCT"', nested.stdout)

            # Exact package directory: the CIRCT lookup succeeds. Any later
            # configure failure is unrelated to CIRCT resolution.
            exact = configure(circt_pkg_dir)
            self.assertIn(
                f"Using CIRCTConfig.cmake in: {circt_pkg_dir}", exact.stdout
            )
            self.assertNotIn('provided by "CIRCT"', exact.stdout)


if __name__ == "__main__":
    unittest.main()
