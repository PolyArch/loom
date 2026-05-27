#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("make-worktree.py")


def load_dispatcher():
    spec = importlib.util.spec_from_file_location("make_worktree", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakePaths:
    def __init__(self):
        root = Path("/repo")
        self.root = root
        self.main = root
        self.llvm_root = root / "externals" / "llvm"
        self.llvm_src = self.llvm_root / "llvm"
        self.llvm_build = self.llvm_root / "build"
        self.llvm_lock = root / "externals" / ".loom-build.llvm.lock"
        self.llvm_stamp = root / "externals" / ".loom-build.llvm.stamp"
        self.loom_build = root / "build"
        self.mlir_dir = self.llvm_build / "lib" / "cmake" / "mlir"
        self.cmake_llvm_dir = self.llvm_build / "lib" / "cmake" / "llvm"
        self.cmake_clang_dir = self.llvm_build / "lib" / "cmake" / "clang"
        self.llvm_lit = self.llvm_build / "bin" / "llvm-lit"

    @property
    def is_main(self):
        return True


class MakeWorktreeTest(unittest.TestCase):
    def setUp(self):
        self.module = load_dispatcher()
        self.paths = FakePaths()
        self.args = Namespace(jobs=1, lock_timeout=1.0)

    def test_configure_llvm_defaults_to_gcc_pair(self):
        calls = []
        with patch.object(self.module, "run", side_effect=lambda cmd: calls.append(cmd)):
            self.module.configure_llvm(self.paths)

        self.assertIn("-DCMAKE_C_COMPILER=gcc", calls[0])
        self.assertIn("-DCMAKE_CXX_COMPILER=g++", calls[0])

    def test_configure_loom_keeps_clang_pair(self):
        calls = []
        with patch.object(self.module, "run", side_effect=lambda cmd: calls.append(cmd)):
            self.module.configure_loom(self.paths)

        self.assertIn("-DCMAKE_C_COMPILER=clang", calls[0])
        self.assertIn("-DCMAKE_CXX_COMPILER=clang++", calls[0])

    def test_build_llvm_rejects_old_gxx(self):
        def fake_check_output(cmd, **kwargs):
            tool = cmd[0]
            if tool == "gcc":
                return b"gcc (GCC) 14.3.1\n"
            if tool == "g++":
                return b"g++ (GCC) 7.3.0\n"
            raise AssertionError(f"unexpected command: {cmd}")

        with (
            patch.object(self.module, "check_git_version"),
            patch.object(self.module, "build_llvm"),
            patch.object(subprocess, "check_output", side_effect=fake_check_output),
            self.assertRaises(SystemExit) as raised,
        ):
            self.module.cmd_build_llvm(self.paths, self.args)

        self.assertEqual(raised.exception.code, 1)

    def test_build_llvm_reconfigures_stale_clang_cache(self):
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            paths = FakePaths()
            paths.llvm_root = temp
            paths.llvm_src = temp / "llvm"
            paths.llvm_build = temp / "build"
            paths.llvm_lock = temp.parent / ".loom-build.llvm.lock"
            paths.llvm_stamp = temp.parent / ".loom-build.llvm.stamp"
            paths.llvm_build.mkdir()
            (paths.llvm_build / "build.ninja").write_text("rule stale\n")
            (paths.llvm_build / "CMakeCache.txt").write_text(
                "CMAKE_C_COMPILER:STRING=/usr/bin/clang\n"
                "CMAKE_CXX_COMPILER:STRING=/usr/bin/clang++\n"
            )

            with (
                patch.object(self.module, "is_nfs", return_value=False),
                patch.object(self.module, "llvm_source_id", return_value="git:new"),
                patch.object(self.module, "configure_llvm") as configure_llvm,
                patch.object(self.module, "run"),
                patch.object(self.module, "write_stamp"),
            ):
                self.module.build_llvm(paths, self.args)

            configure_llvm.assert_called_once_with(paths)

    def test_paths_keep_loom_metadata_outside_llvm_submodule(self):
        root = Path("/repo")
        with patch.object(self.module, "real", side_effect=lambda path: Path(path)):
            with patch.object(self.module, "resolve_main_worktree", return_value=root):
                paths = self.module.Paths(root)

        self.assertEqual(paths.llvm_lock, root / "externals" / ".loom-build.llvm.lock")
        self.assertEqual(paths.llvm_stamp, root / "externals" / ".loom-build.llvm.stamp")

    def test_build_loom_rejects_clang_before_21_1_8(self):
        def fake_check_output(cmd, **kwargs):
            tool = cmd[0]
            if tool == "clang":
                return b"clang version 21.1.8\n"
            if tool == "clang++":
                return b"clang version 21.1.7\n"
            raise AssertionError(f"unexpected command: {cmd}")

        with (
            patch.object(self.module, "check_git_version"),
            patch.object(self.module, "build_loom"),
            patch.object(subprocess, "check_output", side_effect=fake_check_output),
            self.assertRaises(SystemExit) as raised,
        ):
            self.module.cmd_build_loom(self.paths, self.args)

        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
