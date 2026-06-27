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
REPO_TEMP_ROOT = SCRIPT.parents[1] / "temp" / "test-runs"


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

    def patch_compiler_versions(self, versions: dict[str, str]):
        def check_output(cmd, **kwargs):
            tool = cmd[0]
            if tool in versions:
                return versions[tool].encode()
            raise AssertionError(f"unexpected command: {cmd}")

        return patch.object(subprocess, "check_output", side_effect=check_output)

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
        with (
            patch.object(self.module, "check_git_version"),
            patch.object(self.module, "build_llvm"),
            self.patch_compiler_versions(
                {
                    "gcc": "gcc (GCC) 14.3.1\n",
                    "g++": "g++ (GCC) 7.3.0\n",
                }
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            self.module.cmd_build_llvm(self.paths, self.args)

        self.assertEqual(raised.exception.code, 1)

    def test_build_llvm_reconfigures_stale_clang_cache(self):
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="loom-worktree-test-", dir=REPO_TEMP_ROOT) as td:
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
        with (
            patch.object(self.module, "check_git_version"),
            patch.object(self.module, "build_loom"),
            self.patch_compiler_versions(
                {
                    "clang": "clang version 21.1.8\n",
                    "clang++": "clang version 21.1.7\n",
                }
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            self.module.cmd_build_loom(self.paths, self.args)

        self.assertEqual(raised.exception.code, 1)

    def test_cmd_test_isolates_perf_lit_from_artifact_groups(self):
        popen_calls = []
        wait_seen_launch_counts = []
        self.args.jobs = 7

        class FakePipe:
            def close(self):
                pass

        class FakeProcess:
            def __init__(self, cmd, **kwargs):
                self.cmd = cmd
                self.kwargs = kwargs
                self.stdout = FakePipe()
                popen_calls.append(self)

            def wait(self):
                wait_seen_launch_counts.append(len(popen_calls))
                return 0

        with (
            patch.object(self.module, "check_git_version"),
            patch.object(self.module, "check_loom_compilers"),
            patch.object(self.module, "build_loom"),
            patch.object(subprocess, "Popen", side_effect=lambda cmd, **kwargs: FakeProcess(cmd, **kwargs)),
        ):
            self.module.cmd_test(self.paths, self.args)

        lit_calls = [call for call in popen_calls if call.cmd[0] == str(self.paths.llvm_lit)]
        filter_calls = [call for call in popen_calls if call.cmd[0] == sys.executable]
        self.assertEqual(len(lit_calls), 3)
        self.assertEqual(len(filter_calls), 3)
        self.assertEqual(min(wait_seen_launch_counts), 4)
        self.assertEqual(max(wait_seen_launch_counts), 6)
        self.assertEqual(
            {call.cmd[1] for call in filter_calls},
            {str(self.paths.root / "test" / "lit_top_slowest.py")},
        )
        broad_call = lit_calls[0]
        heavy_call = lit_calls[1]
        perf_call = lit_calls[2]
        self.assertEqual(broad_call.cmd[-1], str(self.paths.loom_build / "test"))
        self.assertIn("--filter-out", broad_call.cmd)
        broad_filter = broad_call.cmd[broad_call.cmd.index("--filter-out") + 1]
        self.assertIn("techmap/perf", broad_filter)
        self.assertIn("artifacts/cmsis_cgra_status_rollup\\.mlir", broad_filter)
        self.assertIn("-j7", broad_call.cmd)
        self.assertEqual(broad_call.kwargs["env"]["LOOM_TEST_JOBS"], "7")
        self.assertEqual(broad_call.kwargs["env"]["LOOM_ARTIFACT_TEST_JOBS"], "3")
        self.assertIn("-j2", heavy_call.cmd)
        self.assertEqual(heavy_call.kwargs["env"]["LOOM_TEST_JOBS"], "2")
        self.assertEqual(heavy_call.kwargs["env"]["LOOM_ARTIFACT_TEST_JOBS"], "2")
        self.assertIn(str(self.paths.loom_build / "test" / "artifacts" / "cmsis_cgra_status_rollup.mlir"), heavy_call.cmd)
        self.assertIn("-j1", perf_call.cmd)
        self.assertEqual(perf_call.cmd[-1], str(self.paths.loom_build / "test" / "techmap" / "perf"))

    def test_cmd_test_uses_explicit_lit_worker_budget_for_nested_runners(self):
        popen_calls = []
        self.args.jobs = 32

        class FakePipe:
            def close(self):
                pass

        class FakeProcess:
            def __init__(self, cmd, **kwargs):
                self.cmd = cmd
                self.kwargs = kwargs
                self.stdout = FakePipe()
                popen_calls.append(self)

            def wait(self):
                return 0

        with (
            patch.dict("os.environ", {"LIT_OPTS": "-j1"}),
            patch.object(self.module, "check_git_version"),
            patch.object(self.module, "check_loom_compilers"),
            patch.object(self.module, "build_loom"),
            patch.object(subprocess, "Popen", side_effect=lambda cmd, **kwargs: FakeProcess(cmd, **kwargs)),
        ):
            self.module.cmd_test(self.paths, self.args)

        lit_calls = [call for call in popen_calls if call.cmd[0] == str(self.paths.llvm_lit)]
        self.assertEqual(len(lit_calls), 3)
        broad_call = lit_calls[0]
        heavy_call = lit_calls[1]
        self.assertIn("-j1", broad_call.cmd)
        self.assertNotIn("-j32", broad_call.cmd)
        self.assertEqual(broad_call.kwargs["env"]["LOOM_TEST_JOBS"], "1")
        self.assertIn("-j1", heavy_call.cmd)
        self.assertEqual(heavy_call.kwargs["env"]["LOOM_TEST_JOBS"], "1")

    def test_cmd_test_scales_heavy_artifact_lanes_for_large_budget(self):
        popen_calls = []
        self.args.jobs = 24

        class FakePipe:
            def close(self):
                pass

        class FakeProcess:
            def __init__(self, cmd, **kwargs):
                self.cmd = cmd
                self.kwargs = kwargs
                self.stdout = FakePipe()
                popen_calls.append(self)

            def wait(self):
                return 0

        with (
            patch.object(self.module, "check_git_version"),
            patch.object(self.module, "check_loom_compilers"),
            patch.object(self.module, "build_loom"),
            patch.object(subprocess, "Popen", side_effect=lambda cmd, **kwargs: FakeProcess(cmd, **kwargs)),
        ):
            self.module.cmd_test(self.paths, self.args)

        lit_calls = [call for call in popen_calls if call.cmd[0] == str(self.paths.llvm_lit)]
        self.assertEqual(len(lit_calls), 3)
        heavy_call = lit_calls[1]
        self.assertIn("-j4", heavy_call.cmd)
        self.assertEqual(lit_calls[0].kwargs["env"]["LOOM_ARTIFACT_TEST_JOBS"], "8")
        self.assertEqual(heavy_call.kwargs["env"]["LOOM_TEST_JOBS"], "4")
        self.assertEqual(heavy_call.kwargs["env"]["LOOM_ARTIFACT_TEST_JOBS"], "4")


if __name__ == "__main__":
    unittest.main()
