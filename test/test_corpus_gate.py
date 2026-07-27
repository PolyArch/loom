#!/usr/bin/env python3
"""Anchor tests for the unified corpus gate.

Anchored contracts: inventory aggregation from test/corpus_inventory.py,
exact target/sysroot command construction, deterministic ordered results,
per-case deadline process-group cleanup, and no failure-as-pass.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

import corpus_gate  # noqa: E402
import corpus_inventory  # noqa: E402


VALID_LL = (
    '; ModuleID = "stub.c"\n'
    'source_filename = "stub.c"\n'
    f"{corpus_gate.LLVM_DATALAYOUT_LINE}\n"
    f"{corpus_gate.LLVM_TRIPLE_LINE}\n"
)
VALID_S0 = (
    f"module attributes {{{corpus_gate.S0_TRIPLE_ATTRIBUTE}}} {{\n}}\n"
)

STUB = """#!/usr/bin/env python3
import os
import subprocess
import sys
import time

log = os.environ.get("STUB_LOG")
if log:
    with open(log, "a") as handle:
        handle.write(" ".join(sys.argv) + "\\n")

args = sys.argv[1:]
out = args[args.index("-o") + 1] if "-o" in args else None

fail_suffix = os.environ.get("STUB_FAIL_SUFFIX")
if fail_suffix and out and out.endswith(fail_suffix):
    sys.exit(1)
fail_input = os.environ.get("STUB_FAIL_INPUT_SUFFIX")
if fail_input and any(
    not arg.startswith("-") and arg != out and arg.endswith(fail_input)
    for arg in args
):
    sys.exit(1)

if os.environ.get("STUB_BEHAVIOR") == "sleep":
    with open(os.environ["STUB_PGID_FILE"], "w") as handle:
        handle.write(str(os.getpid()))
    child = subprocess.Popen(["sleep", "60"])
    time.sleep(60)
    child.wait()
    sys.exit(0)

if out and not out.endswith(os.devnull):
    corrupt = os.environ.get("STUB_CORRUPT_SUFFIX")
    if corrupt and out.endswith(corrupt):
        content = "garbage\\n"
    elif out.endswith(".ll"):
        content = VALID_LL
    elif out.endswith(".mlir"):
        content = VALID_S0
    else:
        content = "garbage\\n"
    with open(out, "w") as handle:
        handle.write(content)
sys.exit(0)
"""

TOOL_ENV = ("LOOM_CC", "LOOM_CXX", "LOOM_RAISE", "LOOM_RAISE_OPT")
SYSROOT_ENV = (
    corpus_gate.ENV_SYSROOT,
    corpus_gate.ENV_GCC_TOOLCHAIN,
)


def make_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class CorpusGateTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.tools = self.work / "tools"
        self.tools.mkdir()
        stubbed = STUB.replace("VALID_LL", repr(VALID_LL)).replace(
            "VALID_S0", repr(VALID_S0)
        )
        self.tool_paths: dict[str, str] = {}
        for key, name in (
            ("cc", "stub-cc"),
            ("cxx", "stub-cxx"),
            ("raise", "stub-raise"),
            ("opt", "stub-opt"),
        ):
            path = self.tools / name
            make_executable(path, stubbed)
            self.tool_paths[key] = str(path)
        self.sysroot = self.work / "sysroot"
        (self.sysroot / "include").mkdir(parents=True)
        (self.sysroot / "include" / "stdint.h").write_text("/* stub */\n")
        self.gcc_toolchain = self.work / "gcc-toolchain"
        (
            self.gcc_toolchain / "lib" / "gcc" / corpus_gate.TARGET_TRIPLE
        ).mkdir(parents=True)
        self.out_dir = self.work / "out"
        self.log = self.work / "invocations.log"

        environment = {
            "LOOM_CC": self.tool_paths["cc"],
            "LOOM_CXX": self.tool_paths["cxx"],
            "LOOM_RAISE": self.tool_paths["raise"],
            "LOOM_RAISE_OPT": self.tool_paths["opt"],
            "STUB_LOG": str(self.log),
        }
        for name in (*SYSROOT_ENV, "STUB_BEHAVIOR", "STUB_FAIL_SUFFIX",
                     "STUB_FAIL_INPUT_SUFFIX", "STUB_CORRUPT_SUFFIX",
                     "STUB_PGID_FILE"):
            environment[name] = ""
        patcher = mock.patch.dict(os.environ, environment)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in (*SYSROOT_ENV, "STUB_BEHAVIOR", "STUB_FAIL_SUFFIX",
                     "STUB_FAIL_INPUT_SUFFIX", "STUB_CORRUPT_SUFFIX",
                     "STUB_PGID_FILE"):
            os.environ.pop(name, None)

    def run_gate(self, *arguments: str) -> tuple[int, str, dict[str, object]]:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = corpus_gate.main(
                [
                    *arguments,
                    "--sysroot",
                    str(self.sysroot),
                    "--gcc-toolchain",
                    str(self.gcc_toolchain),
                    "--out-dir",
                    str(self.out_dir),
                ]
            )
        summary = json.loads((self.out_dir / "summary.json").read_text())
        return exit_code, stdout.getvalue(), summary

    def invocation_lines(self) -> list[str]:
        return self.log.read_text().splitlines()


class InventoryAggregationTest(CorpusGateTestBase):
    def test_gate_selection_is_the_inventory(self) -> None:
        inventory = corpus_inventory.load_inventory(corpus_inventory.ROOT)
        for suite in corpus_inventory.SUITE_ORDER:
            exit_code, _, summary = self.run_gate(
                "--suite", suite, "--stage", "llvm", "--jobs", "4"
            )
            self.assertEqual(exit_code, 0)
            expected = [case for case in inventory if case.suite == suite]
            self.assertEqual(summary["case_count"], len(expected))
            self.assertEqual(
                [case["identity"] for case in summary["cases"]],
                [case.identity for case in expected],
            )
            self.assertEqual(summary["suite_counts"], {suite: {"pass": len(expected), "fail": 0}})

    def test_loombench_package_runs_every_source_as_one_case(self) -> None:
        exit_code, _, summary = self.run_gate(
            "--case", "loombench:axpy", "--stage", "s0", "--jobs", "1"
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["case_count"], 1)
        result = summary["cases"][0]
        self.assertEqual(result["identity"], "loombench:axpy")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["sources"], 2)

        invocations = self.invocation_lines()
        compiled = [line for line in invocations if "-emit-llvm" in line]
        raised = [line for line in invocations if "stub-raise " in line]
        verified = [line for line in invocations if "stub-opt " in line]
        self.assertEqual(len(compiled), 2)
        self.assertEqual(len(raised), 2)
        self.assertEqual(len(verified), 2)
        for source in ("main_func.cpp", "main_inline.cpp"):
            self.assertTrue(
                any(
                    f"test/app/axpy/{source}" in line
                    and line.split(" ", 1)[0].endswith("stub-cxx")
                    for line in compiled
                ),
                f"missing compile invocation for {source}",
            )
        # Each source is its own program: nothing links the two mains.
        self.assertFalse(any("main_func.cpp" in line and "main_inline.cpp" in line for line in invocations))

    def test_unknown_case_selector_is_a_configuration_error(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stderr(stdout):
            exit_code = corpus_gate.main(["--case", "loombench:no-such-case"])
        self.assertEqual(exit_code, 2)
        self.assertIn("unknown case selector", stdout.getvalue())


class CommandConstructionTest(CorpusGateTestBase):
    def toolchain(self) -> corpus_gate.Toolchain:
        return corpus_gate.Toolchain(
            cc=self.tool_paths["cc"],
            cxx=self.tool_paths["cxx"],
            raise_tool=self.tool_paths["raise"],
            raise_opt=self.tool_paths["opt"],
            sysroot=self.sysroot,
            gcc_toolchain=self.gcc_toolchain,
        )

    def test_exact_target_and_sysroot_compile_command(self) -> None:
        toolchain = self.toolchain()
        external_root = self.work / "externals"
        source = external_root / "cmsis-dsp" / "Source" / "BasicMathFunctions" / "arm_abs_f32.c"
        output = self.out_dir / "arm_abs_f32.ll"
        command = corpus_gate.compile_command(
            toolchain,
            corpus_gate.suite_compile_flags("cmsis-dsp", external_root),
            source,
            output,
        )
        self.assertEqual(
            command,
            [
                self.tool_paths["cc"],
                "--target=riscv64-unknown-elf",
                "-march=rv64im",
                "-mabi=lp64",
                f"--sysroot={self.sysroot}",
                f"--gcc-toolchain={self.gcc_toolchain}",
                "-D__GNUC_PYTHON__",
                f"-I{external_root / 'cmsis-dsp' / 'Include'}",
                f"-I{external_root / 'cmsis-dsp' / 'PrivateInclude'}",
                "-emit-llvm",
                "-S",
                "-O1",
                str(source),
                "-o",
                str(output),
            ],
        )

    def test_cxx_source_uses_cxx_driver(self) -> None:
        toolchain = self.toolchain()
        command = corpus_gate.compile_command(
            toolchain,
            [],
            ROOT / "test" / "app" / "axpy" / "main_func.cpp",
            self.out_dir / "main_func.ll",
        )
        self.assertEqual(command[0], self.tool_paths["cxx"])

    def namespace(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "sysroot": None,
            "gcc_toolchain": None,
            "riscv_gcc": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def make_fake_gcc(self) -> tuple[str, Path]:
        root = self.work / "fake-toolchain-root"
        sysroot = root / corpus_gate.TARGET_TRIPLE
        (sysroot / "include").mkdir(parents=True)
        (sysroot / "include" / "stdint.h").write_text("/* newlib */\n")
        install = root / "lib" / "gcc" / corpus_gate.TARGET_TRIPLE / "15.2.0"
        install.mkdir(parents=True)
        gcc = root / "bin" / corpus_gate.RISCV_GCC_NAME
        gcc.parent.mkdir(parents=True)
        make_executable(
            gcc,
            "#!/usr/bin/env bash\n"
            'if [[ "$*" == "-print-sysroot" ]]; then\n'
            f'  echo "{root}/bin/../{corpus_gate.TARGET_TRIPLE}"\n'
            'elif [[ "$*" == "-print-search-dirs" ]]; then\n'
            f'  echo "install: {root}/bin/../lib/gcc/{corpus_gate.TARGET_TRIPLE}/15.2.0/"\n'
            "else\n"
            "  exit 1\n"
            "fi\n",
        )
        return str(gcc), root

    def test_sysroot_and_toolchain_derive_from_real_gcc(self) -> None:
        gcc, root = self.make_fake_gcc()
        toolchain = corpus_gate.resolve_toolchain(self.namespace(riscv_gcc=gcc))
        self.assertEqual(toolchain.sysroot, root / corpus_gate.TARGET_TRIPLE)
        self.assertEqual(toolchain.gcc_toolchain, root)

    def test_explicit_sysroot_and_toolchain_win_over_derivation(self) -> None:
        gcc, root = self.make_fake_gcc()
        toolchain = corpus_gate.resolve_toolchain(
            self.namespace(
                sysroot=str(self.sysroot),
                gcc_toolchain=str(self.gcc_toolchain),
                riscv_gcc=gcc,
            )
        )
        self.assertEqual(toolchain.sysroot, self.sysroot.resolve())
        self.assertEqual(toolchain.gcc_toolchain, self.gcc_toolchain.resolve())
        toolchain = corpus_gate.resolve_toolchain(
            self.namespace(sysroot=str(self.sysroot), riscv_gcc=gcc)
        )
        self.assertEqual(toolchain.sysroot, self.sysroot.resolve())
        self.assertEqual(toolchain.gcc_toolchain, root)

    def test_unavailable_toolchain_fails_honestly(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": str(self.tools)}):
            with self.assertRaisesRegex(
                corpus_gate.GateConfigError, "RISC-V cross sysroot"
            ):
                corpus_gate.resolve_toolchain(self.namespace())
        with self.assertRaisesRegex(
            corpus_gate.GateConfigError, "not a configured RISC-V cross sysroot"
        ):
            corpus_gate.resolve_toolchain(
                self.namespace(
                    sysroot=str(self.work / "empty-sysroot"),
                    gcc_toolchain=str(self.gcc_toolchain),
                )
            )


class DeterministicResultsTest(CorpusGateTestBase):
    def normalize_human(self, text: str) -> str:
        text = re.sub(r"\(\d+ source\(s\), \d+\.\d+s\)", "(SOURCES)", text)
        text = re.sub(r"in \d+\.\d+s", "in DURATION", text)
        return re.sub(r"jobs=\d+", "jobs=N", text)

    def test_repeated_runs_and_job_counts_preserve_results(self) -> None:
        selection = [
            "--case",
            "loombench:axpy",
            "--case",
            "cmsis-dsp:BasicMathFunctions/arm_abs_f32.c",
            "--case",
            "cmsis-nn:ConvolutionFunctions/arm_convolve_s8.c",
        ]
        runs: list[tuple[str, list[str], list[str]]] = []
        for jobs in ("1", "4", "1"):
            exit_code, human, summary = self.run_gate(
                *selection, "--stage", "s0", "--jobs", jobs
            )
            self.assertEqual(exit_code, 0)
            identities = [case["identity"] for case in summary["cases"]]
            statuses = [case["status"] for case in summary["cases"]]
            runs.append((self.normalize_human(human), identities, statuses))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])
        self.assertEqual(
            runs[0][1],
            [
                "loombench:axpy",
                "cmsis-dsp:BasicMathFunctions/arm_abs_f32.c",
                "cmsis-nn:ConvolutionFunctions/arm_convolve_s8.c",
            ],
        )


class TimeoutCleanupTest(CorpusGateTestBase):
    def test_deadline_kills_the_case_process_group(self) -> None:
        pgid_file = self.work / "pgid"
        os.environ["STUB_BEHAVIOR"] = "sleep"
        os.environ["STUB_PGID_FILE"] = str(pgid_file)
        started = time.monotonic()
        exit_code, _, summary = self.run_gate(
            "--case",
            "cmsis-dsp:BasicMathFunctions/arm_abs_f32.c",
            "--stage",
            "llvm",
            "--jobs",
            "1",
            "--case-timeout",
            "0.5",
        )
        elapsed = time.monotonic() - started
        self.assertEqual(exit_code, 1)
        self.assertLess(elapsed, 30.0)
        result = summary["cases"][0]
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["category"], "timeout")
        self.assertIn("killed process group", result["detail"])

        pgid = int(pgid_file.read_text().strip())
        # The whole group, including the grandchild sleep, is gone. Killed
        # grandchildren may linger as zombies for a moment, so poll briefly.
        group_gone = False
        for _ in range(100):
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                group_gone = True
                break
            time.sleep(0.05)
        self.assertTrue(group_gone, f"process group {pgid} leaked")
        os.environ.pop("STUB_BEHAVIOR", None)


class NoFailureAsPassTest(CorpusGateTestBase):
    def assert_every_requested_case_fails(
        self, exit_code: int, summary: dict[str, object], category: str
    ) -> None:
        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["failed"], summary["case_count"])
        self.assertEqual(summary["failure_categories"], {category: summary["case_count"]})
        for case in summary["cases"]:
            self.assertEqual(case["status"], "fail")
            self.assertEqual(case["category"], category)
            self.assertIsInstance(case["detail"], str)

    def test_compile_failure_fails_the_gate(self) -> None:
        os.environ["STUB_FAIL_SUFFIX"] = ".ll"
        exit_code, human, summary = self.run_gate(
            "--case", "loombench:axpy", "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "compile")
        self.assertIn("[corpus-gate] FAIL", human)

    def test_fabricated_llvm_artifact_fails_the_gate(self) -> None:
        os.environ["STUB_CORRUPT_SUFFIX"] = ".ll"
        exit_code, _, summary = self.run_gate(
            "--case", "cmsis-dsp:BasicMathFunctions/arm_abs_f32.c", "--stage", "llvm"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "llvm-artifact")

    def test_fabricated_s0_artifact_fails_the_gate(self) -> None:
        os.environ["STUB_CORRUPT_SUFFIX"] = ".scf.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", "cmsis-dsp:BasicMathFunctions/arm_abs_f32.c", "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "s0-artifact")

    def test_raise_and_verify_failures_fail_the_gate(self) -> None:
        os.environ["STUB_FAIL_SUFFIX"] = ".scf.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", "cmsis-dsp:BasicMathFunctions/arm_abs_f32.c", "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "raise")

        os.environ.pop("STUB_FAIL_SUFFIX", None)
        os.environ["STUB_FAIL_INPUT_SUFFIX"] = ".scf.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", "cmsis-dsp:BasicMathFunctions/arm_abs_f32.c", "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "verify")


if __name__ == "__main__":
    unittest.main()
