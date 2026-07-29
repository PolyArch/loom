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
    f"module attributes {{{corpus_gate.MLIR_TRIPLE_ATTRIBUTE}}} {{\n}}\n"
)
VALID_D0 = VALID_S0
VALID_COUNTS = '{"actors": 0, "graphs": 0}\n'
VALID_DFG_REPORT = json.dumps(
    {
        "actors": 3,
        "dynamic_calls": 1,
        "event_count": 11,
        "floating_variance_bytes": 0,
        "floating_variance_kind": "none",
        "graphs": 1,
        "kind": "source_backed_dfg_comparison",
        "memory_bytes_compared": 64,
        "operation_firings": {"arith.addi": 4},
        "simulation_seconds": 0.0001,
        "status": "pass",
        "value_lanes_compared": 0,
        "wavefront_steps": 100,
        "wavefront_steps_per_second": 1_000_000.0,
    },
    sort_keys=True,
) + "\n"


class CorpusGateExecutionPolicyTest(unittest.TestCase):
    def test_defaults_reserve_development_cpus_and_bound_dfg_sim_time(self) -> None:
        with mock.patch.object(corpus_gate.os, "cpu_count", return_value=32):
            self.assertEqual(corpus_gate.default_jobs(), 28)
        with mock.patch.object(corpus_gate.os, "cpu_count", return_value=256):
            self.assertEqual(corpus_gate.default_jobs(), 128)
        with mock.patch.object(corpus_gate.os, "cpu_count", return_value=4):
            self.assertEqual(corpus_gate.default_jobs(), 1)

        self.assertEqual(
            corpus_gate.parse_args(["--stage", "dfg-sim"]).case_timeout,
            15.0,
        )
        self.assertEqual(
            corpus_gate.parse_args(["--stage", "d0"]).case_timeout,
            120.0,
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
counts = next(
    (arg.split("=", 1)[1] for arg in args if arg.startswith("--counts=")),
    None,
)
canonical = next(
    (arg.split("=", 1)[1] for arg in args if arg.startswith("--canonical-output=")),
    None,
)
report = next(
    (arg.split("=", 1)[1] for arg in args if arg.startswith("--output=")),
    None,
)
bitcode_output = next(
    (arg.split("=", 1)[1] for arg in args if arg.startswith("--bitcode-output=")),
    None,
)

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

corrupt = os.environ.get("STUB_CORRUPT_SUFFIX")
if out and not out.endswith(os.devnull):
    if corrupt and out.endswith(corrupt):
        content = "garbage\\n"
    elif out.endswith(".ll"):
        content = VALID_LL
    elif out.endswith(".mlir"):
        content = VALID_D0 if out.endswith(".dfg.mlir") else VALID_S0
    else:
        content = "garbage\\n"
    with open(out, "w") as handle:
        handle.write(content)
    if any("save-temps=resolution" in arg for arg in args):
        with open(out + ".resolution.txt", "w") as handle:
            handle.write("selected-input.o\\n")
        with open(out + ".0.5.precodegen.bc", "wb") as handle:
            handle.write(b"stub bitcode")
if bitcode_output:
    with open(bitcode_output, "wb") as handle:
        handle.write(b"stub bitcode")
if counts:
    content = (
        "garbage\\n"
        if corrupt and counts.endswith(corrupt)
        else VALID_COUNTS
    )
    with open(counts, "w") as handle:
        handle.write(content)
if canonical:
    with open(canonical, "w") as handle:
        handle.write(VALID_D0)
if report:
    content = (
        "garbage\\n"
        if corrupt and report.endswith(corrupt)
        else VALID_DFG_REPORT
    )
    with open(report, "w") as handle:
        handle.write(content)
sys.exit(0)
"""

TOOL_ENV = (
    "LOOM_CC",
    "LOOM_CXX",
    "LOOM_RAISE",
    "LOOM_RAISE_OPT",
    "LOOM_PRE_MAPPING",
    "LOOM_DFG_RUN",
    "LOOM_LLD",
    "LOOM_PAYLOAD",
    "LOOM_LLVM_DIS",
    "LOOM_LLVM_LINK",
)
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
        stubbed = (
            STUB.replace("VALID_LL", repr(VALID_LL))
            .replace("VALID_S0", repr(VALID_S0))
            .replace("VALID_D0", repr(VALID_D0))
            .replace("VALID_COUNTS", repr(VALID_COUNTS))
            .replace("VALID_DFG_REPORT", repr(VALID_DFG_REPORT))
        )
        self.tool_paths: dict[str, str] = {}
        for key, name in (
            ("cc", "stub-cc"),
            ("cxx", "stub-cxx"),
            ("raise", "stub-raise"),
            ("opt", "stub-opt"),
            ("pre_mapping", "stub-pre-mapping"),
            ("dfg_run", "stub-dfg-run"),
            ("lld", "stub-lld"),
            ("payload", "stub-payload"),
            ("llvm_dis", "stub-llvm-dis"),
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
            "LOOM_PRE_MAPPING": self.tool_paths["pre_mapping"],
            "LOOM_DFG_RUN": self.tool_paths["dfg_run"],
            "LOOM_LLD": self.tool_paths["lld"],
            "LOOM_PAYLOAD": self.tool_paths["payload"],
            "LOOM_LLVM_DIS": self.tool_paths["llvm_dis"],
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
    def test_cmsis_nn_harness_materializes_owned_test_invocations(self) -> None:
        workloads = tuple(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity
            in {
                (
                    "cmsis-nn:test_arm_avgpool_s8/"
                    "test_arm_avgpool_s8__test_avgpooling_arm_avgpool_s8"
                ),
                (
                    "cmsis-nn:test_arm_avgpool_s8/"
                    "test_arm_avgpool_s8__test_avgpooling_1_arm_avgpool_s8"
                ),
            }
        )
        harness = corpus_gate.materialize_cmsis_nn_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-nn-harness",
        )

        self.assertEqual(
            harness.targets, tuple(workload.executable for workload in workloads)
        )
        self.assertEqual(
            harness.unity_source,
            corpus_inventory.resolve_externals_root(ROOT) / "unity",
        )
        for workload in workloads:
            runner = (
                harness.source_dir
                / "TestCases"
                / workload.executable
                / "Unity"
                / "TestRunner"
                / "unity_test_arm_avgpool_s8_runner.c"
            ).read_text()
            calls = re.findall(r"RUN_TEST\((test_[A-Za-z0-9_]+)\);", runner)
            self.assertEqual(calls, [workload.producer.test_function])

    def test_whole_program_stage_selects_workload_inventory(self) -> None:
        exit_code, _, summary = self.run_gate(
            "--case",
            "loombench:axpy/axpy_func",
            "--stage",
            "d0",
            "--jobs",
            "1",
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["target"]["code_model"], "medany")
        self.assertEqual(summary["tools"]["lld"], self.tool_paths["lld"])
        self.assertEqual(summary["tools"]["payload"], self.tool_paths["payload"])
        result = summary["cases"][0]
        self.assertEqual(result["identity"], "loombench:axpy/axpy_func")
        self.assertEqual(result["sources"], 1)
        self.assertTrue(
            any(
                "test/app/axpy/main_func.cpp" in line
                for line in self.invocation_lines()
            )
        )

    def test_unimplemented_workload_producer_fails_closed(self) -> None:
        exit_code, _, summary = self.run_gate(
            "--case",
            "cmsis-dsp:official-tests/scalar",
            "--stage",
            "d0",
            "--jobs",
            "1",
        )
        self.assertEqual(exit_code, 1)
        result = summary["cases"][0]
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["category"], "workload-provider-unavailable")
        self.assertIn("cmsis-dsp-test-framework", result["detail"])

    def test_produced_workload_import_uses_build_relative_link_records(self) -> None:
        toolchain = corpus_gate.Toolchain(
            cc=self.tool_paths["cc"],
            cxx=self.tool_paths["cxx"],
            raise_tool=self.tool_paths["raise"],
            raise_opt=self.tool_paths["opt"],
            pre_mapping=self.tool_paths["pre_mapping"],
            dfg_run=self.tool_paths["dfg_run"],
            lld=self.tool_paths["lld"],
            payload=self.tool_paths["payload"],
            llvm_dis=self.tool_paths["llvm_dis"],
            sysroot=self.sysroot,
            gcc_toolchain=self.gcc_toolchain,
        )
        build_dir = self.work / "target-build"
        executable = build_dir / "workloads" / "test_arm_avgpool_s8"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"elf")
        Path(f"{executable}.resolution.txt").write_text("selected.o\n")
        Path(f"{executable}.0.5.precodegen.bc").write_bytes(b"bitcode")
        case_dir = self.work / "case"
        case_dir.mkdir()

        prepared = corpus_gate.import_produced_workload(
            corpus_gate.ProducedWorkload(build_dir, executable),
            toolchain,
            case_dir,
            time.monotonic() + 5.0,
        )

        self.assertIsInstance(prepared, corpus_gate.LinkedWorkloadModules)
        invocation = next(
            line for line in self.invocation_lines() if "stub-payload " in line
        )
        self.assertIn(
            "--resolution=workloads/test_arm_avgpool_s8.resolution.txt",
            invocation,
        )
        self.assertNotIn(str(build_dir), invocation)

    def test_llvm_gate_selection_is_the_source_inventory(self) -> None:
        inventory = corpus_inventory.load_source_inventory(corpus_inventory.ROOT)
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

    def test_source_package_compiles_every_translation_unit(self) -> None:
        exit_code, _, summary = self.run_gate(
            "--case", "loombench:axpy", "--stage", "llvm", "--jobs", "1"
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["case_count"], 1)
        result = summary["cases"][0]
        self.assertEqual(result["identity"], "loombench:axpy")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["sources"], 2)

        invocations = self.invocation_lines()
        compiled = [line for line in invocations if "-emit-llvm" in line]
        self.assertEqual(len(compiled), 2)
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

    def test_d0_stage_runs_pre_mapping_per_workload(self) -> None:
        exit_code, _, summary = self.run_gate(
            "--case",
            "loombench:axpy/axpy_func",
            "--stage",
            "d0",
            "--jobs",
            "1",
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["case_count"], 1)
        result = summary["cases"][0]
        self.assertEqual(result["status"], "pass")
        # The structured counts keep a graph-free whole-program result
        # distinguishable from a nonempty Spatial graph.
        self.assertEqual(result["graphs"], 0)
        self.assertEqual(result["actors"], 0)

        invocations = self.invocation_lines()
        compiled = [line for line in invocations if "-emit-llvm" in line]
        linked = [line for line in invocations if "save-temps=resolution" in line]
        imported = [line for line in invocations if "stub-payload " in line]
        disassembled = [line for line in invocations if "stub-llvm-dis " in line]
        pre_mapped = [
            line for line in invocations if "stub-pre-mapping " in line
        ]
        self.assertEqual(len(compiled), 0)
        self.assertEqual(len(linked), 1)
        self.assertEqual(len(imported), 1)
        self.assertEqual(len(disassembled), 1)
        self.assertEqual(len(pre_mapped), 1)
        self.assertIn(f"-fuse-ld={self.tool_paths['lld']}", linked[0])
        self.assertIn("-ffat-lto-objects", invocations[0])
        for line in pre_mapped:
            self.assertIn("--builtin=small", line)
            self.assertIn("--artifact-store=", line)
            self.assertIn("--counts=", line)
            self.assertRegex(line, r"\S+/program\.ll ")
        # The d0 stage replaces the raise/verify chain with the single
        # production pre-Mapping path.
        self.assertFalse(any("stub-raise " in line for line in invocations))
        self.assertFalse(any("stub-opt " in line for line in invocations))

    def test_dfg_sim_stage_runs_source_backed_comparison_per_workload(self) -> None:
        exit_code, _, summary = self.run_gate(
            "--case",
            "loombench:axpy/axpy_func",
            "--stage",
            "dfg-sim",
            "--jobs",
            "1",
        )
        self.assertEqual(exit_code, 0)
        result = summary["cases"][0]
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["graphs"], 1)
        self.assertEqual(result["actors"], 3)
        self.assertEqual(result["dfg_simulation"]["dynamic_calls"], 1)
        self.assertEqual(result["dfg_simulation"]["memory_bytes_compared"], 64)
        self.assertEqual(
            result["dfg_simulation"]["wavefront_steps_per_second"], 1_000_000.0
        )

        invocations = self.invocation_lines()
        linked = [line for line in invocations if "save-temps=resolution" in line]
        simulated = [line for line in invocations if "stub-dfg-run " in line]
        self.assertEqual(len(linked), 1)
        self.assertEqual(len(simulated), 1)
        for line in simulated:
            self.assertIn("--builtin=small", line)
            self.assertNotIn("--native-llvm=", line)
            self.assertIn("--canonical-output=", line)
            self.assertIn("--output=", line)
            self.assertIn("--candidate-jobs=1", line)


class CommandConstructionTest(CorpusGateTestBase):
    def toolchain(self) -> corpus_gate.Toolchain:
        return corpus_gate.Toolchain(
            cc=self.tool_paths["cc"],
            cxx=self.tool_paths["cxx"],
            raise_tool=self.tool_paths["raise"],
            raise_opt=self.tool_paths["opt"],
            pre_mapping=self.tool_paths["pre_mapping"],
            dfg_run=self.tool_paths["dfg_run"],
            lld=self.tool_paths["lld"],
            payload=self.tool_paths["payload"],
            llvm_dis=self.tool_paths["llvm_dis"],
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
                "-march=rv64imafdc_zicsr_zifencei",
                "-mabi=lp64d",
                "-mcmodel=medany",
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

    def test_dfg_sim_command_binds_target_and_outputs(self) -> None:
        config = self.out_dir / "resolved-config.yaml"
        command = corpus_gate.dfg_sim_command(
            self.toolchain(),
            self.out_dir / "target.ll",
            self.out_dir / "store",
            self.out_dir / "program.dfg.mlir",
            self.out_dir / "simulation.json",
            3,
            config,
        )
        self.assertEqual(
            command,
            [
                self.tool_paths["dfg_run"],
                "--builtin=small",
                f"--config={config}",
                f"--artifact-store={self.out_dir / 'store'}",
                f"--canonical-output={self.out_dir / 'program.dfg.mlir'}",
                f"--output={self.out_dir / 'simulation.json'}",
                "--candidate-jobs=3",
                str(self.out_dir / "target.ll"),
            ],
        )

    def test_dfg_report_requires_exact_substantive_projection(self) -> None:
        report = self.out_dir / "simulation.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(VALID_DFG_REPORT)
        parsed, defect = corpus_gate.parse_dfg_simulation_report(report)
        self.assertIsNone(defect)
        self.assertEqual(parsed.graphs, 1)
        self.assertEqual(parsed.operation_firings, {"arith.addi": 4})

        payload = json.loads(VALID_DFG_REPORT)
        payload["memory_bytes_compared"] = 0
        payload["value_lanes_compared"] = 1
        report.write_text(json.dumps(payload))
        parsed, defect = corpus_gate.parse_dfg_simulation_report(report)
        self.assertIsNone(defect)
        self.assertEqual(parsed.value_lanes_compared, 1)

        payload["value_lanes_compared"] = 0
        report.write_text(json.dumps(payload))
        parsed, defect = corpus_gate.parse_dfg_simulation_report(report)
        self.assertIsNone(parsed)
        self.assertIn("substantive", defect)

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
            "loombench:axpy/axpy_func",
            "--case",
            "loombench:axpy/axpy_inline",
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
                "loombench:axpy/axpy_func",
                "loombench:axpy/axpy_inline",
            ],
        )

    def test_d0_repeated_runs_preserve_results_and_counts(self) -> None:
        selection = ["--case", "loombench:axpy/axpy_func"]
        runs: list[tuple[str, str, str, int, int]] = []
        for jobs in ("1", "4", "1"):
            exit_code, human, summary = self.run_gate(
                *selection, "--stage", "d0", "--jobs", jobs
            )
            self.assertEqual(exit_code, 0)
            case = summary["cases"][0]
            runs.append(
                (
                    self.normalize_human(human),
                    case["identity"],
                    case["status"],
                    case["graphs"],
                    case["actors"],
                )
            )
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])

    def test_d0_candidate_workers_are_forwarded_and_reported(self) -> None:
        config = self.work / "resolved-config.yaml"
        config.write_text(
            "dse:\n  structured_ownership:\n"
            "    scope_expansion_limit: 16\n"
        )
        exit_code, human, summary = self.run_gate(
            "--case",
            "loombench:axpy/axpy_func",
            "--stage",
            "d0",
            "--jobs",
            "1",
            "--candidate-jobs",
            "3",
            "--config",
            str(config),
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["candidate_jobs"], 3)
        self.assertEqual(summary["config"], str(config.resolve()))
        self.assertIn("candidate-jobs=3", human)
        self.assertIn(f"config={config.resolve()}", human)
        pre_mapping = [
            line
            for line in self.invocation_lines()
            if line.split(" ", 1)[0].endswith("stub-pre-mapping")
        ]
        self.assertEqual(len(pre_mapping), 1)
        self.assertTrue(
            all("--candidate-jobs=3" in invocation for invocation in pre_mapping)
        )
        self.assertTrue(
            all(
                f"--config={config.resolve()}" in invocation
                for invocation in pre_mapping
            )
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
        os.environ["STUB_FAIL_SUFFIX"] = ".o"
        exit_code, human, summary = self.run_gate(
            "--case", "loombench:axpy/axpy_func", "--stage", "s0"
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
            "--case", "loombench:axpy/axpy_func", "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "s0-artifact")

    def test_raise_and_verify_failures_fail_the_gate(self) -> None:
        os.environ["STUB_FAIL_SUFFIX"] = ".scf.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", "loombench:axpy/axpy_func", "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "raise")

        os.environ.pop("STUB_FAIL_SUFFIX", None)
        os.environ["STUB_FAIL_INPUT_SUFFIX"] = ".scf.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", "loombench:axpy/axpy_func", "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "verify")

    def test_pre_mapping_failure_fails_the_gate(self) -> None:
        os.environ["STUB_FAIL_SUFFIX"] = ".dfg.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", "loombench:axpy/axpy_func", "--stage", "d0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "pre-mapping")

    def test_fabricated_d0_artifact_fails_the_gate(self) -> None:
        os.environ["STUB_CORRUPT_SUFFIX"] = ".dfg.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", "loombench:axpy/axpy_func", "--stage", "d0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "d0-artifact")

    def test_malformed_d0_counts_fail_the_gate(self) -> None:
        os.environ["STUB_CORRUPT_SUFFIX"] = ".counts.json"
        exit_code, _, summary = self.run_gate(
            "--case", "loombench:axpy/axpy_func", "--stage", "d0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "d0-artifact")

    def test_d0_counts_reject_undeclared_fields(self) -> None:
        counts = self.work / "counts.json"
        counts.write_text('{"actors": 0, "graphs": 0, "other": 0}\n')
        parsed, defect = corpus_gate.parse_d0_counts(counts)
        self.assertIsNone(parsed)
        self.assertIn("unexpected fields", defect)


if __name__ == "__main__":
    unittest.main()
