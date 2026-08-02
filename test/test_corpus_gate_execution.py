#!/usr/bin/env python3
"""Anchor tests for corpus gate execution and failure handling."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unittest
from pathlib import Path
from unittest import mock

from test.test_corpus_gate import (
    AXPY_WORKLOAD_ID,
    ROOT,
    VALID_DFG_REPORT,
    VECADD_WORKLOAD_ID,
    CorpusGateTestBase,
    make_executable,
)

import corpus_gate


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
        source = (
            external_root
            / "cmsis-dsp"
            / "Source"
            / "BasicMathFunctions"
            / "arm_abs_f32.c"
        )
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
                "-gline-tables-only",
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

    def test_standard_float16_profile_preserves_target_and_uses_standard_type(
        self,
    ) -> None:
        toolchain = self.toolchain()
        external_root = self.work / "externals"
        source = (
            external_root
            / "cmsis-dsp"
            / "Source"
            / "BasicMathFunctions"
            / "arm_abs_f16.c"
        )
        command = corpus_gate.compile_command(
            toolchain,
            corpus_gate.suite_compile_flags(
                "cmsis-dsp",
                external_root,
                target_profile="riscv64-standard-float16",
            ),
            source,
            self.out_dir / "arm_abs_f16.ll",
        )

        self.assertIn("--target=riscv64-unknown-elf", command)
        self.assertIn("-march=rv64imafdc_zicsr_zifencei", command)
        self.assertIn("-mabi=lp64d", command)
        self.assertIn("-D__ARM_FP16_FORMAT_IEEE=1", command)
        self.assertIn("-D__fp16=_Float16", command)

    def test_dfg_sim_command_binds_target_and_outputs(self) -> None:
        config = self.out_dir / "resolved-config.yaml"
        protocol = ("arm_abs_f32",)
        command = corpus_gate.dfg_sim_command(
            self.toolchain(),
            self.out_dir / "target.ll",
            self.out_dir / "store",
            self.out_dir / "program.dfg.mlir",
            self.out_dir / "simulation.json",
            3,
            corpus_gate.DfgExecutionLimits(400, 500, 600),
            15.0,
            protocol,
            config,
            expected_entry_result=0,
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
                "--max-event-steps=400",
                "--max-event-count=500",
                "--max-capture-bytes=600",
                "--max-simulation-wall-seconds=15.0",
                "--expected-entry-result=0",
                "--operator-protocol-symbol=arm_abs_f32",
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
        self.assertEqual(parsed.canonical_dataflow_identity, "03" * 32)
        self.assertEqual(parsed.simulation_workload_identity, "04" * 32)
        self.assertEqual(parsed.simulation_runtime_input_identity, "05" * 32)
        self.assertEqual(parsed.execution_terminal, "retired")

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
        text = re.sub(
            r"\(\d+ source\(s\), \d+\.\d+s wall, \d+\.\d+s CPU, "
            r"\d+\.\d+ MiB peak RSS\)",
            "(SOURCES)",
            text,
        )
        text = re.sub(r"in \d+\.\d+s", "in DURATION", text)
        return re.sub(r"jobs=\d+", "jobs=N", text)

    def test_repeated_runs_and_job_counts_preserve_results(self) -> None:
        selection = [
            "--case",
            AXPY_WORKLOAD_ID,
            "--case",
            VECADD_WORKLOAD_ID,
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
                AXPY_WORKLOAD_ID,
                VECADD_WORKLOAD_ID,
            ],
        )

    def test_d0_repeated_runs_preserve_results_and_counts(self) -> None:
        selection = ["--case", AXPY_WORKLOAD_ID]
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
            "dse:\n  structured_ownership:\n    scope_expansion_limit: 16\n"
        )
        exit_code, human, summary = self.run_gate(
            "--case",
            AXPY_WORKLOAD_ID,
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
        self.assertEqual(
            summary["failure_categories"], {category: summary["case_count"]}
        )
        for case in summary["cases"]:
            self.assertEqual(case["status"], "fail")
            self.assertEqual(case["category"], category)
            self.assertIsInstance(case["detail"], str)

    def test_compile_failure_fails_the_gate(self) -> None:
        os.environ["STUB_FAIL_SUFFIX"] = ".o"
        exit_code, human, summary = self.run_gate(
            "--case", AXPY_WORKLOAD_ID, "--stage", "s0"
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
            "--case", AXPY_WORKLOAD_ID, "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "s0-artifact")

    def test_raise_and_verify_failures_fail_the_gate(self) -> None:
        os.environ["STUB_FAIL_SUFFIX"] = ".scf.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", AXPY_WORKLOAD_ID, "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "raise")

        os.environ.pop("STUB_FAIL_SUFFIX", None)
        os.environ["STUB_FAIL_INPUT_SUFFIX"] = ".scf.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", AXPY_WORKLOAD_ID, "--stage", "s0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "verify")

    def test_pre_mapping_failure_fails_the_gate(self) -> None:
        os.environ["STUB_FAIL_SUFFIX"] = ".dfg.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", AXPY_WORKLOAD_ID, "--stage", "d0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "pre-mapping")

    def test_fabricated_d0_artifact_fails_the_gate(self) -> None:
        os.environ["STUB_CORRUPT_SUFFIX"] = ".dfg.mlir"
        exit_code, _, summary = self.run_gate(
            "--case", AXPY_WORKLOAD_ID, "--stage", "d0"
        )
        self.assert_every_requested_case_fails(exit_code, summary, "d0-artifact")

    def test_malformed_d0_counts_fail_the_gate(self) -> None:
        os.environ["STUB_CORRUPT_SUFFIX"] = ".counts.json"
        exit_code, _, summary = self.run_gate(
            "--case", AXPY_WORKLOAD_ID, "--stage", "d0"
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
