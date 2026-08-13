#!/usr/bin/env python3
"""Anchor tests for the unified corpus gate.

Anchored contracts: inventory aggregation from test/corpus_inventory.py,
exact target/sysroot command construction, deterministic ordered results,
per-case deadline process-group cleanup, and no failure-as-pass.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import re
import signal
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
import corpus_workload_provider  # noqa: E402


_OPERATOR_WORKLOADS = corpus_inventory.load_workload_inventory(ROOT)
AXPY_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "loombench" and row.case == "axpy"
)
VECADD_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "loombench" and row.case == "vecadd"
)
DSP_ABS_F32_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-abs-f32"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_ABS_F16_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-abs-f16"
    and row.target_profile == "riscv64-standard-float16"
)
DSP_ADD_F32_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-add-f32"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_CLIP_F32_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-clip-f32"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_ADD_Q15_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-add-q15"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_DOT_Q31_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-dot-prod-q31"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_CLIP_Q7_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-clip-q7"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_WELCH_F32_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-welch-f32"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_HAMMING_F64_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-hamming-f64"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_VEXP_F32_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-vexp-f32"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_ENTROPY_F32_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-entropy-f32"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_KL_F64_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-kullback-leibler-f64"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_SQRT_F32_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-sqrt-f32"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_MFCC_Q31_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-mfcc-q31"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_CLARKE_F32_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-clarke-f32"
    and row.target_profile == "riscv64-portable-scalar"
)
DSP_FIR_F32_WORKLOAD_ID = next(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if row.suite == "cmsis-dsp"
    and row.case == "arm-fir-f32"
    and row.target_profile == "riscv64-portable-scalar"
)
NN_CONVOLVE_1X1_S8_FAST_WORKLOAD_ID = (
    "cmsis-nn:arm-convolve-1x1-s8-fast:e4fc696adf47aaf4"
)
AVGPOOL_HARNESS_WORKLOAD_IDS = tuple(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if isinstance(row.producer, corpus_inventory.CmsisNnWorkloadProducer)
    and row.producer.target == "test_arm_avgpool_s8"
)[:2]
DUPLICATE_HARNESS_WORKLOAD_IDS = tuple(
    row.operator_id
    for row in _OPERATOR_WORKLOADS
    if isinstance(row.producer, corpus_inventory.CmsisNnWorkloadProducer)
    and row.producer.target == "test_arm_fully_connected_s8"
    and row.producer.test_function == "test_fc_per_fc_per_ch_arm_fully_connected_s8"
)
UNSUPPORTED_MVE_WORKLOAD_ID = next(
    row.operator_id for row in _OPERATOR_WORKLOADS if row.target_profile == "mve"
)


VALID_LL = (
    '; ModuleID = "stub.c"\n'
    'source_filename = "stub.c"\n'
    f"{corpus_gate.LLVM_DATALAYOUT_LINE}\n"
    f"{corpus_gate.LLVM_TRIPLE_LINE}\n"
)
VALID_S0 = f"module attributes {{{corpus_gate.MLIR_TRIPLE_ATTRIBUTE}}} {{\n}}\n"
VALID_D0 = VALID_S0
VALID_COUNT_VALUES = {
    "actors": 3,
    "central_generate_input_artifacts": 4,
    "central_generate_invocations": 6,
    "central_generate_lineage_edges": 5,
    "central_generate_output_artifacts": 7,
    "central_plan_executions": 2,
    "graphs": 1,
}
VALID_COUNTS = json.dumps(VALID_COUNT_VALUES, sort_keys=True) + "\n"
GRAPH_FREE_COUNTS = json.dumps(
    {**VALID_COUNT_VALUES, "actors": 0, "graphs": 0}, sort_keys=True
) + "\n"
VALID_DFG_REPORT = (
    json.dumps(
        {
            "actor_refs": [
                {"artifact": "03" * 32, "entity": "0"},
                {"artifact": "03" * 32, "entity": "1"},
                {"artifact": "03" * 32, "entity": "2"},
            ],
            "actors": 3,
            "artifacts": {
                "canonical_dataflow": "03" * 32,
                "canonical_dataflow_initial": "06" * 32,
                "simulation_runtime_input": "05" * 32,
                "simulation_workload": "04" * 32,
                "structured_initial": "07" * 32,
                "structured_selected": "08" * 32,
            },
            "compiler_target": {
                "data_layout": corpus_gate.LLVM_DATALAYOUT_LINE.split('"')[1],
                "host_binding": "01" * 32,
                "instruction_bindings": ["02" * 32],
                "instruction_core_count": 1,
                "target_triple": corpus_gate.LLVM_TRIPLE_LINE.split('"')[1],
            },
            "dynamic_calls": 1,
            "event_count": 11,
            "execution_terminal": "retired",
            "floating_variance_bytes": 0,
            "floating_variance_kind": "none",
            "graphs": 1,
            "kind": "source_backed_dfg_comparison",
            "memory_bytes_compared": 64,
            "operation_firings": {"arith.addi": 4},
            "selected_source_files": [
                str(ROOT / "test" / "app" / "axpy" / "main_func.cpp")
            ],
            "simulation_seconds": 0.0001,
            "source_oracle": {"comparison": "equivalent", "entry_result": 0},
            "status": "pass",
            "transform_lineage": {
                "dataflow_rewrite": [7],
                "execution_shape": 1,
                "memory_communication": [2],
                "ownership": 1,
                "schedule": 1,
                "special_math_accuracy": 0,
            },
            "value_lanes_compared": 0,
            "wavefront_steps": 100,
            "wavefront_steps_per_second": 1_000_000.0,
        },
        sort_keys=True,
    )
    + "\n"
)


class CorpusGateExecutionPolicyTest(unittest.TestCase):
    def test_step_rejects_a_process_tree_that_outlives_its_leader(self) -> None:
        with tempfile.TemporaryDirectory(prefix="loom-orphan-step-") as root:
            root_path = Path(root)
            pgid_path = root_path / "pgid"
            script = (
                "import os,subprocess; "
                f"open({str(pgid_path)!r},'w').write(str(os.getpid())); "
                "subprocess.Popen(['sleep','60'])"
            )
            failure = corpus_gate.run_step(
                [sys.executable, "-c", script],
                root_path / "step.log",
                time.monotonic() + 5.0,
                corpus_gate.CATEGORY_COMPILE,
            )
            pgid = int(pgid_path.read_text())
            try:
                self.assertIsNotNone(failure)
                self.assertEqual(failure.category, corpus_gate.CATEGORY_INTERNAL)
                self.assertIn("outlived its process-group leader", failure.detail)
            finally:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_dfg_report_rejects_noncanonical_compiler_target_projection(self) -> None:
        payload = json.loads(VALID_DFG_REPORT)
        payload["compiler_target"]["instruction_bindings"] *= 2
        with tempfile.TemporaryDirectory(prefix="loom-target-report-") as root:
            report_path = Path(root) / "report.json"
            report_path.write_text(json.dumps(payload))
            report, defect = corpus_gate.parse_dfg_simulation_report(report_path)

        self.assertIsNone(report)
        self.assertIn("invalid compiler target", defect)

    def test_dfg_report_rejects_noncanonical_artifact_identity_projection(
        self,
    ) -> None:
        payload = json.loads(VALID_DFG_REPORT)
        payload["artifacts"]["canonical_dataflow"] = "03" * 31
        with tempfile.TemporaryDirectory(prefix="loom-artifact-report-") as root:
            report_path = Path(root) / "report.json"
            report_path.write_text(json.dumps(payload))
            report, defect = corpus_gate.parse_dfg_simulation_report(report_path)

        self.assertIsNone(report)
        self.assertIn("invalid artifact identities", defect)

    def test_dfg_report_rejects_foreign_actor_reference(self) -> None:
        payload = json.loads(VALID_DFG_REPORT)
        payload["actor_refs"][0]["artifact"] = "09" * 32
        with tempfile.TemporaryDirectory(prefix="loom-actor-report-") as root:
            report_path = Path(root) / "report.json"
            report_path.write_text(json.dumps(payload))
            report, defect = corpus_gate.parse_dfg_simulation_report(report_path)

        self.assertIsNone(report)
        self.assertIn("invalid ActorRef", defect)

    def test_defaults_reserve_development_cpus_and_bound_dfg_sim_time(self) -> None:
        with mock.patch.object(corpus_gate.os, "cpu_count", return_value=32):
            self.assertEqual(corpus_gate.default_jobs(), 28)
        with mock.patch.object(corpus_gate.os, "cpu_count", return_value=256):
            self.assertEqual(corpus_gate.default_jobs(), 120)
        with mock.patch.object(corpus_gate.os, "cpu_count", return_value=4):
            self.assertEqual(corpus_gate.default_jobs(), 1)

        self.assertEqual(
            corpus_gate.parse_args(["--stage", "dfg-sim"]).case_timeout,
            30.0,
        )
        self.assertEqual(
            corpus_gate.parse_args(["--stage", "dfg-sim"]).dfg_simulation_timeout,
            15.0,
        )
        self.assertEqual(
            corpus_gate.parse_args(["--stage", "d0"]).case_timeout,
            120.0,
        )
        dfg_args = corpus_gate.parse_args(["--stage", "dfg-sim"])
        self.assertEqual(dfg_args.dfg_max_wavefront_steps, 1_000_000)
        self.assertEqual(dfg_args.dfg_max_event_count, 10_000_000)
        self.assertEqual(dfg_args.dfg_max_capture_bytes, 256 * 1024 * 1024)

    def test_dfg_case_slots_account_for_linked_module_pressure(self) -> None:
        one_source = mock.Mock(sources=("a.c",))
        five_sources = mock.Mock(sources=tuple(f"s{i}.c" for i in range(5)))
        nine_sources = mock.Mock(sources=tuple(f"s{i}.c" for i in range(9)))

        self.assertEqual(corpus_gate.case_resource_slots(one_source, "dfg-sim", 28), 1)
        self.assertEqual(
            corpus_gate.case_resource_slots(five_sources, "dfg-sim", 28), 5
        )
        self.assertEqual(
            corpus_gate.case_resource_slots(nine_sources, "dfg-sim", 28), 9
        )
        self.assertEqual(corpus_gate.case_resource_slots(nine_sources, "d0", 28), 1)
        self.assertEqual(corpus_gate.case_resource_slots(nine_sources, "dfg-sim", 2), 2)


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
        owner = next((arg for arg in args if arg.endswith(".o")), "selected-input.o")
        with open(out + ".resolution.txt", "w") as handle:
            handle.write(owner + "\\n")
            handle.write(f"-r={owner},kernel,pl\\n")
        with open(out + ".0.5.precodegen.bc", "wb") as handle:
            handle.write(b"stub bitcode")
if bitcode_output:
    with open(bitcode_output, "wb") as handle:
        handle.write(b"stub bitcode")
if counts:
    if os.environ.get("STUB_GRAPH_FREE"):
        content = GRAPH_FREE_COUNTS
    else:
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
            .replace("GRAPH_FREE_COUNTS", repr(GRAPH_FREE_COUNTS))
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
        (self.gcc_toolchain / "lib" / "gcc" / corpus_gate.TARGET_TRIPLE).mkdir(
            parents=True
        )
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
        for name in (
            *SYSROOT_ENV,
            "STUB_BEHAVIOR",
            "STUB_FAIL_SUFFIX",
            "STUB_FAIL_INPUT_SUFFIX",
            "STUB_CORRUPT_SUFFIX",
            "STUB_PGID_FILE",
            "STUB_GRAPH_FREE",
        ):
            environment[name] = ""
        patcher = mock.patch.dict(os.environ, environment)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in (
            *SYSROOT_ENV,
            "STUB_BEHAVIOR",
            "STUB_FAIL_SUFFIX",
            "STUB_FAIL_INPUT_SUFFIX",
            "STUB_CORRUPT_SUFFIX",
            "STUB_PGID_FILE",
            "STUB_GRAPH_FREE",
        ):
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

    def test_cmsis_nn_harness_materializes_owned_test_invocations(self) -> None:
        workloads = tuple(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity in AVGPOOL_HARNESS_WORKLOAD_IDS
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
            self.assertEqual(
                harness.protocol_symbols(workload.executable),
                (workload.producer.test_function,),
            )
            runner_path = (
                harness.source_dir
                / "TestCases"
                / workload.executable
                / "Unity"
                / "TestRunner"
                / "unity_test_arm_avgpool_s8_runner.c"
            )
            runner = runner_path.read_text()
            calls = re.findall(r"RUN_TEST\((test_[A-Za-z0-9_]+)\);", runner)
            self.assertEqual(calls, [workload.producer.test_function])
            compiled_owner = harness.protocol_source(workload.executable)
            self.assertEqual(
                compiled_owner,
                runner_path.parent.parent / runner_path.name.replace("_runner", ""),
            )
        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        for workload in workloads:
            self.assertIn(
                f"target_compile_options({workload.executable} PRIVATE "
                "-fno-inline-functions)",
                cmake,
            )

    def test_cmsis_nn_harness_isolates_profiles_of_the_same_test(self) -> None:
        workloads = tuple(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity in DUPLICATE_HARNESS_WORKLOAD_IDS
        )
        self.assertEqual(len(workloads), 2)

        harness = corpus_gate.materialize_cmsis_nn_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-nn-profile-harness",
        )

        self.assertEqual(len(harness.targets), 2)
        self.assertEqual(len(set(harness.targets)), 2)
        for workload, target in zip(workloads, harness.targets, strict=True):
            self.assertTrue(target.endswith(workload.identity.rsplit(":", 1)[-1]))

    def test_cmsis_nn_convolution_uses_direct_protocol_and_official_oracle(
        self,
    ) -> None:
        workload = next(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity == NN_CONVOLVE_1X1_S8_FAST_WORKLOAD_ID
        )

        harness = corpus_gate.materialize_cmsis_nn_harness(
            (workload,),
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-nn-convolution-harness",
        )

        self.assertEqual(
            harness.protocol_symbols(workload.executable),
            ("loom_corpus_operator_protocol",),
        )
        self.assertEqual(harness.expected_entry_result(workload.executable), 0)
        compiled_owner = harness.protocol_source(workload.executable)
        self.assertEqual(compiled_owner.name, "OperatorProtocol.c")
        source = compiled_owner.read_text()
        protocol = source.split("int main(void)", maxsplit=1)[0]
        self.assertIn("arm_convolve_1x1_s8_fast_get_buffer_size", protocol)
        self.assertIn("arm_convolve_1x1_s8_fast", protocol)
        self.assertIn('"TestCases/TestData/kernel1x1/test_data.h"', source)
        self.assertIn("kernel1x1_output_ref", source)
        self.assertIn(
            "output[index] != kernel1x1_output_ref[index]",
            source,
        )
        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        self.assertNotIn("LOOM_UNITY_SOURCE", cmake)
        self.assertNotIn("target_link_libraries", cmake)
        for source_path in workload.sources:
            self.assertIn(source_path.removeprefix("externals/cmsis-nn/"), cmake)

    def test_cmsis_dsp_abs_uses_direct_protocol_and_official_oracle(self) -> None:
        workload = next(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity == DSP_ABS_F32_WORKLOAD_ID
        )

        harness = corpus_gate.materialize_cmsis_dsp_harness(
            (workload,),
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-dsp-harness",
        )

        self.assertEqual(harness.targets, (workload.executable,))
        self.assertEqual(
            harness.protocol_symbols(workload.executable),
            ("loom_corpus_operator_protocol",),
        )
        self.assertEqual(harness.expected_entry_result(workload.executable), 0)
        compiled_owner = harness.protocol_source(workload.executable)
        self.assertEqual(compiled_owner.name, "OperatorProtocol.cpp")
        self.assertEqual(harness.inline_definitions(workload.executable), ())
        source = compiled_owner.read_text()
        protocol = source.split("int main()", maxsplit=1)[0]
        self.assertIn("arm_abs_f32(input, output, count)", protocol)
        self.assertIn(
            "for (std::uint32_t index = 0; index < kSampleCount; ++index)", source
        )
        self.assertIn("std::memcmp(output, kExpected, sizeof(kExpected))", source)
        self.assertIn("return oracle_matches(output) ? 0 : 1;", source)
        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        self.assertNotIn("Testing/Source/Tests/BasicTestsF32.cpp", cmake)
        self.assertNotIn("Testing/testmain.cpp", cmake)
        configure = corpus_gate.cmake_configure_command(
            harness,
            corpus_inventory.resolve_externals_root(ROOT) / "cmsis-dsp",
            self.work / "cmsis-dsp-build",
            corpus_gate.CmakeToolchain(
                c_compiler="cc",
                cxx_compiler="c++",
                archiver=Path("ar"),
                ranlib=Path("ranlib"),
                compiler_flags=(),
                linker_flags=(),
            ),
        )
        self.assertTrue(
            any(flag.startswith("-DLOOM_CMSIS_DSP_SOURCE=") for flag in configure)
        )
        self.assertFalse(
            any(flag.startswith("-DLOOM_CMSIS_NN_SOURCE=") for flag in configure)
        )

    def test_cmsis_dsp_source_options_do_not_depend_on_case_selection(self) -> None:
        inventory = corpus_inventory.load_workload_inventory(ROOT)
        mfcc = next(
            workload for workload in inventory if workload.case == "arm-mfcc-f32"
        )
        vlog = next(
            workload for workload in inventory if workload.case == "arm-vlog-f32"
        )
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        mfcc_only = corpus_gate.materialize_cmsis_dsp_harness(
            (mfcc,), external_root, self.work / "cmsis-dsp-mfcc-only"
        )
        with_vlog = corpus_gate.materialize_cmsis_dsp_harness(
            (mfcc, vlog), external_root, self.work / "cmsis-dsp-mfcc-vlog"
        )

        def source_options(harness: corpus_gate.CmsisDspHarness) -> tuple[str, ...]:
            cmake = (harness.source_dir / "CMakeLists.txt").read_text()
            return tuple(
                line
                for line in cmake.splitlines()
                if line.startswith("set_property(SOURCE ")
            )

        selected_alone = source_options(mfcc_only)
        self.assertEqual(selected_alone, source_options(with_vlog))
        self.assertTrue(any("arm_vlog_f32.c" in line for line in selected_alone))

    def test_cmsis_dsp_basic_f32_uses_exact_direct_family(self) -> None:
        workloads = tuple(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity in {DSP_ADD_F32_WORKLOAD_ID, DSP_CLIP_F32_WORKLOAD_ID}
        )
        by_identity = {workload.identity: workload for workload in workloads}

        harness = corpus_gate.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-dsp-add-harness",
        )

        workload = by_identity[DSP_ADD_F32_WORKLOAD_ID]
        self.assertEqual(
            harness.protocol_symbols(workload.executable),
            ("loom_corpus_operator_protocol",),
        )
        self.assertEqual(harness.expected_entry_result(workload.executable), 0)
        source = harness.protocol_source(workload.executable).read_text()
        protocol = source.split("int main()", maxsplit=1)[0]
        self.assertIn("arm_add_f32(input_a, input_b, output, count)", protocol)
        self.assertIn("kInputA", source)
        self.assertIn("kInputB", source)
        self.assertIn("kExpected", source)
        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        self.assertNotIn("Testing/Source/Tests/BasicTestsF32.cpp", cmake)
        self.assertNotIn("Testing/testmain.cpp", cmake)

        clip = by_identity[DSP_CLIP_F32_WORKLOAD_ID]
        clip_source = harness.protocol_source(clip.executable).read_text()
        self.assertIn("constexpr std::uint32_t kSampleCount = 259;", clip_source)
        self.assertIn("input_a, output, -0.5f, -0.1f, kSampleCount", clip_source)

    def test_cmsis_dsp_basic_integer_uses_exact_direct_family(self) -> None:
        selected = {
            DSP_ADD_Q15_WORKLOAD_ID,
            DSP_DOT_Q31_WORKLOAD_ID,
            DSP_CLIP_Q7_WORKLOAD_ID,
        }
        workloads = tuple(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity in selected
        )
        by_identity = {workload.identity: workload for workload in workloads}

        harness = corpus_gate.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-dsp-basic-integer-harness",
        )

        for workload in workloads:
            self.assertEqual(
                harness.protocol_symbols(workload.executable),
                ("loom_corpus_operator_protocol",),
            )
            self.assertEqual(harness.expected_entry_result(workload.executable), 0)

        add_source = harness.protocol_source(
            by_identity[DSP_ADD_Q15_WORKLOAD_ID].executable
        ).read_text()
        self.assertIn("arm_add_q15(input_a, input_b, output, count)", add_source)
        self.assertIn("kAbsoluteError = 2;", add_source)
        self.assertIn("within_absolute_error", add_source)

        dot_source = harness.protocol_source(
            by_identity[DSP_DOT_Q31_WORKLOAD_ID].executable
        ).read_text()
        self.assertIn("q63_t *output", dot_source)
        self.assertIn("arm_dot_prod_q31(input_a, input_b, count", dot_source)
        self.assertIn("kAbsoluteError = 131072;", dot_source)

        clip_source = harness.protocol_source(
            by_identity[DSP_CLIP_Q7_WORKLOAD_ID].executable
        ).read_text()
        self.assertIn("input_a, output, -64, -13, kSampleCount", clip_source)

        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        self.assertNotIn("Testing/Source/Tests/BasicTestsQ15.cpp", cmake)
        self.assertNotIn("Testing/Source/Tests/BasicTestsQ31.cpp", cmake)
        self.assertNotIn("Testing/Source/Tests/BasicTestsQ7.cpp", cmake)

    def test_cmsis_dsp_window_uses_exact_direct_family(self) -> None:
        selected = {
            DSP_WELCH_F32_WORKLOAD_ID,
            DSP_HAMMING_F64_WORKLOAD_ID,
        }
        workloads = tuple(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity in selected
        )
        by_identity = {workload.identity: workload for workload in workloads}
        self.assertEqual(by_identity[DSP_WELCH_F32_WORKLOAD_ID].compiler_flags, ())
        self.assertEqual(
            by_identity[DSP_HAMMING_F64_WORKLOAD_ID].compiler_flags,
            ("-fno-math-errno",),
        )

        harness = corpus_gate.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-dsp-window-harness",
        )

        for workload in workloads:
            self.assertEqual(
                harness.protocol_symbols(workload.executable),
                ("loom_corpus_operator_protocol",),
            )
            self.assertEqual(harness.expected_entry_result(workload.executable), 0)

        welch_source = harness.protocol_source(
            by_identity[DSP_WELCH_F32_WORKLOAD_ID].executable
        ).read_text()
        self.assertIn("arm_welch_f32(output, count)", welch_source)
        self.assertIn("kAbsoluteError = 2.0e-6f", welch_source)
        self.assertIn("kSampleCount = 128", welch_source)

        hamming_source = harness.protocol_source(
            by_identity[DSP_HAMMING_F64_WORKLOAD_ID].executable
        ).read_text()
        self.assertIn("arm_hamming_f64(output, count)", hamming_source)
        self.assertIn("kAbsoluteError = 3.0e-15", hamming_source)

        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        hamming_operator_source = Path(
            by_identity[DSP_HAMMING_F64_WORKLOAD_ID].sources[0]
        ).relative_to("externals/cmsis-dsp")
        self.assertIn(
            f'"${{LOOM_CMSIS_DSP_SOURCE}}/{hamming_operator_source.as_posix()}" '
            "TARGET_DIRECTORY CMSISDSP APPEND PROPERTY "
            'COMPILE_OPTIONS "-fno-math-errno")',
            cmake,
        )
        self.assertNotIn("Testing/Source/Tests/WindowTestsF32.cpp", cmake)
        self.assertNotIn("Testing/Source/Tests/WindowTestsF64.cpp", cmake)

    def test_cmsis_dsp_elementary_math_uses_exact_direct_family(self) -> None:
        selected = {
            DSP_VEXP_F32_WORKLOAD_ID,
            DSP_ENTROPY_F32_WORKLOAD_ID,
            DSP_KL_F64_WORKLOAD_ID,
            DSP_SQRT_F32_WORKLOAD_ID,
        }
        workloads = tuple(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity in selected
        )
        by_identity = {workload.identity: workload for workload in workloads}
        for workload in workloads:
            self.assertEqual(workload.compiler_flags, ("-fno-math-errno",))

        harness = corpus_gate.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-dsp-elementary-math-harness",
        )

        for workload in workloads:
            self.assertEqual(
                harness.protocol_symbols(workload.executable),
                ("loom_corpus_operator_protocol",),
            )
            self.assertEqual(harness.expected_entry_result(workload.executable), 0)

        vexp = by_identity[DSP_VEXP_F32_WORKLOAD_ID]
        vexp_compiled = harness.protocol_source(vexp.executable)
        vexp_source = vexp_compiled.read_text()
        self.assertIn("arm_vexp_f32(input, output, count)", vexp_source)
        self.assertIn("constexpr float32_t kExpected[]", vexp_source)

        entropy = by_identity[DSP_ENTROPY_F32_WORKLOAD_ID]
        entropy_source = harness.protocol_source(entropy.executable).read_text()
        self.assertIn(
            "arm_entropy_f32(input + offset, dimensions[index])", entropy_source
        )
        self.assertIn("constexpr std::uint32_t kDimensions[]", entropy_source)

        divergence = by_identity[DSP_KL_F64_WORKLOAD_ID]
        divergence_source = harness.protocol_source(divergence.executable).read_text()
        self.assertIn(
            "arm_kullback_leibler_f64(input_a + offset, input_b + offset,",
            divergence_source,
        )

        sqrt = by_identity[DSP_SQRT_F32_WORKLOAD_ID]
        sqrt_source = harness.protocol_source(sqrt.executable).read_text()
        self.assertIn("arm_sqrt_f32(input[index], &output[index])", sqrt_source)
        self.assertIn("status[index]", sqrt_source)
        self.assertIn("ARM_MATH_ARGUMENT_ERROR", sqrt_source)

        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        self.assertNotIn("Testing/Source/Tests/FastMathF32.cpp", cmake)
        self.assertNotIn("Testing/Source/Tests/StatsTestsF32.cpp", cmake)
        self.assertNotIn("Testing/Source/Tests/StatsTestsF64.cpp", cmake)
        for workload in workloads:
            self.assertIn(
                f"target_compile_options({workload.executable} PRIVATE "
                "-fno-inline-functions -fno-math-errno)",
                cmake,
            )

    def test_cmsis_dsp_harness_links_upstream_operator_support_data(self) -> None:
        workload = next(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity == DSP_MFCC_Q31_WORKLOAD_ID
        )

        harness = corpus_gate.materialize_cmsis_dsp_harness(
            (workload,),
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-dsp-mfcc-harness",
        )

        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        self.assertIn('Testing/Source/Tests/mfccdata.c"', cmake)

    def test_cmsis_dsp_benchmark_vector_generates_direct_protocol(self) -> None:
        workload = next(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity == DSP_CLARKE_F32_WORKLOAD_ID
        )

        harness = corpus_gate.materialize_cmsis_dsp_harness(
            (workload,),
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-dsp-direct-harness",
        )

        self.assertEqual(
            harness.protocol_symbols(workload.executable),
            ("loom_corpus_operator_protocol",),
        )
        self.assertIsNone(harness.expected_entry_result(workload.executable))
        compiled_owner = harness.protocol_source(workload.executable)
        self.assertEqual(compiled_owner.name, "OperatorProtocol.cpp")
        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        self.assertNotIn("Testing/Source/Benchmarks/ControllerF32.cpp", cmake)
        source = compiled_owner.read_text()
        self.assertIn('extern "C" LOOM_NOINLINE void', source)
        self.assertIn("arm_clarke_f32(input_a[index], input_b[index]", source)
        self.assertIn("output_a[index]", source)
        self.assertIn("output_b[index]", source)
        self.assertIn("float32_t output_a[kSampleCount]{};", source)
        self.assertIn("float32_t output_b[kSampleCount]{};", source)
        self.assertNotIn("ControllerF32", source)

    def test_cmsis_dsp_stateful_fir_keeps_init_and_execution_atomic(self) -> None:
        workload = next(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity == DSP_FIR_F32_WORKLOAD_ID
        )

        harness = corpus_gate.materialize_cmsis_dsp_harness(
            (workload,),
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "cmsis-dsp-fir-harness",
        )

        self.assertEqual(
            harness.protocol_symbols(workload.executable),
            ("loom_corpus_operator_protocol",),
        )
        self.assertEqual(harness.expected_entry_result(workload.executable), 0)
        compiled_owner = harness.protocol_source(workload.executable)
        self.assertEqual(compiled_owner.name, "OperatorProtocol.cpp")
        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        self.assertNotIn("Testing/Source/Tests/FIRF32.cpp", cmake)
        source = compiled_owner.read_text()
        protocol = source.split("int main()", maxsplit=1)[0]
        self.assertIn("arm_fir_init_f32", protocol)
        self.assertEqual(protocol.count("arm_fir_f32("), 2)
        self.assertIn("kExpected", source)
        self.assertIn("return oracle_matches(output) ? 0 : 1;", source)

    def test_provider_build_failure_is_isolated_to_missing_targets(self) -> None:
        workloads = tuple(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity in (DSP_ABS_F32_WORKLOAD_ID, DSP_MFCC_Q31_WORKLOAD_ID)
        )
        self.assertEqual(len(workloads), 2)
        harness = corpus_workload_provider.CmsisDspHarness(
            self.work / "source",
            tuple(workload.executable for workload in workloads),
            tuple(self.work / f"shared-{index}" for index in range(2)),
            tuple(
                (workload.producer.test_class, workload.producer.test_method)
                for workload in workloads
            ),
            ((), ()),
            tuple(
                self.work / f"protocol-{index}.cpp" for index in range(len(workloads))
            ),
        )
        target_build = self.out_dir / "_providers" / "cmsis-dsp" / "target"

        def run_provider_step(command, log_path, deadline, category):
            del log_path, deadline, category
            if "--build" not in command:
                return None
            executable = harness.executable(target_build, workloads[0].executable)
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"elf")
            return corpus_gate.StepFailure(
                corpus_gate.CATEGORY_FINAL_LINK,
                "one provider target failed",
            )

        with (
            mock.patch.object(
                corpus_gate,
                "materialize_cmsis_dsp_harness",
                return_value=harness,
            ),
            mock.patch.object(corpus_gate, "run_step", side_effect=run_provider_step),
        ):
            results = corpus_gate.prepare_workload_providers(
                workloads,
                self.toolchain(),
                corpus_inventory.resolve_externals_root(ROOT),
                self.out_dir,
                jobs=2,
                timeout=5.0,
            )

        self.assertIsInstance(
            results[workloads[0].identity], corpus_gate.ProducedWorkload
        )
        self.assertEqual(
            results[workloads[0].identity].protocol_symbols,
            tuple(call.symbol for call in workloads[0].protocol),
        )
        self.assertIsInstance(results[workloads[1].identity], corpus_gate.StepFailure)

    def test_provider_requires_an_atomic_wrapper_for_multi_call_protocol(self) -> None:
        workload = next(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity == DSP_FIR_F32_WORKLOAD_ID
        )
        self.assertGreater(len(workload.protocol), 1)
        harness = corpus_workload_provider.CmsisDspHarness(
            self.work / "source",
            (workload.executable,),
            (self.work / "shared",),
            ((workload.producer.test_class, workload.producer.test_method),),
            ((),),
            (self.work / "protocol.cpp",),
        )
        target_build = self.out_dir / "_providers" / "cmsis-dsp" / "target"

        def run_provider_step(command, log_path, deadline, category):
            del log_path, deadline, category
            if "--build" in command:
                executable = harness.executable(target_build, workload.executable)
                executable.parent.mkdir(parents=True)
                executable.write_bytes(b"elf")
            return None

        with (
            mock.patch.object(
                corpus_gate,
                "materialize_cmsis_dsp_harness",
                return_value=harness,
            ),
            mock.patch.object(corpus_gate, "run_step", side_effect=run_provider_step),
        ):
            result = corpus_gate.prepare_workload_providers(
                (workload,),
                self.toolchain(),
                corpus_inventory.resolve_externals_root(ROOT),
                self.out_dir,
                jobs=2,
                timeout=5.0,
            )[workload.identity]

        self.assertIsInstance(result, corpus_gate.StepFailure)
        self.assertEqual(result.category, corpus_gate.CATEGORY_FINAL_LINK_ARTIFACT)
        self.assertIn("atomic protocol wrapper", result.detail)

    def test_standard_float16_provider_uses_an_isolated_profile_build(self) -> None:
        workload = next(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity == DSP_ABS_F16_WORKLOAD_ID
        )
        harness = corpus_workload_provider.CmsisDspHarness(
            self.work / "source",
            (workload.executable,),
            (self.work / "shared",),
            ((workload.producer.test_class, workload.producer.test_method),),
            ((workload.protocol[0].symbol,),),
            (self.work / "protocol.cpp",),
        )
        configure_commands: list[list[str]] = []

        def run_provider_step(command, log_path, deadline, category):
            del log_path, deadline, category
            if "--build" not in command:
                configure_commands.append(command)
                return None
            build_dir = Path(command[command.index("--build") + 1])
            executable = harness.executable(build_dir, workload.executable)
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"elf")
            return None

        with (
            mock.patch.object(
                corpus_gate,
                "materialize_cmsis_dsp_harness",
                return_value=harness,
            ),
            mock.patch.object(corpus_gate, "run_step", side_effect=run_provider_step),
        ):
            result = corpus_gate.prepare_workload_providers(
                (workload,),
                self.toolchain(),
                corpus_inventory.resolve_externals_root(ROOT),
                self.out_dir,
                jobs=2,
                timeout=5.0,
            )[workload.identity]

        self.assertIsInstance(result, corpus_gate.ProducedWorkload)
        self.assertEqual(len(configure_commands), 1)
        configure = configure_commands[0]
        self.assertTrue(
            any("-D__ARM_FP16_FORMAT_IEEE=1" in argument for argument in configure)
        )
        self.assertTrue(any("-D__fp16=_Float16" in argument for argument in configure))

    def test_standard_float16_harness_enables_float16_library_sources(self) -> None:
        workload = next(
            case
            for case in corpus_inventory.load_workload_inventory(ROOT)
            if case.identity == DSP_ABS_F16_WORKLOAD_ID
        )

        harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
            (workload,),
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "f16-harness",
        )

        cmake = (harness.source_dir / "CMakeLists.txt").read_text()
        self.assertIn('set(DISABLEFLOAT16 OFF CACHE BOOL "" FORCE)', cmake)
        self.assertNotIn('set(DISABLEFLOAT16 ON CACHE BOOL "" FORCE)', cmake)

    def test_incompatible_target_profile_is_typed_unsupported(self) -> None:
        exit_code, human, summary = self.run_gate(
            "--case",
            UNSUPPORTED_MVE_WORKLOAD_ID,
            "--stage",
            "d0",
            "--jobs",
            "1",
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["unsupported"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["cases"][0]["status"], "unsupported")
        self.assertEqual(
            summary["cases"][0]["category"],
            corpus_gate.CATEGORY_TARGET_PROFILE_UNSUPPORTED,
        )
        self.assertIn("requires arm", summary["cases"][0]["detail"])
        self.assertIn("1 unsupported", human)

    def test_whole_program_stage_selects_workload_inventory(self) -> None:
        exit_code, _, summary = self.run_gate(
            "--case",
            AXPY_WORKLOAD_ID,
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
        self.assertEqual(result["identity"], AXPY_WORKLOAD_ID)
        self.assertEqual(result["sources"], 1)
        self.assertTrue(
            any(
                "test/app/axpy/main_func.cpp" in line
                for line in self.invocation_lines()
            )
        )

    def test_produced_workload_import_uses_build_relative_link_records(self) -> None:
        toolchain = self.toolchain()
        build_dir = self.work / "target-build"
        executable = build_dir / "workloads" / "test_arm_avgpool_s8"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"elf")
        Path(f"{executable}.resolution.txt").write_text("selected.o\n")
        Path(f"{executable}.0.5.precodegen.bc").write_bytes(b"bitcode")
        (build_dir / "compile_commands.json").write_text("[]\n")
        case_dir = self.work / "case"
        case_dir.mkdir()

        prepared = corpus_gate.import_produced_workload(
            corpus_gate.ProducedWorkload(
                target_build_dir=build_dir,
                target_executable=executable,
                protocol_symbols=("protocol",),
            ),
            toolchain,
            case_dir,
            time.monotonic() + 5.0,
            corpus_gate.CaseResourceUsage(),
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

    def test_selected_source_projection_uses_spatial_provenance(self) -> None:
        kernel_object = self.work / "kernel.o"
        wrapper_object = self.work / "wrapper.o"
        kernel_source = ROOT / "test" / "app" / "vecadd" / "main_func.cpp"
        wrapper_source = self.work / "generated" / "unity_runner.c"
        linked = corpus_gate.LinkedWorkloadModules(
            target=self.work / "program.ll",
            resolution=self.work / "program.resolution.txt",
            link_root=self.work,
            object_sources=(
                (kernel_object, kernel_source),
                (wrapper_object, wrapper_source),
            ),
        )

        selected, defect = corpus_gate.resolve_selected_corpus_sources(
            linked,
            (str(kernel_source), str(wrapper_source)),
            corpus_inventory.resolve_externals_root(ROOT),
            ROOT,
            frozenset({"test/app/vecadd/main_func.cpp"}),
        )

        self.assertIsNone(defect)
        self.assertEqual(selected, ("test/app/vecadd/main_func.cpp",))

    def test_selected_source_projection_rejects_harness_only(self) -> None:
        wrapper_object = self.work / "wrapper.o"
        wrapper_source = self.work / "generated" / "unity_runner.c"
        linked = corpus_gate.LinkedWorkloadModules(
            target=self.work / "program.ll",
            resolution=self.work / "program.resolution.txt",
            link_root=self.work,
            object_sources=((wrapper_object, wrapper_source),),
        )

        selected, defect = corpus_gate.resolve_selected_corpus_sources(
            linked,
            (str(wrapper_source),),
            corpus_inventory.resolve_externals_root(ROOT),
            ROOT,
            frozenset({"externals/cmsis-nn/Source/kernel.c"}),
        )

        self.assertIsNone(selected)
        self.assertIn("does not cover an exact corpus source row", defect)

    def test_selected_source_projection_accepts_exact_inline_definition(self) -> None:
        wrapper_object = self.work / "wrapper.o"
        wrapper_source = self.work / "generated" / "operator_protocol.c"
        inline_definition = (
            corpus_inventory.resolve_externals_root(ROOT)
            / "cmsis-nn"
            / "Include"
            / "arm_nnsupportfunctions.h"
        )
        linked = corpus_gate.LinkedWorkloadModules(
            target=self.work / "program.ll",
            resolution=self.work / "program.resolution.txt",
            link_root=self.work,
            object_sources=((wrapper_object, wrapper_source),),
            inline_definition_sources=(inline_definition,),
        )

        selected, defect = corpus_gate.resolve_selected_corpus_sources(
            linked,
            (str(wrapper_source), str(inline_definition)),
            corpus_inventory.resolve_externals_root(ROOT),
            ROOT,
            frozenset(),
        )

        self.assertIsNone(defect)
        self.assertEqual(
            selected,
            ("externals/cmsis-nn/Include/arm_nnsupportfunctions.h",),
        )

    def test_inline_definition_cannot_alias_missing_provenance(self) -> None:
        wrapper_object = self.work / "wrapper.o"
        wrapper_source = self.work / "generated" / "operator_protocol.c"
        inline_definition = (
            corpus_inventory.resolve_externals_root(ROOT)
            / "cmsis-nn"
            / "Include"
            / "arm_nnsupportfunctions.h"
        )
        linked = corpus_gate.LinkedWorkloadModules(
            target=self.work / "program.ll",
            resolution=self.work / "program.resolution.txt",
            link_root=self.work,
            object_sources=((wrapper_object, wrapper_source),),
            inline_definition_sources=(inline_definition,),
        )

        selected, defect = corpus_gate.resolve_selected_corpus_sources(
            linked,
            (str(wrapper_source),),
            corpus_inventory.resolve_externals_root(ROOT),
            ROOT,
            frozenset(),
        )

        self.assertIsNone(selected)
        self.assertIn("does not cover an exact corpus source", defect)

    def test_selected_source_projection_has_no_owner_alias(self) -> None:
        parameters = inspect.signature(
            corpus_gate.resolve_selected_corpus_sources
        ).parameters

        self.assertEqual(
            tuple(parameters),
            (
                "linked",
                "selected_source_files",
                "external_root",
                "repo_root",
                "allowed_sources",
            ),
        )

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
            self.assertEqual(
                summary["suite_counts"],
                {suite: {"pass": len(expected), "unsupported": 0, "fail": 0}},
            )

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
        self.assertFalse(
            any(
                "main_func.cpp" in line and "main_inline.cpp" in line
                for line in invocations
            )
        )

    def test_unknown_case_selector_is_a_configuration_error(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stderr(stdout):
            exit_code = corpus_gate.main(["--case", "loombench:no-such-case"])
        self.assertEqual(exit_code, 2)
        self.assertIn("unknown case selector", stdout.getvalue())

    def test_d0_stage_runs_pre_mapping_per_workload(self) -> None:
        exit_code, _, summary = self.run_gate(
            "--case",
            AXPY_WORKLOAD_ID,
            "--stage",
            "d0",
            "--jobs",
            "1",
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["case_count"], 1)
        result = summary["cases"][0]
        self.assertEqual(result["status"], "pass")
        self.assertGreaterEqual(result["cpu_seconds"], 0.0)
        self.assertGreater(result["peak_resident_bytes"], 0)
        self.assertGreaterEqual(summary["cpu_seconds"], result["cpu_seconds"])
        self.assertEqual(summary["peak_resident_bytes"], result["peak_resident_bytes"])
        self.assertEqual(result["graphs"], 1)
        self.assertEqual(result["actors"], 3)

        invocations = self.invocation_lines()
        compiled = [line for line in invocations if "-emit-llvm" in line]
        linked = [line for line in invocations if "save-temps=resolution" in line]
        imported = [line for line in invocations if "stub-payload " in line]
        disassembled = [line for line in invocations if "stub-llvm-dis " in line]
        pre_mapped = [line for line in invocations if "stub-pre-mapping " in line]
        self.assertEqual(len(compiled), 0)
        self.assertEqual(len(linked), 1)
        self.assertEqual(len(imported), 1)
        self.assertEqual(len(disassembled), 1)
        self.assertEqual(len(pre_mapped), 1)
        self.assertIn(f"-fuse-ld={self.tool_paths['lld']}", linked[0])
        self.assertIn("-ffat-lto-objects", invocations[0])
        for line in pre_mapped:
            self.assertIn("--artifact-store=", line)
            self.assertIn("--counts=", line)
            self.assertRegex(line, r"\S+/program\.ll ")
        # The d0 stage replaces the raise/verify chain with the single
        # production pre-Mapping path.
        self.assertFalse(any("stub-raise " in line for line in invocations))
        self.assertFalse(any("stub-opt " in line for line in invocations))

    def test_d0_stage_rejects_graph_free_workload(self) -> None:
        with mock.patch.dict(os.environ, {"STUB_GRAPH_FREE": "1"}):
            exit_code, _, summary = self.run_gate(
                "--case",
                AXPY_WORKLOAD_ID,
                "--stage",
                "d0",
                "--jobs",
                "1",
            )

        self.assertEqual(exit_code, 1)
        result = summary["cases"][0]
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["category"], corpus_gate.CATEGORY_PRE_MAPPING)
        self.assertIn("no nonempty Spatial graph", result["detail"])

    def test_dfg_sim_stage_runs_source_backed_comparison_per_workload(self) -> None:
        exit_code, _, summary = self.run_gate(
            "--case",
            AXPY_WORKLOAD_ID,
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
            result["dfg_simulation"]["artifacts"],
            {
                "canonical_dataflow": "03" * 32,
                "simulation_runtime_input": "05" * 32,
                "simulation_workload": "04" * 32,
            },
        )
        self.assertEqual(result["dfg_simulation"]["execution_terminal"], "retired")
        self.assertEqual(
            result["dfg_simulation"]["selected_source_files"],
            [str(ROOT / "test" / "app" / "axpy" / "main_func.cpp")],
        )
        self.assertEqual(result["selected_sources"], ["test/app/axpy/main_func.cpp"])
        self.assertEqual(
            result["dfg_simulation"]["wavefront_steps_per_second"], 1_000_000.0
        )

        invocations = self.invocation_lines()
        linked = [line for line in invocations if "save-temps=resolution" in line]
        simulated = [line for line in invocations if "stub-dfg-run " in line]
        self.assertEqual(len(linked), 1)
        self.assertEqual(len(simulated), 1)
        for line in simulated:
            self.assertNotIn("--native-llvm=", line)
            self.assertIn("--canonical-output=", line)
            self.assertIn("--output=", line)
            self.assertIn("--candidate-jobs=1", line)


if __name__ == "__main__":
    unittest.main()
