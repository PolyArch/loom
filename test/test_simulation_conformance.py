#!/usr/bin/env python3
"""Anchor tests for paired Spatial/System simulation conformance policy."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

import simulation_conformance  # noqa: E402


class PairedSimulationBudgetTest(unittest.TestCase):
    def test_warmed_median_and_floor_define_the_system_budget(self) -> None:
        tiny = simulation_conformance.paired_system_budget(
            [0.002, 0.003, 0.004],
            spatial_absolute_budget_seconds=15.0,
        )
        ordinary = simulation_conformance.paired_system_budget(
            [0.8, 1.0, 1.2],
            spatial_absolute_budget_seconds=15.0,
        )
        capped = simulation_conformance.paired_system_budget(
            [20.0, 21.0, 22.0],
            spatial_absolute_budget_seconds=15.0,
        )

        self.assertEqual(tiny.spatial_reference_seconds, 0.1)
        self.assertAlmostEqual(tiny.system_budget_seconds, 0.3)
        self.assertEqual(ordinary.spatial_reference_seconds, 1.0)
        self.assertEqual(ordinary.system_budget_seconds, 3.0)
        self.assertEqual(capped.system_budget_seconds, 45.0)

    def test_paired_result_keeps_budget_rate_and_hard_ratio_distinct(self) -> None:
        budget = simulation_conformance.paired_system_budget(
            [1.0, 1.0, 1.0],
            spatial_absolute_budget_seconds=15.0,
        )
        within = simulation_conformance.evaluate_paired_execution(
            budget,
            simulation_conformance.ActiveExecutionTiming(
                active_wall_seconds=2.5,
                reference_cycles=500_000,
                bridge_message_count=17,
                accelerator_invocation_count=3,
                peak_resident_bytes=4096,
            ),
        )
        slow = simulation_conformance.evaluate_paired_execution(
            budget,
            simulation_conformance.ActiveExecutionTiming(
                active_wall_seconds=10.0,
                reference_cycles=500_000,
            ),
        )

        self.assertTrue(within.within_system_budget)
        self.assertFalse(within.hard_ratio_failure)
        self.assertEqual(within.reference_cycles_per_second, 200_000.0)
        self.assertTrue(within.meets_reference_rate_target)
        self.assertEqual(within.system_timing.bridge_message_count, 17)
        self.assertEqual(within.system_timing.accelerator_invocation_count, 3)
        self.assertEqual(within.system_timing.peak_resident_bytes, 4096)

        self.assertFalse(slow.within_system_budget)
        self.assertTrue(slow.hard_ratio_failure)
        self.assertFalse(slow.meets_reference_rate_target)

    def test_invalid_measurements_fail_closed(self) -> None:
        for samples in ([], [0.0], [-1.0], [math.inf], [math.nan]):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    simulation_conformance.paired_system_budget(samples, 15.0)

        with self.assertRaises(ValueError):
            simulation_conformance.ActiveExecutionTiming(0.0, 1)
        with self.assertRaises(ValueError):
            simulation_conformance.ActiveExecutionTiming(1.0, -1)

    def test_cgra_gate_budget_is_derived_from_the_complete_profile_suite(self) -> None:
        def reference(identity: int) -> dict[str, object]:
            return {
                "schema": "test.artifact",
                "schema_version": "1.0",
                "artifact": f"{identity:064x}",
            }

        operator_gate_sha256, operators = (
            simulation_conformance.load_cgra_representative_operators()
        )
        profiles: list[dict[str, object]] = []
        for ordinal, operator in enumerate(operators, start=1):
            measurement = {
                "active_wall_nanoseconds": 200_000_000,
                "active_process_cpu_nanoseconds": 190_000_000,
                "input_load_wall_nanoseconds": 1_000_000,
                "input_load_process_cpu_nanoseconds": 1_000_000,
                "engine_active_wall_nanoseconds": 198_000_000,
                "engine_active_process_cpu_nanoseconds": 188_000_000,
                "observation_projection_wall_nanoseconds": 1_000_000,
                "observation_projection_process_cpu_nanoseconds": 1_000_000,
                "artifact_publication_wall_nanoseconds": 100_000,
                "artifact_publication_process_cpu_nanoseconds": 100_000,
                "reference_cycles": 50_000,
                "event_frame_count": 100,
                "physical_request_count": 10,
                "physical_grant_count": 10,
                "physical_retirement_count": 10,
                "physical_grant_wait_cycle_sum": 4,
                "physical_grant_wait_cycle_max": 2,
                "physical_grant_delayed_count": 2,
                "evaluation_evidence": reference(600 + ordinal),
            }
            profiles.append(
                {
                    "schema": "loom.cgra_budget_profile.2",
                    "workload": operator.workload,
                    "operator_id": operator.operator_id,
                    "qualification_limit_nanoseconds": 45_000_000_000,
                    "warmup_runs": 1,
                    "measurement_runs": 3,
                    "batch_peak_resident_bytes": 4096,
                    "canonical_dataflow": reference(ordinal),
                    "simulation_workload": reference(100 + ordinal),
                    "simulation_runtime_input": reference(200 + ordinal),
                    "resolved_config": {
                        "schema": "loom.config.resolved",
                        "schema_version": "11.0",
                        "artifact": f"{300:064x}",
                    },
                    "fabric": reference(301),
                    "tech_mapping": reference(400 + ordinal),
                    "initial_spatial_mapping": reference(500 + ordinal),
                    "spatial_mapping": reference(500 + ordinal),
                    "repaired_spatial_mapping": None,
                    "parent_system_mapping": None,
                    "transport_repair_constraint": None,
                    "pre_repair_evidence": None,
                    "warmup_evidence": reference(700 + ordinal),
                    "measurements": [dict(measurement) for _ in range(3)],
                }
            )
        self.assertEqual(
            simulation_conformance.derive_cgra_spatial_budget_nanoseconds(profiles),
            500_000_000,
        )
        configuration = {
            "schema": "loom.cgra_simulation_gate.2",
            "policy": {
                "qualification_limit_nanoseconds": 45_000_000_000,
                "warmup_runs": 1,
                "measurement_runs": 3,
                "reference_rate_target_cycles_per_second": 100_000,
            },
            "operator_gate": {
                "path": simulation_conformance.CGRA_OPERATOR_GATE_RELATIVE_PATH,
                "sha256": operator_gate_sha256,
            },
            "spatial_absolute_budget_nanoseconds": 500_000_000,
            "profiles": profiles,
        }
        scratch_root = ROOT / "temp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(configuration), encoding="ascii")
            loaded = simulation_conformance.load_cgra_gate_configuration(path)
            self.assertEqual(loaded.spatial_absolute_budget_nanoseconds, 500_000_000)
            self.assertEqual(loaded.spatial_absolute_budget_seconds, 0.5)
            configuration["spatial_absolute_budget_nanoseconds"] = 500_000_001
            path.write_text(json.dumps(configuration), encoding="ascii")
            with self.assertRaises(ValueError):
                simulation_conformance.load_cgra_gate_configuration(path)
            configuration["spatial_absolute_budget_nanoseconds"] = 500_000_000
            operator_gate = configuration["operator_gate"]
            assert isinstance(operator_gate, dict)
            operator_gate["sha256"] = "0" * 64
            path.write_text(json.dumps(configuration), encoding="ascii")
            with self.assertRaises(ValueError):
                simulation_conformance.load_cgra_gate_configuration(path)

    def test_outer_worker_limit_reserves_cpus_and_obeys_memory_limit(self) -> None:
        self.assertEqual(
            simulation_conformance.outer_worker_limit(
                cpu_count=32, memory_derived_limit=128
            ),
            28,
        )
        self.assertEqual(
            simulation_conformance.outer_worker_limit(
                cpu_count=256, memory_derived_limit=80
            ),
            80,
        )
        self.assertEqual(
            simulation_conformance.outer_worker_limit(
                cpu_count=256, memory_derived_limit=256
            ),
            120,
        )
        self.assertEqual(
            simulation_conformance.outer_worker_limit(
                cpu_count=4, memory_derived_limit=128
            ),
            1,
        )


class ExecutionProcessTest(unittest.TestCase):
    def test_timeout_terminates_and_reaps_the_process_group(self) -> None:
        command = (
            sys.executable,
            "-c",
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; "
            "time.sleep(60)']); print(child.pid, flush=True); time.sleep(60)",
        )
        result = simulation_conformance.execute_process(
            command, 0.2, termination_grace_seconds=0.2
        )
        self.assertIs(
            result.disposition,
            simulation_conformance.ProcessDisposition.TIMED_OUT,
        )
        self.assertTrue(result.process_group_terminated)
        child_pid = int(result.stdout.strip())
        for _ in range(20):
            if not Path(f"/proc/{child_pid}").exists():
                break
            time.sleep(0.025)
        self.assertFalse(Path(f"/proc/{child_pid}").exists())

    def test_timeout_kills_a_descendant_that_ignores_termination(self) -> None:
        child = (
            "import os,signal,time; "
            "os.close(1); os.close(2); "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        )
        command = (
            sys.executable,
            "-c",
            "import subprocess,sys,time; "
            f"child=subprocess.Popen([sys.executable,'-c',{child!r}]); "
            "print(child.pid, flush=True); time.sleep(60)",
        )
        result = simulation_conformance.execute_process(
            command, 0.2, termination_grace_seconds=0.2
        )
        self.assertIs(
            result.disposition,
            simulation_conformance.ProcessDisposition.TIMED_OUT,
        )
        self.assertTrue(result.process_group_terminated)
        child_pid = int(result.stdout.strip())
        for _ in range(20):
            if not Path(f"/proc/{child_pid}").exists():
                break
            time.sleep(0.025)
        self.assertFalse(Path(f"/proc/{child_pid}").exists())

    def test_statistics_parser_keeps_cycle_and_tick_domains_distinct(self) -> None:
        row = (
            "execution-matrix cell=paired-system-cgra "
            "paired_work_fingerprint=0123456789abcdef0123456789abcdef"
            "0123456789abcdef0123456789abcdef deterministic_work=99 "
            "accelerator_reference_cycles=7 gem5_ticks=101 "
            "setup_wall_us=1000 preparation_wall_us=2000 "
            "gem5_configuration_wall_us=3000 provider_wall_us=250000 "
            "active_wall_us=250000 provider_cpu_us=125000 "
            "gem5_active_process_cpu_us=10000 "
            "gem5_observation_process_cpu_us=500 "
            "engine_process_cpu_us=2000 cgra_input_load_wall_us=0 "
            "cgra_input_load_process_cpu_us=0 "
            "cgra_engine_active_wall_us=200000 "
            "cgra_engine_active_process_cpu_us=8000 "
            "cgra_observation_projection_wall_us=0 "
            "cgra_observation_projection_process_cpu_us=0 "
            "cgra_artifact_publication_wall_us=0 "
            "cgra_artifact_publication_process_cpu_us=0 "
            "bridge_callback_cpu_us=1000 "
            "bridge_wait_wall_us=4000 bridge_message_count=2 "
            "accelerator_invocation_count=1 cgra_event_frame_count=99 "
            "gem5_observation_wall_us=1000 "
            "observation_wall_us=3000 peak_rss_kib=64\n"
        )
        measurement = simulation_conformance.parse_execution_matrix_measurement(
            row,
            "paired-system-cgra",
        )
        self.assertEqual(measurement.accelerator_reference_cycles, 7)
        self.assertEqual(measurement.gem5_ticks, 101)
        self.assertEqual(measurement.timing.active_wall_seconds, 0.25)
        self.assertEqual(measurement.timing.peak_resident_bytes, 64 * 1024)
        invalid_rows = (
            row.replace("gem5_ticks=101", "gem5_ticks=not_applicable"),
            row.replace("gem5_ticks=101", "gem5_ticks=+101"),
            row.replace("active_wall_us=250000", "active_wall_us=0"),
            row.replace("cgra_event_frame_count=99", "cgra_event_frame_count=0"),
            row.replace("deterministic_work=99", f"deterministic_work={1 << 64}"),
        )
        for invalid in invalid_rows:
            with self.subTest(row=invalid):
                with self.assertRaises(ValueError):
                    simulation_conformance.parse_execution_matrix_measurement(
                        invalid, "paired-system-cgra"
                    )
        spatial_with_ticks = (
            row.replace("cell=paired-system-cgra", "cell=paired-spatial-cgra")
            .replace("gem5_active_process_cpu_us=10000", "gem5_active_process_cpu_us=0")
            .replace(
                "gem5_observation_process_cpu_us=500",
                "gem5_observation_process_cpu_us=0",
            )
            .replace("bridge_message_count=2", "bridge_message_count=0")
            .replace("accelerator_invocation_count=1", "accelerator_invocation_count=0")
        )
        with self.assertRaises(ValueError):
            simulation_conformance.parse_execution_matrix_measurement(
                spatial_with_ticks, "paired-spatial-cgra"
            )

    def test_real_process_pair_produces_a_typed_positive_control(self) -> None:
        scratch_root = ROOT / "temp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            runner = Path(directory) / "matrix-runner"
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "import os,sys\n"
                "requested = sys.argv[1]\n"
                "batch = requested == 'paired-spatial-cgra-batch'\n"
                "cell = 'paired-spatial-cgra' if batch else requested\n"
                "system = cell == 'paired-system-cgra'\n"
                "count = int(sys.argv[3]) if batch else 1\n"
                "for _ in range(count):\n"
                " print(f'execution-matrix cell={cell} '"
                "f'paired_work_fingerprint={'0123456789abcdef' * 4} '"
                "f'deterministic_work=50000 '"
                "f'accelerator_reference_cycles=50000 '"
                "f'gem5_ticks={101 if system else \"not_applicable\"} '"
                "f'setup_wall_us=1000 preparation_wall_us=2000 '"
                "f'gem5_configuration_wall_us=3000 '"
                "f'provider_wall_us={200000 if system else 100000} '"
                "f'active_wall_us={200000 if system else 100000} '"
                "f'provider_cpu_us=1000 '"
                "f'gem5_active_process_cpu_us={500 if system else 0} '"
                "f'gem5_observation_process_cpu_us={50 if system else 0} '"
                "f'engine_process_cpu_us={400 if system else 0} '"
                "f'cgra_input_load_wall_us={0 if system else 1000} '"
                "f'cgra_input_load_process_cpu_us={0 if system else 10} '"
                "f'cgra_engine_active_wall_us=90000 '"
                "f'cgra_engine_active_process_cpu_us=300 '"
                "f'cgra_observation_projection_wall_us={0 if system else 9000} '"
                "f'cgra_observation_projection_process_cpu_us={0 if system else 20} '"
                "f'cgra_artifact_publication_wall_us={0 if system else 10} '"
                "f'cgra_artifact_publication_process_cpu_us={0 if system else 5} '"
                "f'bridge_callback_cpu_us={100 if system else 0} '"
                "f'bridge_wait_wall_us={200 if system else 0} '"
                "f'bridge_message_count={2 if system else 0} '"
                "f'accelerator_invocation_count={1 if system else 0} '"
                "f'cgra_event_frame_count=50000 '"
                "f'gem5_observation_wall_us=1000 observation_wall_us=3000 '"
                "f'peak_rss_kib=64')\n",
                encoding="ascii",
            )
            runner.chmod(0o755)
            readiness = Path(directory) / "readiness.json"
            readiness.write_text("{}\n", encoding="ascii")
            report = simulation_conformance.run_paired_execution_matrix(
                runner,
                readiness,
                spatial_warmup_runs=1,
                spatial_measurement_runs=3,
            )
        self.assertIs(
            report.disposition,
            simulation_conformance.ConformanceDisposition.PASSED,
        )
        self.assertIsNotNone(report.paired_result)
        self.assertEqual(report.system_measurement.accelerator_reference_cycles, 50_000)
        projected = simulation_conformance._report_json(report)
        self.assertEqual(projected["system"]["setup_wall_seconds"], 0.001)
        self.assertEqual(projected["system"]["preparation_wall_seconds"], 0.002)
        self.assertEqual(projected["system"]["gem5_configuration_wall_seconds"], 0.003)
        self.assertEqual(projected["system"]["provider_wall_seconds"], 0.2)
        self.assertEqual(projected["system"]["provider_process_cpu_seconds"], 0.001)
        self.assertEqual(projected["system"]["gem5_observation_wall_seconds"], 0.001)
        self.assertEqual(projected["system"]["observation_wall_seconds"], 0.003)

    def test_real_process_pair_classifies_a_slow_system_control(self) -> None:
        scratch_root = ROOT / "temp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            runner = Path(directory) / "matrix-runner"
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "requested=sys.argv[1]\n"
                "batch=requested=='paired-spatial-cgra-batch'\n"
                "cell='paired-spatial-cgra' if batch else requested\n"
                "system=cell=='paired-system-cgra'\n"
                "count=int(sys.argv[3]) if batch else 1\n"
                "for _ in range(count):\n"
                " print(f'execution-matrix cell={cell} '"
                "f'paired_work_fingerprint={'0123456789abcdef' * 4} '"
                "f'deterministic_work=500000 '"
                "f'accelerator_reference_cycles=500000 '"
                "f'gem5_ticks={101 if system else \"not_applicable\"} '"
                "f'setup_wall_us=1 preparation_wall_us=1 '"
                "f'gem5_configuration_wall_us=0 provider_wall_us=2000000 '"
                "f'active_wall_us={2000000 if system else 100000} '"
                "f'provider_cpu_us=1 gem5_active_process_cpu_us=0 '"
                "f'gem5_observation_process_cpu_us=0 engine_process_cpu_us=0 '"
                "f'cgra_input_load_wall_us=0 '"
                "f'cgra_input_load_process_cpu_us=0 '"
                "f'cgra_engine_active_wall_us=1 '"
                "f'cgra_engine_active_process_cpu_us=0 '"
                "f'cgra_observation_projection_wall_us=0 '"
                "f'cgra_observation_projection_process_cpu_us=0 '"
                "f'cgra_artifact_publication_wall_us=0 '"
                "f'cgra_artifact_publication_process_cpu_us=0 '"
                "f'bridge_callback_cpu_us=0 bridge_wait_wall_us=0 '"
                "f'bridge_message_count={1 if system else 0} '"
                "f'accelerator_invocation_count={1 if system else 0} '"
                "f'cgra_event_frame_count=1 '"
                "f'gem5_observation_wall_us=0 observation_wall_us=0 '"
                "f'peak_rss_kib=1')\n",
                encoding="ascii",
            )
            runner.chmod(0o755)
            readiness = Path(directory) / "readiness.json"
            readiness.write_text("{}\n", encoding="ascii")
            report = simulation_conformance.run_paired_execution_matrix(
                runner, readiness
            )
        self.assertIs(
            report.disposition,
            simulation_conformance.ConformanceDisposition.HARD_RATIO_EXCEEDED,
        )

    def test_real_process_pair_rejects_different_work(self) -> None:
        scratch_root = ROOT / "temp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            runner = Path(directory) / "matrix-runner"
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "import os,sys\n"
                "requested=sys.argv[1]\n"
                "batch=requested=='paired-spatial-cgra-batch'\n"
                "cell='paired-spatial-cgra' if batch else requested\n"
                "system=cell=='paired-system-cgra'\n"
                "count=int(sys.argv[3]) if batch else 1\n"
                "mismatch=os.environ['LOOM_TEST_PAIRED_MISMATCH']\n"
                "fingerprint=('1' if system and mismatch=='identity' else '0') * 64\n"
                "frames=50001 if system and mismatch=='frames' else 50000\n"
                "for _ in range(count):\n"
                " print(f'execution-matrix cell={cell} '"
                "f'paired_work_fingerprint={fingerprint} '"
                "f'deterministic_work=50000 accelerator_reference_cycles=50000 '"
                "f'gem5_ticks={101 if system else \"not_applicable\"} '"
                "f'setup_wall_us=1 preparation_wall_us=1 '"
                "f'gem5_configuration_wall_us=0 provider_wall_us=100000 '"
                "f'active_wall_us=100000 provider_cpu_us=1 '"
                "f'gem5_active_process_cpu_us={1 if system else 0} '"
                "f'gem5_observation_process_cpu_us={1 if system else 0} '"
                "f'engine_process_cpu_us=1 cgra_input_load_wall_us=0 '"
                "f'cgra_input_load_process_cpu_us=0 '"
                "f'cgra_engine_active_wall_us=1 '"
                "f'cgra_engine_active_process_cpu_us=1 bridge_callback_cpu_us=1 '"
                "f'cgra_observation_projection_wall_us=0 '"
                "f'cgra_observation_projection_process_cpu_us=0 '"
                "f'cgra_artifact_publication_wall_us=0 '"
                "f'cgra_artifact_publication_process_cpu_us=0 '"
                "f'bridge_wait_wall_us=1 bridge_message_count={1 if system else 0} '"
                "f'accelerator_invocation_count={1 if system else 0} '"
                "f'cgra_event_frame_count={frames} '"
                "f'gem5_observation_wall_us=0 observation_wall_us=0 '"
                "f'peak_rss_kib=1')\n",
                encoding="ascii",
            )
            runner.chmod(0o755)
            readiness = Path(directory) / "readiness.json"
            readiness.write_text("{}\n", encoding="ascii")
            for mismatch in ("identity", "frames"):
                with self.subTest(mismatch=mismatch):
                    os.environ["LOOM_TEST_PAIRED_MISMATCH"] = mismatch
                    try:
                        report = simulation_conformance.run_paired_execution_matrix(
                            runner, readiness
                        )
                    finally:
                        del os.environ["LOOM_TEST_PAIRED_MISMATCH"]
                    self.assertIs(
                        report.disposition,
                        simulation_conformance.ConformanceDisposition.PAIRED_WORK_MISMATCH,
                    )


if __name__ == "__main__":
    unittest.main()
