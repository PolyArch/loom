#!/usr/bin/env python3
"""Anchor tests for paired Spatial/System simulation conformance policy."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import textwrap
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
            spatial_absolute_budget_seconds=(
                simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS
            ),
        )
        ordinary = simulation_conformance.paired_system_budget(
            [0.8, 1.0, 1.2],
            spatial_absolute_budget_seconds=(
                simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS
            ),
        )
        absolute_budget = simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS
        capped = simulation_conformance.paired_system_budget(
            [absolute_budget + 1.0] * 3,
            spatial_absolute_budget_seconds=absolute_budget,
        )

        self.assertEqual(tiny.spatial_reference_seconds, 0.1)
        self.assertAlmostEqual(tiny.system_budget_seconds, 0.3)
        self.assertEqual(ordinary.spatial_reference_seconds, 1.0)
        self.assertEqual(ordinary.system_budget_seconds, 3.0)
        self.assertEqual(
            capped.system_budget_seconds,
            simulation_conformance.SYSTEM_BUDGET_MULTIPLIER * absolute_budget,
        )

    def test_paired_result_keeps_budget_rate_and_hard_ratio_distinct(self) -> None:
        budget = simulation_conformance.paired_system_budget(
            [1.0, 1.0, 1.0],
            spatial_absolute_budget_seconds=(
                simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS
            ),
        )
        within = simulation_conformance.evaluate_paired_execution(
            budget,
            simulation_conformance.ActiveExecutionTiming(
                active_wall_seconds=2.5,
                reference_cycles=500_000,
                event_count=17,
                activation_count=3,
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
        self.assertEqual(within.event_count, 17)
        self.assertEqual(within.activation_count, 3)
        self.assertEqual(within.peak_resident_bytes, 4096)

        self.assertFalse(slow.within_system_budget)
        self.assertTrue(slow.hard_ratio_failure)
        self.assertFalse(slow.meets_reference_rate_target)

    def test_invalid_measurements_fail_closed(self) -> None:
        for samples in ([], [0.0], [-1.0], [math.inf], [math.nan]):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    simulation_conformance.paired_system_budget(
                        samples,
                        simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS,
                    )

        with self.assertRaises(ValueError):
            simulation_conformance.ActiveExecutionTiming(0.0, 1)
        with self.assertRaises(ValueError):
            simulation_conformance.ActiveExecutionTiming(1.0, -1)

    def test_cgra_gate_budget_is_derived_from_the_complete_profile_suite(self) -> None:
        def reference(
            identity: int,
            schema: str = "loom.mapping",
            version: str = "6.0",
        ) -> dict[str, object]:
            return {
                "schema": schema,
                "schema_version": version,
                "artifact": f"{identity:064x}",
            }

        operator_gate_sha256, operators = (
            simulation_conformance.load_cgra_representative_operators()
        )
        generator_units = (
            "seed_attempt",
            "assignment_attempt_per_seed",
            "endpoint_expansion",
            "negotiation_iteration",
            "calibration_proposal",
            "proposal_per_level_base",
            "proposal_per_movable_decision",
            "exact_repair_region_decision",
            "exact_repair_solver_call",
        )
        work_ledger = {
            "completion_goal": "exhaust_configured_work",
            "configured_seed_attempts": 4,
            "outcome": "completed",
            "incomplete_reason": None,
            "work_units": [
                {
                    "unit": unit,
                    "planned": 4 if unit == "seed_attempt" else 8,
                    "consumed": 4 if unit == "seed_attempt" else 8,
                }
                for unit in generator_units
            ],
        }
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
                "evaluation_evidence": reference(
                    600 + ordinal, "evaluation.evidence", "1.0"
                ),
            }
            profiles.append(
                {
                    "schema": "loom.cgra_budget_profile.4",
                    "workload": operator.workload,
                    "operator_id": operator.operator_id,
                    "protocol_symbol": operator.protocol_symbol,
                    "qualification_limit_nanoseconds": 45_000_000_000,
                    "warmup_runs": 1,
                    "measurement_runs": 3,
                    "batch_peak_resident_bytes": 4096,
                    "canonical_dataflow": reference(
                        ordinal, "loom.canonical_dataflow", "3.0"
                    ),
                    "simulation_workload": reference(
                        100 + ordinal, "loom.simulation_workload", "1.1"
                    ),
                    "simulation_runtime_input": reference(
                        200 + ordinal, "loom.simulation_runtime_input", "2.0"
                    ),
                    "resolved_config": {
                        "schema": "loom.config.resolved",
                        "schema_version": "11.0",
                        "artifact": f"{300:064x}",
                    },
                    "fabric": reference(301, "loom.fabric", "6.0"),
                    "tech_mapping": reference(400 + ordinal),
                    "tech_mapping_search": {
                        "outcome": "incomplete",
                        "incomplete_reason": "candidate_semantic_limit_reached",
                        "candidates": [reference(400 + ordinal)],
                        "work_units": [
                            {"unit": unit, "planned": 1, "consumed": 1}
                            for unit in (
                                "match_row_attempt",
                                "partial_cover_expansion",
                                "candidate_evaluation",
                                "publication_slot",
                            )
                        ],
                    },
                    "initial_spatial_mapping": reference(500 + ordinal),
                    "spatial_mapping": reference(500 + ordinal),
                    "spatial_pnr": {
                        **work_ledger,
                        "candidates": [reference(500 + ordinal)],
                    },
                    "transport_repair": None,
                    "warmup_evidence": reference(
                        700 + ordinal, "evaluation.evidence", "1.0"
                    ),
                    "measurements": [dict(measurement) for _ in range(3)],
                }
            )
        first_profile = profiles[0]
        incomplete_pnr = dict(first_profile["spatial_pnr"])
        incomplete_pnr["outcome"] = "incomplete"
        incomplete_pnr["incomplete_reason"] = "candidate_proof_not_established"
        incomplete_pnr["candidates"] = []
        incomplete_work = [
            dict(entry) for entry in incomplete_pnr["work_units"]
        ]
        incomplete_work[0]["consumed"] = 3
        incomplete_pnr["work_units"] = incomplete_work
        typed_outcome = {
            "schema": "loom.cgra_budget_profile_outcome.1",
            "workload": first_profile["workload"],
            "operator_id": first_profile["operator_id"],
            "protocol_symbol": first_profile["protocol_symbol"],
            "stage": "spatial_pnr",
            "resolved_config": first_profile["resolved_config"],
            "fabric": first_profile["fabric"],
            "tech_mapping_search": first_profile["tech_mapping_search"],
            "spatial_pnr": incomplete_pnr,
        }
        self.assertEqual(
            simulation_conformance.validate_cgra_profile_outcome(typed_outcome),
            ("incomplete", "candidate_proof_not_established"),
        )
        incomplete_pnr["incomplete_reason"] = "infeasible"
        with self.assertRaises(ValueError):
            simulation_conformance.validate_cgra_profile_outcome(typed_outcome)
        incomplete_pnr["incomplete_reason"] = "candidate_proof_not_established"

        infeasible_tech = dict(first_profile["tech_mapping_search"])
        infeasible_tech["outcome"] = "proven_infeasible"
        infeasible_tech["incomplete_reason"] = None
        infeasible_tech["candidates"] = []
        typed_outcome["stage"] = "tech_mapping"
        typed_outcome["tech_mapping_search"] = infeasible_tech
        typed_outcome["spatial_pnr"] = None
        self.assertEqual(
            simulation_conformance.validate_cgra_profile_outcome(typed_outcome),
            ("proven_infeasible", None),
        )
        self.assertEqual(
            simulation_conformance.derive_cgra_spatial_budget_nanoseconds(profiles),
            500_000_000,
        )
        configuration = {
            "schema": "loom.cgra_simulation_gate.4",
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
            repaired_profile = profiles[0]
            repair_child = reference(900)
            repair_pnr = json.loads(json.dumps(repaired_profile["spatial_pnr"]))
            repair_pnr["candidates"] = [repair_child]
            repaired_profile["spatial_mapping"] = repair_child
            repaired_profile["transport_repair"] = {
                "parent_system_mapping": reference(901),
                "pre_repair_evidence": reference(
                    902, "evaluation.evidence", "1.0"
                ),
                "attempts": [
                    {
                        "parent_spatial_mapping": repaired_profile[
                            "initial_spatial_mapping"
                        ],
                        "constraint_set": reference(
                            903, "loom.mapping_constraints", "1.0"
                        ),
                        "spatial_pnr": repair_pnr,
                        "child_spatial_mapping": repair_child,
                        "accepted_for_simulation": True,
                    }
                ],
            }
            path.write_text(json.dumps(configuration), encoding="ascii")
            simulation_conformance.load_cgra_gate_configuration(path)
            repaired_profile["spatial_mapping"] = reference(904)
            path.write_text(json.dumps(configuration), encoding="ascii")
            with self.assertRaises(ValueError):
                simulation_conformance.load_cgra_gate_configuration(path)
            repaired_profile["spatial_mapping"] = repair_child
            repair_attempt = repaired_profile["transport_repair"]["attempts"][0]
            repair_attempt["constraint_set"] = reference(903)
            path.write_text(json.dumps(configuration), encoding="ascii")
            with self.assertRaises(ValueError):
                simulation_conformance.load_cgra_gate_configuration(path)
            repaired_profile["transport_repair"] = None
            repaired_profile["spatial_mapping"] = repaired_profile[
                "initial_spatial_mapping"
            ]
            canonical_dataflow = repaired_profile["canonical_dataflow"]
            repaired_profile["canonical_dataflow"] = reference(905)
            path.write_text(json.dumps(configuration), encoding="ascii")
            with self.assertRaises(ValueError):
                simulation_conformance.load_cgra_gate_configuration(path)
            repaired_profile["canonical_dataflow"] = canonical_dataflow
            profile_pnr = profiles[0]["spatial_pnr"]
            assert isinstance(profile_pnr, dict)
            profile_pnr["completion_goal"] = "first_verified_candidate"
            path.write_text(json.dumps(configuration), encoding="ascii")
            with self.assertRaises(ValueError):
                simulation_conformance.load_cgra_gate_configuration(path)
            profile_pnr["completion_goal"] = "exhaust_configured_work"
            work_units = profile_pnr["work_units"]
            assert isinstance(work_units, list)
            seed_work = work_units[0]
            assert isinstance(seed_work, dict)
            seed_work["consumed"] = 3
            path.write_text(json.dumps(configuration), encoding="ascii")
            with self.assertRaises(ValueError):
                simulation_conformance.load_cgra_gate_configuration(path)
            seed_work["consumed"] = 4
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


def _measurement_row(
    cell: str,
    *,
    work_fingerprint: str = "0" * 64,
    config_fingerprint: str = "1" * 64,
    cycles: int = 50_000,
    frames: int = 49_999,
    active_wall_ns: int | None = None,
) -> str:
    system = cell == "paired-system-cgra"
    wall = (
        active_wall_ns
        if active_wall_ns is not None
        else (200_000_000 if system else 100_000_000)
    )
    return (
        "paired-simulation "
        "schema=loom.paired_simulation_measurement.2 "
        f"cell={cell} "
        f"attempt={'diagnostic' if system else 'ordinary'} "
        f"invocation={cell} work_fingerprint={work_fingerprint} "
        f"config_fingerprint={config_fingerprint} "
        f"accelerator_reference_cycles={cycles} cgra_event_frames={frames} "
        f"active_wall_ns={wall} active_cpu_ns=90000000 "
        f"gem5_ticks={123 if system else 'not_applicable'} "
        "setup_wall_ns=5000000 process_peak_rss_bytes=4096 "
        f"measurement_source={'fresh_system_diagnostic' if system else 'direct_spatial_attempt'} "
        f"rss_scope={'child_process_lifetime' if system else 'self_process_lifetime'}"
    )


class PairedMeasurementParsingTest(unittest.TestCase):
    def test_parser_preserves_reference_cycles_and_gem5_ticks(self) -> None:
        spatial = simulation_conformance.parse_execution_matrix_measurement(
            _measurement_row("paired-spatial-cgra"), "paired-spatial-cgra"
        )
        system = simulation_conformance.parse_execution_matrix_measurement(
            _measurement_row("paired-system-cgra"), "paired-system-cgra"
        )

        self.assertEqual(spatial.timing.reference_cycles, 50_000)
        self.assertIsNone(spatial.gem5_ticks)
        self.assertEqual(system.timing.reference_cycles, 50_000)
        self.assertEqual(system.gem5_ticks, 123)
        self.assertEqual(system.timing.event_count, 49_999)
        self.assertEqual(system.config_fingerprint, "1" * 64)
        self.assertEqual(spatial.attempt, "ordinary")
        self.assertEqual(system.attempt, "diagnostic")
        self.assertEqual(system.invocation, "paired-system-cgra")
        self.assertEqual(spatial.timing.engine_cpu_seconds, 0.09)
        self.assertEqual(spatial.timing.host_cpu_seconds, 0.0)
        self.assertEqual(system.timing.engine_cpu_seconds, 0.0)
        self.assertEqual(system.timing.host_cpu_seconds, 0.09)

    def test_parser_rejects_noncanonical_or_wrongly_owned_rows(self) -> None:
        row = _measurement_row("paired-system-cgra")
        invalid_rows = (
            row.replace("config_fingerprint=" + "1" * 64, "config_fingerprint=ABC"),
            row.replace("attempt=diagnostic", "attempt=ordinary"),
            row.replace("active_wall_ns=200000000", "active_wall_ns=0"),
            row.replace("gem5_ticks=123", "gem5_ticks=not_applicable"),
            row.replace("attempt=diagnostic", "attempt=ordinary"),
            row.replace(
                "invocation=paired-system-cgra",
                "invocation=diagnostic-system-cgra",
            ),
            row + " unexpected=1",
        )
        for invalid in invalid_rows:
            with self.subTest(row=invalid):
                with self.assertRaises(ValueError):
                    simulation_conformance.parse_execution_matrix_measurement(
                        invalid, "paired-system-cgra"
                    )


class ProcessGroupExecutionTest(unittest.TestCase):
    def _assert_not_live(self, process_id: int) -> None:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            stat_path = Path("/proc") / str(process_id) / "stat"
            if not stat_path.exists():
                return
            stat = stat_path.read_text(encoding="ascii")
            if stat[stat.rfind(")") + 2 :].split()[0] == "Z":
                return
            time.sleep(0.01)
        self.fail(f"process {process_id} remained live")

    def test_timeout_kills_a_descendant_that_ignores_sigterm(self) -> None:
        scratch_root = ROOT / "temp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            child_pid_path = Path(directory) / "child.pid"
            child_code = (
                "import signal,time;"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                "time.sleep(30)"
            )
            parent_code = (
                "import subprocess,sys,time;"
                f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
                f"open({str(child_pid_path)!r},'w').write(str(p.pid));"
                "time.sleep(30)"
            )
            result = simulation_conformance.execute_process(
                (sys.executable, "-c", parent_code),
                0.1,
                termination_grace_seconds=0.1,
            )
            self.assertIs(
                result.disposition,
                simulation_conformance.ProcessDisposition.TIMED_OUT,
            )
            self.assertTrue(result.process_group_terminated)
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
            self._assert_not_live(child_pid)

    def test_leaked_descendant_is_cleaned_and_reported(self) -> None:
        scratch_root = ROOT / "temp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            child_pid_path = Path(directory) / "child.pid"
            parent_code = (
                "import subprocess,sys;"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
                f"open({str(child_pid_path)!r},'w').write(str(p.pid))"
            )
            result = simulation_conformance.execute_process(
                (sys.executable, "-c", parent_code),
                1.0,
                termination_grace_seconds=0.1,
            )
            self.assertIs(
                result.disposition,
                simulation_conformance.ProcessDisposition.CLEANUP_FAILED,
            )
            self.assertTrue(result.process_group_terminated)
            self._assert_not_live(int(child_pid_path.read_text(encoding="ascii")))

    def test_nonzero_exit_remains_distinct_from_timeout(self) -> None:
        result = simulation_conformance.execute_process(
            (sys.executable, "-c", "raise SystemExit(7)"), 1.0
        )
        self.assertIs(
            result.disposition,
            simulation_conformance.ProcessDisposition.NONZERO_EXIT,
        )
        self.assertEqual(result.return_code, 7)


class PairedMeasurementRunnerTest(unittest.TestCase):
    def _write_runner(self, directory: str) -> tuple[Path, Path]:
        runner = Path(directory) / "matrix-runner"
        runner.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import os
                import sys
                import time

                requested = sys.argv[1]
                batch = requested == "paired-spatial-cgra-batch"
                cell = "paired-spatial-cgra" if batch else requested
                system = cell == "paired-system-cgra"
                if os.environ.get("LOOM_TEST_SLEEP") == ("system" if system else "spatial"):
                    time.sleep(30)
                count = int(sys.argv[3]) if batch else 1
                mismatch = os.environ.get("LOOM_TEST_MISMATCH", "")
                work = ("2" if system and mismatch == "work" else "0") * 64
                config = ("3" if system and mismatch == "config" else "1") * 64
                frames = 50001 if system and mismatch == "frames" else 49999
                for _ in range(count):
                    wall = 200000000 if system else 100000000
                    print(
                        "paired-simulation "
                        "schema=loom.paired_simulation_measurement.2 "
                        f"cell={cell} "
                        f"attempt={'diagnostic' if system else 'ordinary'} "
                        f"invocation={cell} work_fingerprint={work} "
                        f"config_fingerprint={config} "
                        "accelerator_reference_cycles=50000 "
                        f"cgra_event_frames={frames} active_wall_ns={wall} "
                        "active_cpu_ns=90000000 "
                        f"gem5_ticks={123 if system else 'not_applicable'} "
                        "setup_wall_ns=5000000 process_peak_rss_bytes=4096 "
                        f"measurement_source={'fresh_system_diagnostic' if system else 'direct_spatial_attempt'} "
                        f"rss_scope={'child_process_lifetime' if system else 'self_process_lifetime'}"
                    )
                """
            ),
            encoding="ascii",
        )
        runner.chmod(0o755)
        readiness = Path(directory) / "readiness.json"
        readiness.write_text("{}\n", encoding="ascii")
        return runner, readiness

    def test_real_process_pair_produces_a_provisional_measurement(self) -> None:
        scratch_root = ROOT / "temp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            runner, readiness = self._write_runner(directory)
            report = simulation_conformance.run_paired_execution_matrix(
                runner,
                readiness,
                spatial_warmup_runs=1,
                spatial_measurement_runs=3,
                termination_grace_seconds=0.1,
            )

        self.assertIs(
            report.disposition,
            simulation_conformance.MeasurementDisposition.MEASURED,
        )
        self.assertEqual(len(report.spatial_measurements), 3)
        self.assertIsNotNone(report.system_measurement)
        projected = simulation_conformance.report_json(report)
        self.assertEqual(projected["publication_status"], "provisional_bootstrap_only")
        self.assertEqual(projected["durable_replay_profiles"], 0)

    def test_pair_rejects_each_exact_work_or_config_mismatch(self) -> None:
        scratch_root = ROOT / "temp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            runner, readiness = self._write_runner(directory)
            for mismatch in ("work", "config", "frames"):
                with self.subTest(mismatch=mismatch):
                    os.environ["LOOM_TEST_MISMATCH"] = mismatch
                    try:
                        report = simulation_conformance.run_paired_execution_matrix(
                            runner,
                            readiness,
                            spatial_warmup_runs=1,
                            spatial_measurement_runs=2,
                            termination_grace_seconds=0.1,
                        )
                    finally:
                        del os.environ["LOOM_TEST_MISMATCH"]
                    self.assertIs(
                        report.disposition,
                        simulation_conformance.MeasurementDisposition.PAIRED_WORK_MISMATCH,
                    )
                    self.assertIsNotNone(report.system_measurement)

    def test_runner_preserves_spatial_timeout_as_incomplete(self) -> None:
        scratch_root = ROOT / "temp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            runner, readiness = self._write_runner(directory)
            os.environ["LOOM_TEST_SLEEP"] = "spatial"
            try:
                report = simulation_conformance.run_paired_execution_matrix(
                    runner,
                    readiness,
                    spatial_process_timeout_seconds=0.1,
                    system_process_timeout_seconds=1.0,
                    termination_grace_seconds=0.1,
                )
            finally:
                del os.environ["LOOM_TEST_SLEEP"]
        self.assertIs(
            report.disposition,
            simulation_conformance.MeasurementDisposition.SPATIAL_TIMED_OUT,
        )


if __name__ == "__main__":
    unittest.main()
