#!/usr/bin/env python3
"""Anchor tests for paired Spatial/System simulation conformance policy."""

from __future__ import annotations

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
        self.assertEqual(spatial.timing.engine_cpu_seconds, 0.09)
        self.assertEqual(spatial.timing.host_cpu_seconds, 0.0)
        self.assertEqual(system.timing.engine_cpu_seconds, 0.0)
        self.assertEqual(system.timing.host_cpu_seconds, 0.09)

    def test_parser_rejects_noncanonical_or_wrongly_owned_rows(self) -> None:
        row = _measurement_row("paired-system-cgra")
        invalid_rows = (
            row.replace("config_fingerprint=" + "1" * 64, "config_fingerprint=ABC"),
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
