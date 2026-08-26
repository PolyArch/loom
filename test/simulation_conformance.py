"""Shared conformance policy for paired Spatial and System simulations."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.timeout_budgets import Tier, seconds as timeout_seconds  # noqa: E402


SPATIAL_REFERENCE_FLOOR_SECONDS = 0.1
SYSTEM_BUDGET_MULTIPLIER = 3.0
HARD_FAILURE_RATIO = 10.0
REFERENCE_RATE_TARGET_HZ = 100_000.0
DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS = float(timeout_seconds(Tier.FAST))
CGRA_SPATIAL_BOOTSTRAP_BUDGET_SECONDS = float(timeout_seconds(Tier.MEDIUM))
RESERVED_DEVELOPMENT_CPUS = 4
MAX_OUTER_WORKERS = 120


def _require_finite_positive(value: float, what: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{what} must be finite and positive")


def _require_nonnegative(value: float, what: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{what} must be finite and nonnegative")


@dataclass(frozen=True)
class ActiveExecutionTiming:
    active_wall_seconds: float
    reference_cycles: int
    engine_cpu_seconds: float = 0.0
    bridge_cpu_seconds: float = 0.0
    host_cpu_seconds: float = 0.0
    observation_cpu_seconds: float = 0.0
    event_count: int = 0
    activation_count: int = 0
    peak_resident_bytes: int = 0

    def __post_init__(self) -> None:
        _require_finite_positive(self.active_wall_seconds, "active wall time")
        if self.reference_cycles < 0:
            raise ValueError("reference-cycle count must be nonnegative")
        for value, what in (
            (self.engine_cpu_seconds, "engine CPU time"),
            (self.bridge_cpu_seconds, "Bridge CPU time"),
            (self.host_cpu_seconds, "host CPU time"),
            (self.observation_cpu_seconds, "observation CPU time"),
        ):
            _require_nonnegative(value, what)
        for value, what in (
            (self.event_count, "event count"),
            (self.activation_count, "activation count"),
            (self.peak_resident_bytes, "peak resident bytes"),
        ):
            if value < 0:
                raise ValueError(f"{what} must be nonnegative")


@dataclass(frozen=True)
class PairedSystemBudget:
    spatial_reference_seconds: float
    system_budget_seconds: float
    hard_failure_seconds: float


@dataclass(frozen=True)
class PairedExecutionResult:
    spatial_reference_seconds: float
    system_active_wall_seconds: float
    system_to_spatial_ratio: float
    system_budget_seconds: float
    within_system_budget: bool
    hard_ratio_failure: bool
    reference_cycles: int
    reference_cycles_per_second: float
    meets_reference_rate_target: bool
    engine_cpu_seconds: float
    bridge_cpu_seconds: float
    host_cpu_seconds: float
    observation_cpu_seconds: float
    event_count: int
    activation_count: int
    peak_resident_bytes: int


def paired_system_budget(
    warmed_spatial_active_seconds: Sequence[float],
    spatial_absolute_budget_seconds: float,
    *,
    reference_floor_seconds: float = SPATIAL_REFERENCE_FLOOR_SECONDS,
    system_multiplier: float = SYSTEM_BUDGET_MULTIPLIER,
) -> PairedSystemBudget:
    if not warmed_spatial_active_seconds:
        raise ValueError("at least one warmed Spatial timing sample is required")
    for sample in warmed_spatial_active_seconds:
        _require_finite_positive(sample, "warmed Spatial timing sample")
    _require_finite_positive(spatial_absolute_budget_seconds, "Spatial absolute budget")
    _require_finite_positive(reference_floor_seconds, "Spatial reference floor")
    _require_finite_positive(system_multiplier, "System budget multiplier")

    reference = max(
        float(statistics.median(warmed_spatial_active_seconds)),
        reference_floor_seconds,
    )
    system_budget = min(
        system_multiplier * reference,
        system_multiplier * spatial_absolute_budget_seconds,
    )
    return PairedSystemBudget(
        spatial_reference_seconds=reference,
        system_budget_seconds=system_budget,
        hard_failure_seconds=HARD_FAILURE_RATIO * reference,
    )


def evaluate_paired_execution(
    budget: PairedSystemBudget,
    system_timing: ActiveExecutionTiming,
) -> PairedExecutionResult:
    _require_finite_positive(budget.spatial_reference_seconds, "Spatial reference time")
    _require_finite_positive(budget.system_budget_seconds, "System budget")
    _require_finite_positive(budget.hard_failure_seconds, "hard failure time")

    ratio = system_timing.active_wall_seconds / budget.spatial_reference_seconds
    rate = system_timing.reference_cycles / system_timing.active_wall_seconds
    return PairedExecutionResult(
        spatial_reference_seconds=budget.spatial_reference_seconds,
        system_active_wall_seconds=system_timing.active_wall_seconds,
        system_to_spatial_ratio=ratio,
        system_budget_seconds=budget.system_budget_seconds,
        within_system_budget=(
            system_timing.active_wall_seconds <= budget.system_budget_seconds
        ),
        hard_ratio_failure=(ratio >= HARD_FAILURE_RATIO),
        reference_cycles=system_timing.reference_cycles,
        reference_cycles_per_second=rate,
        meets_reference_rate_target=(rate >= REFERENCE_RATE_TARGET_HZ),
        engine_cpu_seconds=system_timing.engine_cpu_seconds,
        bridge_cpu_seconds=system_timing.bridge_cpu_seconds,
        host_cpu_seconds=system_timing.host_cpu_seconds,
        observation_cpu_seconds=system_timing.observation_cpu_seconds,
        event_count=system_timing.event_count,
        activation_count=system_timing.activation_count,
        peak_resident_bytes=system_timing.peak_resident_bytes,
    )


def outer_worker_limit(
    *,
    memory_derived_limit: int,
    cpu_count: int | None = None,
    reserved_cpus: int = RESERVED_DEVELOPMENT_CPUS,
    maximum_workers: int = MAX_OUTER_WORKERS,
) -> int:
    if memory_derived_limit < 1:
        raise ValueError("memory-derived worker limit must be positive")
    if reserved_cpus < 0:
        raise ValueError("reserved CPU count must be nonnegative")
    if maximum_workers < 1:
        raise ValueError("maximum worker count must be positive")
    available_cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    if available_cpus < 1:
        raise ValueError("CPU count must be positive")
    available_workers = max(1, available_cpus - reserved_cpus)
    return min(available_workers, memory_derived_limit, maximum_workers)


class ProcessDisposition(str, Enum):
    COMPLETED = "completed"
    LAUNCH_FAILED = "launch_failed"
    NONZERO_EXIT = "nonzero_exit"
    TIMED_OUT = "timed_out"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass(frozen=True)
class ProcessExecution:
    disposition: ProcessDisposition
    command: tuple[str, ...]
    elapsed_seconds: float
    return_code: int | None
    stdout: str
    stderr: str
    process_group_terminated: bool


@dataclass(frozen=True)
class ExecutionMatrixMeasurement:
    cell: str
    attempt: str
    invocation: str
    work_fingerprint: str
    config_fingerprint: str
    gem5_ticks: int | None
    setup_wall_seconds: float
    active_process_cpu_seconds: float
    measurement_source: str
    rss_scope: str
    timing: ActiveExecutionTiming


class MeasurementDisposition(str, Enum):
    MEASURED = "measured"
    SPATIAL_EXECUTION_FAILED = "spatial_execution_failed"
    SPATIAL_TIMED_OUT = "spatial_timed_out"
    SPATIAL_BOOTSTRAP_BUDGET_EXCEEDED = "spatial_bootstrap_budget_exceeded"
    SYSTEM_EXECUTION_FAILED = "system_execution_failed"
    SYSTEM_TIMED_OUT = "system_timed_out"
    PROCESS_CLEANUP_FAILED = "process_cleanup_failed"
    PAIRED_WORK_MISMATCH = "paired_work_mismatch"
    SYSTEM_BOOTSTRAP_BUDGET_EXCEEDED = "system_bootstrap_budget_exceeded"
    HARD_RATIO_EXCEEDED = "hard_ratio_exceeded"
    REFERENCE_RATE_BELOW_TARGET = "reference_rate_below_target"


@dataclass(frozen=True)
class PairedMeasurementReport:
    disposition: MeasurementDisposition
    spatial_measurements: tuple[ExecutionMatrixMeasurement, ...]
    system_measurement: ExecutionMatrixMeasurement | None
    paired_result: PairedExecutionResult | None
    spatial_process: ProcessExecution | None
    system_process: ProcessExecution | None
    diagnostic: str = ""


def _live_process_group_members(process_group: int) -> tuple[int, ...]:
    proc_root = Path("/proc")
    if proc_root.is_dir():
        members: list[int] = []
        for entry in proc_root.iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="ascii")
                suffix = stat[stat.rfind(")") + 2 :].split()
                state = suffix[0]
                member_group = int(suffix[2])
            except (OSError, UnicodeDecodeError, ValueError, IndexError):
                continue
            if member_group == process_group and state != "Z":
                members.append(int(entry.name))
        return tuple(sorted(members))
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return ()
    except PermissionError:
        pass
    return (process_group,)


def _wait_for_process_group_exit(process_group: int, grace_seconds: float) -> bool:
    deadline = time.monotonic() + grace_seconds
    while _live_process_group_members(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(0.01, remaining))
    return True


def _terminate_process_group(process_group: int, grace_seconds: float) -> bool:
    if not _live_process_group_members(process_group):
        return True
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    if _wait_for_process_group_exit(process_group, grace_seconds):
        return True
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return _wait_for_process_group_exit(process_group, grace_seconds)


def _captured_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def execute_process(
    command: Sequence[str | os.PathLike[str]],
    timeout_budget_seconds: float,
    *,
    termination_grace_seconds: float | None = None,
    environment: Mapping[str, str] | None = None,
) -> ProcessExecution:
    _require_finite_positive(timeout_budget_seconds, "process timeout budget")
    if termination_grace_seconds is None:
        termination_grace_seconds = float(timeout_seconds(Tier.ULTRAFAST))
    _require_finite_positive(termination_grace_seconds, "termination grace")
    normalized = tuple(os.fspath(argument) for argument in command)
    if not normalized:
        raise ValueError("process command must not be empty")
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            normalized,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=None if environment is None else dict(environment),
        )
    except OSError as error:
        return ProcessExecution(
            ProcessDisposition.LAUNCH_FAILED,
            normalized,
            time.monotonic() - started,
            None,
            "",
            str(error),
            False,
        )

    try:
        stdout, stderr = process.communicate(timeout=timeout_budget_seconds)
    except subprocess.TimeoutExpired as timeout:
        terminated = _terminate_process_group(process.pid, termination_grace_seconds)
        try:
            stdout, stderr = process.communicate(timeout=termination_grace_seconds)
        except subprocess.TimeoutExpired as cleanup_timeout:
            stdout = _captured_text(cleanup_timeout.stdout or timeout.stdout)
            stderr = _captured_text(cleanup_timeout.stderr or timeout.stderr)
            terminated = False
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                pass
        return ProcessExecution(
            (
                ProcessDisposition.TIMED_OUT
                if terminated
                else ProcessDisposition.CLEANUP_FAILED
            ),
            normalized,
            time.monotonic() - started,
            process.returncode,
            _captured_text(stdout),
            _captured_text(stderr),
            terminated,
        )

    if _live_process_group_members(process.pid):
        terminated = _terminate_process_group(process.pid, termination_grace_seconds)
        return ProcessExecution(
            ProcessDisposition.CLEANUP_FAILED,
            normalized,
            time.monotonic() - started,
            process.returncode,
            stdout,
            stderr,
            terminated,
        )
    return ProcessExecution(
        (
            ProcessDisposition.COMPLETED
            if process.returncode == 0
            else ProcessDisposition.NONZERO_EXIT
        ),
        normalized,
        time.monotonic() - started,
        process.returncode,
        stdout,
        stderr,
        False,
    )


_MEASUREMENT_FIELDS = {
    "schema",
    "cell",
    "attempt",
    "invocation",
    "work_fingerprint",
    "config_fingerprint",
    "accelerator_reference_cycles",
    "cgra_event_frames",
    "active_wall_ns",
    "active_cpu_ns",
    "gem5_ticks",
    "setup_wall_ns",
    "process_peak_rss_bytes",
    "measurement_source",
    "rss_scope",
}
_MEASUREMENT_SCHEMA = "loom.paired_simulation_measurement.2"
_MEASUREMENT_PREFIX = "paired-simulation "


def _parse_u64(value: str) -> int:
    if (
        not value
        or any(character not in "0123456789" for character in value)
        or (len(value) != 1 and value.startswith("0"))
    ):
        raise ValueError("measurement contains a noncanonical unsigned integer")
    parsed = int(value)
    if parsed > (1 << 64) - 1:
        raise ValueError("measurement escapes the unsigned 64-bit domain")
    return parsed


def _parse_fingerprint(value: str, what: str) -> str:
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{what} fingerprint is not canonical")
    return value


def _parse_measurement_row(row: str, expected_cell: str) -> ExecutionMatrixMeasurement:
    fields: dict[str, str] = {}
    for item in row.split()[1:]:
        key, separator, value = item.partition("=")
        if not separator or not key or not value or key in fields:
            raise ValueError("paired measurement row is malformed")
        fields[key] = value
    if set(fields) != _MEASUREMENT_FIELDS:
        raise ValueError("paired measurement row has the wrong shape")
    if fields["schema"] != _MEASUREMENT_SCHEMA or fields["cell"] != expected_cell:
        raise ValueError("paired measurement row has the wrong identity")
    if expected_cell not in {"paired-spatial-cgra", "paired-system-cgra"}:
        raise ValueError("measurement cell is outside the paired CGRA domain")

    expected_attempt = (
        "diagnostic" if expected_cell == "paired-system-cgra" else "ordinary"
    )
    if (
        fields["attempt"] != expected_attempt
        or fields["invocation"] != expected_cell
    ):
        raise ValueError("paired measurement has the wrong attempt provenance")

    work_fingerprint = _parse_fingerprint(fields["work_fingerprint"], "work")
    config_fingerprint = _parse_fingerprint(fields["config_fingerprint"], "config")
    cycles = _parse_u64(fields["accelerator_reference_cycles"])
    frames = _parse_u64(fields["cgra_event_frames"])
    active_wall_ns = _parse_u64(fields["active_wall_ns"])
    active_cpu_ns = _parse_u64(fields["active_cpu_ns"])
    setup_wall_ns = _parse_u64(fields["setup_wall_ns"])
    peak_rss_bytes = _parse_u64(fields["process_peak_rss_bytes"])
    if cycles == 0 or frames == 0 or active_wall_ns == 0:
        raise ValueError("paired measurement contains no completed active work")

    if expected_cell == "paired-spatial-cgra":
        expected_source = "direct_spatial_attempt"
        expected_rss_scope = "self_process_lifetime"
        if fields["gem5_ticks"] != "not_applicable":
            raise ValueError("Spatial measurement contains gem5 ticks")
        gem5_ticks = None
        engine_cpu_seconds = active_cpu_ns / 1_000_000_000.0
        host_cpu_seconds = 0.0
    else:
        expected_source = "fresh_system_diagnostic"
        expected_rss_scope = "child_process_lifetime"
        gem5_ticks = _parse_u64(fields["gem5_ticks"])
        if gem5_ticks == 0:
            raise ValueError("System measurement contains no gem5 progress")
        engine_cpu_seconds = 0.0
        host_cpu_seconds = active_cpu_ns / 1_000_000_000.0
    if (
        fields["measurement_source"] != expected_source
        or fields["rss_scope"] != expected_rss_scope
    ):
        raise ValueError("paired measurement has the wrong ownership boundary")

    return ExecutionMatrixMeasurement(
        cell=expected_cell,
        attempt=expected_attempt,
        invocation=expected_cell,
        work_fingerprint=work_fingerprint,
        config_fingerprint=config_fingerprint,
        gem5_ticks=gem5_ticks,
        setup_wall_seconds=setup_wall_ns / 1_000_000_000.0,
        active_process_cpu_seconds=active_cpu_ns / 1_000_000_000.0,
        measurement_source=expected_source,
        rss_scope=expected_rss_scope,
        timing=ActiveExecutionTiming(
            active_wall_seconds=active_wall_ns / 1_000_000_000.0,
            reference_cycles=cycles,
            engine_cpu_seconds=engine_cpu_seconds,
            host_cpu_seconds=host_cpu_seconds,
            event_count=frames,
            peak_resident_bytes=peak_rss_bytes,
        ),
    )


def parse_execution_matrix_measurements(
    output: str, expected_cell: str, expected_count: int
) -> tuple[ExecutionMatrixMeasurement, ...]:
    if expected_count < 1:
        raise ValueError("expected measurement count must be positive")
    rows = [
        line for line in output.splitlines() if line.startswith(_MEASUREMENT_PREFIX)
    ]
    if len(rows) != expected_count:
        raise ValueError("execution matrix output has the wrong measurement count")
    return tuple(_parse_measurement_row(row, expected_cell) for row in rows)


def parse_execution_matrix_measurement(
    output: str, expected_cell: str
) -> ExecutionMatrixMeasurement:
    return parse_execution_matrix_measurements(output, expected_cell, 1)[0]


def _failed_report(
    disposition: MeasurementDisposition,
    spatial_measurements: Sequence[ExecutionMatrixMeasurement],
    *,
    spatial_process: ProcessExecution | None,
    system_process: ProcessExecution | None,
    diagnostic: str,
    system_measurement: ExecutionMatrixMeasurement | None = None,
) -> PairedMeasurementReport:
    return PairedMeasurementReport(
        disposition=disposition,
        spatial_measurements=tuple(spatial_measurements),
        system_measurement=system_measurement,
        paired_result=None,
        spatial_process=spatial_process,
        system_process=system_process,
        diagnostic=diagnostic,
    )


def _process_failure_disposition(
    process: ProcessExecution, *, spatial: bool
) -> MeasurementDisposition:
    if process.disposition is ProcessDisposition.CLEANUP_FAILED:
        return MeasurementDisposition.PROCESS_CLEANUP_FAILED
    if process.disposition is ProcessDisposition.TIMED_OUT:
        return (
            MeasurementDisposition.SPATIAL_TIMED_OUT
            if spatial
            else MeasurementDisposition.SYSTEM_TIMED_OUT
        )
    return (
        MeasurementDisposition.SPATIAL_EXECUTION_FAILED
        if spatial
        else MeasurementDisposition.SYSTEM_EXECUTION_FAILED
    )


def run_paired_execution_matrix(
    execution_matrix_runner: Path,
    gem5_readiness: Path,
    *,
    spatial_warmup_runs: int = 1,
    spatial_measurement_runs: int = 3,
    spatial_process_timeout_seconds: float | None = None,
    system_process_timeout_seconds: float | None = None,
    termination_grace_seconds: float | None = None,
) -> PairedMeasurementReport:
    if spatial_warmup_runs < 1 or spatial_measurement_runs < 1:
        raise ValueError("paired measurement requires warmup and measured runs")
    spatial_timeout = (
        float(timeout_seconds(Tier.MEDIUM))
        if spatial_process_timeout_seconds is None
        else spatial_process_timeout_seconds
    )
    system_timeout = (
        float(timeout_seconds(Tier.XLONG))
        if system_process_timeout_seconds is None
        else system_process_timeout_seconds
    )
    runner = str(execution_matrix_runner.resolve(strict=True))
    readiness = str(gem5_readiness.resolve(strict=True))
    scratch_root = ROOT / "temp"
    scratch_root.mkdir(exist_ok=True)
    process_environment = dict(os.environ)
    process_environment["TMPDIR"] = str(scratch_root)

    spatial_process = execute_process(
        (
            runner,
            "paired-spatial-cgra-batch",
            str(spatial_warmup_runs),
            str(spatial_measurement_runs),
        ),
        spatial_timeout,
        termination_grace_seconds=termination_grace_seconds,
        environment=process_environment,
    )
    if spatial_process.disposition is not ProcessDisposition.COMPLETED:
        return _failed_report(
            _process_failure_disposition(spatial_process, spatial=True),
            (),
            spatial_process=spatial_process,
            system_process=None,
            diagnostic=spatial_process.stderr.strip() or spatial_process.stdout.strip(),
        )
    try:
        spatial_measurements = parse_execution_matrix_measurements(
            spatial_process.stdout,
            "paired-spatial-cgra",
            spatial_measurement_runs,
        )
    except ValueError as error:
        return _failed_report(
            MeasurementDisposition.SPATIAL_EXECUTION_FAILED,
            (),
            spatial_process=spatial_process,
            system_process=None,
            diagnostic=str(error),
        )

    expected = spatial_measurements[0]
    expected_work = (
        expected.work_fingerprint,
        expected.config_fingerprint,
        expected.timing.reference_cycles,
        expected.timing.event_count,
    )
    if any(
        (
            sample.work_fingerprint,
            sample.config_fingerprint,
            sample.timing.reference_cycles,
            sample.timing.event_count,
        )
        != expected_work
        for sample in spatial_measurements[1:]
    ):
        return _failed_report(
            MeasurementDisposition.PAIRED_WORK_MISMATCH,
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=None,
            diagnostic="Spatial measurements do not name one exact work/config pair",
        )
    if any(
        sample.timing.active_wall_seconds > CGRA_SPATIAL_BOOTSTRAP_BUDGET_SECONDS
        for sample in spatial_measurements
    ):
        return _failed_report(
            MeasurementDisposition.SPATIAL_BOOTSTRAP_BUDGET_EXCEEDED,
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=None,
            diagnostic="a Spatial sample exceeded the provisional bootstrap budget",
        )

    system_process = execute_process(
        (runner, "paired-system-cgra", readiness),
        system_timeout,
        termination_grace_seconds=termination_grace_seconds,
        environment=process_environment,
    )
    if system_process.disposition is not ProcessDisposition.COMPLETED:
        return _failed_report(
            _process_failure_disposition(system_process, spatial=False),
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=system_process,
            diagnostic=system_process.stderr.strip() or system_process.stdout.strip(),
        )
    try:
        system_measurement = parse_execution_matrix_measurement(
            system_process.stdout, "paired-system-cgra"
        )
    except ValueError as error:
        return _failed_report(
            MeasurementDisposition.SYSTEM_EXECUTION_FAILED,
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=system_process,
            diagnostic=str(error),
        )

    observed_system_work = (
        system_measurement.work_fingerprint,
        system_measurement.config_fingerprint,
        system_measurement.timing.reference_cycles,
        system_measurement.timing.event_count,
    )
    if observed_system_work != expected_work:
        return _failed_report(
            MeasurementDisposition.PAIRED_WORK_MISMATCH,
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=system_process,
            diagnostic="System measurement names different work or configuration",
            system_measurement=system_measurement,
        )

    budget = paired_system_budget(
        [sample.timing.active_wall_seconds for sample in spatial_measurements],
        CGRA_SPATIAL_BOOTSTRAP_BUDGET_SECONDS,
    )
    paired = evaluate_paired_execution(budget, system_measurement.timing)
    if paired.hard_ratio_failure:
        disposition = MeasurementDisposition.HARD_RATIO_EXCEEDED
    elif not paired.within_system_budget:
        disposition = MeasurementDisposition.SYSTEM_BOOTSTRAP_BUDGET_EXCEEDED
    elif not paired.meets_reference_rate_target:
        disposition = MeasurementDisposition.REFERENCE_RATE_BELOW_TARGET
    else:
        disposition = MeasurementDisposition.MEASURED
    return PairedMeasurementReport(
        disposition=disposition,
        spatial_measurements=spatial_measurements,
        system_measurement=system_measurement,
        paired_result=paired,
        spatial_process=spatial_process,
        system_process=system_process,
    )


def _measurement_json(
    measurement: ExecutionMatrixMeasurement | None,
) -> dict[str, object] | None:
    if measurement is None:
        return None
    return {
        "cell": measurement.cell,
        "attempt": measurement.attempt,
        "invocation": measurement.invocation,
        "work_fingerprint": measurement.work_fingerprint,
        "config_fingerprint": measurement.config_fingerprint,
        "accelerator_reference_cycles": measurement.timing.reference_cycles,
        "cgra_event_frames": measurement.timing.event_count,
        "active_wall_seconds": measurement.timing.active_wall_seconds,
        "active_process_cpu_seconds": measurement.active_process_cpu_seconds,
        "gem5_ticks": measurement.gem5_ticks,
        "setup_wall_seconds": measurement.setup_wall_seconds,
        "process_peak_rss_bytes": measurement.timing.peak_resident_bytes,
        "measurement_source": measurement.measurement_source,
        "rss_scope": measurement.rss_scope,
    }


def _process_json(process: ProcessExecution | None) -> dict[str, object] | None:
    if process is None:
        return None
    return {
        "disposition": process.disposition.value,
        "elapsed_seconds": process.elapsed_seconds,
        "return_code": process.return_code,
        "process_group_terminated": process.process_group_terminated,
    }


def report_json(report: PairedMeasurementReport) -> dict[str, object]:
    projected: dict[str, object] = {
        "schema": "loom.paired_simulation_measurement_report.1",
        "publication_status": "provisional_bootstrap_only",
        "disposition": report.disposition.value,
        "bootstrap_budget_source": "config.timeout_budgets:medium",
        "spatial_timeout_source": "config.timeout_budgets:medium",
        "system_timeout_source": "config.timeout_budgets:xlong",
        "cleanup_timeout_source": "config.timeout_budgets:ultrafast",
        "durable_replay_profiles": 0,
        "spatial": [
            _measurement_json(measurement)
            for measurement in report.spatial_measurements
        ],
        "system": _measurement_json(report.system_measurement),
        "spatial_process": _process_json(report.spatial_process),
        "system_process": _process_json(report.system_process),
        "diagnostic": report.diagnostic,
        "paired": None,
    }
    if report.paired_result is not None:
        paired = report.paired_result
        projected["paired"] = {
            "spatial_reference_seconds": paired.spatial_reference_seconds,
            "system_active_wall_seconds": paired.system_active_wall_seconds,
            "system_to_spatial_ratio": paired.system_to_spatial_ratio,
            "system_budget_seconds": paired.system_budget_seconds,
            "within_system_budget": paired.within_system_budget,
            "hard_ratio_failure": paired.hard_ratio_failure,
            "reference_cycles": paired.reference_cycles,
            "reference_cycles_per_second": paired.reference_cycles_per_second,
            "meets_reference_rate_target": paired.meets_reference_rate_target,
        }
    return projected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_matrix_runner", type=Path)
    parser.add_argument("gem5_readiness", type=Path)
    parser.add_argument("--spatial-warmup-runs", type=int, default=1)
    parser.add_argument("--spatial-measurement-runs", type=int, default=3)
    arguments = parser.parse_args(argv)
    report = run_paired_execution_matrix(
        arguments.execution_matrix_runner,
        arguments.gem5_readiness,
        spatial_warmup_runs=arguments.spatial_warmup_runs,
        spatial_measurement_runs=arguments.spatial_measurement_runs,
    )
    print(json.dumps(report_json(report), indent=2, sort_keys=True))
    return 0 if report.disposition is MeasurementDisposition.MEASURED else 1


if __name__ == "__main__":
    raise SystemExit(main())
