#!/usr/bin/env python3
"""Regression test for real CMSIS DFG evidence in CGRA status rollup."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import csv
from pathlib import Path

import artifact_test_common

sys.path.insert(0, str(Path(__file__).resolve().parent))
from default_batch_test_common import default_batch_hardware  # noqa: E402
import test_cgra_sim_evidence_sweep as cgra_sweep  # noqa: E402
from test_cgra_status_summary import assert_sha256_file, one_row, read_rows  # noqa: E402


def run(
    repo: Path,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"command failed with {result.returncode}: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(
            f"command unexpectedly passed: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def assert_counts(data: dict[str, object], suite: str, expected: dict[str, int]) -> None:
    counts = data.get("counts")
    if not isinstance(counts, dict) or counts.get(suite) != expected:
        raise AssertionError(f"unexpected {suite} counts: {counts.get(suite) if isinstance(counts, dict) else counts}")


def assert_cmsis_dfg_only_counts(data: dict[str, object]) -> None:
    assert_counts(
        data,
        "cmsis-dsp",
        {
            "total": 16,
            "pass": 0,
            "fail": 0,
            "blocked": 14,
            "unsupported": 2,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "cmsis-nn",
        {
            "total": 18,
            "pass": 0,
            "fail": 0,
            "blocked": 13,
            "unsupported": 5,
            "missing_status": 0,
        },
    )


def assert_no_cmsis_pass(rows: list[dict[str, str]]) -> None:
    passed = [row for row in rows if row["suite"] in {"cmsis-dsp", "cmsis-nn"} and row["status"] == "pass"]
    if passed:
        raise AssertionError(f"CMSIS DFG-only rollup must not claim CGRA pass rows: {passed[:3]}")


def assert_cmsis_unsupported_row(
    repo: Path,
    rows: list[dict[str, str]],
    suite: str,
    case: str,
    expected_diagnostic: str,
) -> None:
    row = one_row(rows, suite, case)
    if (
        row["status"] != "unsupported"
        or row["diagnostic_class"] != "dfg_report_unsupported"
        or row["owner"] != "sim_report"
        or row["blocking_prerequisite"] != "dfg_report"
        or row["dfg_status"] != "unsupported"
        or expected_diagnostic not in row["diagnostic"]
    ):
        raise AssertionError(f"{suite}/{case} should expose exact DFG unsupported evidence: {row}")
    assert_sha256_file(row["dfg_mlir"], row["dfg_mlir_fingerprint"], repo)
    assert_sha256_file(row["dfg_report"], row["dfg_report_fingerprint"], repo)


def assert_no_sim_stage_evidence(row: dict[str, str]) -> None:
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        if row[key]:
            raise AssertionError(f"DFG-only row should not consume stale {key}: {row}")
    for key in ("dfg_status", "mapping_status", "cgra_status", "comparison_status"):
        if row[key] != "not_run":
            raise AssertionError(f"DFG-only row should leave {key}=not_run: {row}")


def route_endpoint_matches(actual: object, expected: str) -> bool:
    if not isinstance(actual, str):
        return False
    if expected.startswith("re:"):
        return re.fullmatch(expected[3:], actual) is not None
    return actual == expected


def assert_routed_endpoint_shape(
    route: dict[str, object],
    edge_ref: str,
    *,
    source_endpoint: str,
    sink_endpoint: str,
    label: str,
) -> None:
    segments = route.get("segments", [])
    if (
        not segments
        or not isinstance(segments[0], dict)
        or not isinstance(segments[-1], dict)
        or not route_endpoint_matches(segments[0].get("source_endpoint"), source_endpoint)
        or not route_endpoint_matches(segments[-1].get("sink_endpoint"), sink_endpoint)
    ):
        raise AssertionError(f"{label} route endpoints changed for {edge_ref}: {route}")
    if not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in segments):
        raise AssertionError(f"{label} route should traverse Fabric paths: {route}")


def assert_cmsis_attempt_guard_rejects_bad_relu_q7_report(repo: Path) -> None:
    sys.path.insert(0, str(repo / "test" / "e2e"))
    import run_cmsis_dfg_sim_attempts as attempts  # noqa: E402

    red1 = next(
        attempt
        for attempt in attempts.ATTEMPTS
        if attempt.artifact_stem == "arm_relu_q7.red1"
    )
    bad_report = {
        "kind": "dfg_sim_report",
        "workload": "ActivationFunctions/arm_relu_q7.c",
        "graph": "g_t_arm_relu_q7_red_1_0",
        "status": "pass",
        "optimistic_cycles": 3,
        "pipeline_latency_throughput_cycles": 2,
        "operation_mix_cycles": 1,
        "memory_address_setup_cycles": 0,
        "cycle_breakdown": [
            {
                "category": "pipeline_latency_throughput",
                "cycles": 2,
                "evidence": "bad ReLU guard fixture",
                "modeled": True,
            },
            {
                "category": "operation_mix",
                "cycles": 1,
                "evidence": "bad ReLU guard fixture",
                "modeled": True,
            },
            {
                "category": "memory_address_setup",
                "cycles": 0,
                "evidence": "bad ReLU guard fixture",
                "modeled": True,
            },
        ],
        "dynamic_work_items": 3,
        "operation_fire_counts": {
            "dataflow.load": 1,
            "arith.cmpi": 1,
            "arith.select": 1,
            "dataflow.store": 1,
        },
        "final_outputs": ["none"],
        "final_memory_state": {
            "arg5": ["i8:0", "i8:2", "i8:-3"],
        },
    }
    try:
        attempts.validate_attempt_report(
            red1,
            bad_report,
            repo / "temp" / "bad-arm_relu_q7.red1.dfg.report.json",
        )
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("CMSIS attempt guard accepted a false arm_relu_q7.red1 pass report")
    if "arm_relu_q7.red1" not in message or (
        "operation_fire_counts" not in message
        and "final_memory_state" not in message
    ):
        raise AssertionError(f"CMSIS attempt guard produced an imprecise diagnostic: {message}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def assert_manifest_projection(json_path: Path, csv_path: Path) -> None:
    data = json.loads(json_path.read_text())
    rows = read_csv_rows(csv_path)
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise AssertionError(f"LoomBench manifest JSON cases must be a list: {data}")
    if data.get("csv_projection") != str(csv_path):
        raise AssertionError(f"LoomBench manifest JSON should name CSV projection: {data}")
    if data.get("case_count") != len(cases) or len(cases) != len(rows):
        raise AssertionError(f"LoomBench manifest JSON/CSV row counts diverged: {data}, {rows}")
    by_case = {row["case"]: row for row in rows}
    for case_data in cases:
        if not isinstance(case_data, dict):
            raise AssertionError(f"LoomBench manifest case is not an object: {case_data}")
        row = by_case.get(str(case_data.get("case", "")))
        if row is None:
            raise AssertionError(f"LoomBench manifest CSV missing case: {case_data}")
        for key in (
            "case",
            "source_row",
            "software_root",
            "source_fingerprint",
            "import_state",
            "manifest_case",
            "owner",
            "reason",
        ):
            if row.get(key, "") != str(case_data.get(key, "")):
                raise AssertionError(f"LoomBench manifest CSV/JSON mismatch for {key}: {row}, {case_data}")


def assert_no_legacy_mode(repo: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_synthetic_root = out_dir / "no-legacy-loombench-root"
    stale_synthetic_root.mkdir()
    (stale_synthetic_root / "stale_case").mkdir()
    stale_manifest = out_dir / "loombench-manifest.json"
    (out_dir / "loombench-old-app-inventory.csv").write_text(
        "case,source_path,source_fingerprint\n"
        "stale_legacy_case,stale/main.cpp,0000000000000000000000000000000000000000000000000000000000000000\n"
    )
    (out_dir / "loombench-app-import-status.csv").write_text(
        "case,import_state,manifest_case,reason\n"
        "stale_legacy_case,deferred,,stale sidecar should not survive no-legacy mode\n"
    )
    (out_dir / "loombench-manifest.csv").write_text(
        "case,source_row,software_root,source_fingerprint,import_state,manifest_case,owner,reason\n"
        "stale_legacy_case,stale_legacy_case,stale,0000000000000000000000000000000000000000000000000000000000000000,deferred,,test,stale sidecar should not survive no-legacy mode\n"
    )
    stale_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "loombench_manifest",
                "csv_projection": "",
                "case_count": 1,
                "cases": [
                    {
                        "case": "stale_legacy_case",
                        "source_row": "stale_legacy_case",
                        "software_root": "stale",
                        "source_fingerprint": "0" * 64,
                        "main_source": "main.cpp",
                        "implementation_sources": [],
                        "headers": [],
                        "feature_tags": [],
                        "import_state": "deferred",
                        "manifest_case": "",
                        "oracle": "legacy_reference",
                        "input_profile": "legacy_default",
                        "tier_states": {
                            "source": "blocked",
                            "raise": "blocked",
                            "dataflow": "blocked",
                            "cgra_status": "blocked",
                        },
                        "owner": "test",
                        "reason": "stale sidecar should not be consumed without a legacy root",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--no-legacy-loombench",
        ],
    )
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    loombench_rows = [row for row in rows if row["suite"] == "loombench"]
    if loombench_rows:
        raise AssertionError(f"no-legacy rollup should not emit LoomBench rows: {loombench_rows[:3]}")
    if (out_dir / "loombench-manifest.csv").exists():
        raise AssertionError("no-legacy rollup should not emit LoomBench manifest CSV artifacts")
    stale_sidecars = [
        path
        for path in (
            out_dir / "loombench-old-app-inventory.csv",
            out_dir / "loombench-app-import-status.csv",
            stale_manifest,
            out_dir / "loombench-manifest.csv",
        )
        if path.exists()
    ]
    if stale_sidecars:
        raise AssertionError(f"no-legacy rollup should remove stale LoomBench sidecars: {stale_sidecars}")


def assert_explicit_legacy_root_must_exist(repo: Path, out_dir: Path) -> None:
    missing_root = out_dir / "does-not-exist"
    proc = subprocess.run(
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir / "rollup"),
            "--legacy-loombench-root",
            str(missing_root),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode == 0:
        raise AssertionError("explicit missing legacy LoomBench root should fail")
    if str(missing_root) not in proc.stderr:
        raise AssertionError(f"missing legacy root diagnostic should name the path: {proc.stderr}")


def assert_default_legacy_root_mode(repo: Path, out_dir: Path) -> None:
    legacy_root = out_dir / "auto-legacy-loombench-root"
    write_legacy_case(legacy_root, "legacy_missing")
    write_legacy_case(legacy_root, "vecsum")
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir / "rollup"),
        ],
        env={**os.environ, "LOOM_LEGACY_LOOMBENCH_ROOT": str(legacy_root)},
    )
    rows = read_rows(out_dir / "rollup" / "cgra-status-summary.csv")
    data = json.loads((out_dir / "rollup" / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "loombench",
        {
            "total": 2,
            "pass": 0,
            "fail": 0,
            "blocked": 2,
            "unsupported": 0,
            "missing_status": 0,
        },
    )
    legacy_missing = one_row(rows, "loombench", "legacy_missing")
    if (
        legacy_missing["status"] != "blocked"
        or legacy_missing["diagnostic_class"] != "loombench_import_deferred"
        or legacy_missing["blocking_prerequisite"] != "app_import"
    ):
        raise AssertionError(f"default legacy root should publish deferred LoomBench row: {legacy_missing}")
    vecsum = one_row(rows, "loombench", "vecsum")
    if (
        vecsum["status"] != "blocked"
        or vecsum["diagnostic_class"] != "loombench_workload_identity_bridge_ready"
        or vecsum["blocking_prerequisite"] != "sim_evidence"
        or vecsum["manifest_case"] != "vecsum"
    ):
        raise AssertionError(f"default legacy root should publish bridge-ready LoomBench row: {vecsum}")
    for artifact in (
        out_dir / "rollup" / "loombench-old-app-inventory.csv",
        out_dir / "rollup" / "loombench-app-import-status.csv",
        out_dir / "rollup" / "loombench-manifest.json",
        out_dir / "rollup" / "loombench-manifest.csv",
    ):
        if not artifact.is_file():
            raise AssertionError(f"default legacy root mode should emit {artifact}")


def assert_direct_cmsis_dfg_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    csv_output = out_dir / "cgra-status-summary.csv"
    json_output = out_dir / "cgra-status-summary.json"
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cgra_status_summary.sh",
            "--output",
            str(csv_output),
            "--json-output",
            str(json_output),
            "--legacy-loombench-root",
            str(legacy_root),
            "--cmsis-dfg-auto",
        ],
    )
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cgra_status_audit.sh",
            "--input",
            str(csv_output),
            "--json-input",
            str(json_output),
            "--legacy-loombench-root",
            str(legacy_root),
        ],
    )
    rows = read_rows(csv_output)
    data = json.loads(json_output.read_text())
    assert_cmsis_dfg_only_counts(data)
    assert_no_cmsis_pass(rows)
    for artifact in (
        out_dir / "cmsis-dsp-dfg" / "arm_add_q15.dfg.mlir",
        out_dir / "cmsis-nn-dfg" / "arm_relu_q15.dfg.mlir",
        out_dir / "cmsis-nn-dfg" / "arm_relu6_s8.dfg.mlir",
    ):
        if not artifact.is_file():
            raise AssertionError(f"direct CMSIS DFG mode should emit {artifact}")
    dsp_add = one_row(rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
    nn_relu = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
    nn_relu6 = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu6_s8.c")
    for row in (dsp_add, nn_relu, nn_relu6):
        if (
            row["status"] != "blocked"
            or row["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or row["blocking_prerequisite"] != "dfg_sim_report"
            or row["required_slice_count"] != "1"
            or not row["graph_ids"]
        ):
            raise AssertionError(f"direct CMSIS DFG mode should publish exact DFG blockers: {row}")
        assert_sha256_file(row["dfg_mlir"], row["dfg_mlir_fingerprint"], repo)


def assert_app_cgra_sweep_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--full-sim-default-batch",
        ],
    )
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "app",
        {
            "total": 122,
            "pass": 114,
            "fail": 0,
            "blocked": 0,
            "unsupported": 8,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "loombench",
        {
            "total": 10,
            "pass": 8,
            "fail": 0,
            "blocked": 1,
            "unsupported": 1,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "cmsis-dsp",
        {
            "total": 16,
            "pass": 14,
            "fail": 0,
            "blocked": 0,
            "unsupported": 2,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "cmsis-nn",
        {
            "total": 18,
            "pass": 12,
            "fail": 0,
            "blocked": 0,
            "unsupported": 6,
            "missing_status": 0,
        },
    )
    expected_hardware = default_batch_hardware(repo)
    for case, hardware in expected_hardware.items():
        assert_app_cgra_pass_row(repo, rows, case, expected_hardware=hardware)
        for suffix in ("dfg.report.json", "mapping.json", "cgra.report.json"):
            artifact = out_dir / "current-sim-cycle" / f"{case}.{suffix}"
            if not artifact.is_file():
                raise AssertionError(f"app CGRA sweep mode should emit {artifact}")
    assert_loombench_cgra_pass_row(repo, rows, "rle_decode", expected_hardware="shared_memory_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "cdma", expected_hardware="shared_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "line_intersect", expected_hardware="shared_signal_window_adg")
    assert_loombench_cgra_pass_row(repo, rows, "database_join", expected_hardware="shared_memory_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "depthwise_conv", expected_hardware="shared_memory_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "normalize", expected_hardware="shared_signal_window_adg")
    assert_loombench_cgra_pass_row(repo, rows, "spmm", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "cdma", expected_hardware="shared_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "conv2d", expected_hardware="shared_memory_reduction_adg")
    sim_evidence = out_dir / "current-sim-cycle"
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "SupportFunctions/arm_copy_f32.c", "arm_copy_f32"
    )
    assert_cmsis_reshape_memcpy_cgra_evidence(repo, rows, sim_evidence)
    assert_shared_app_blocker_rows(repo, rows, out_dir / "current-sim-cycle")

    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-case",
            "vecsum",
        ],
    )
    stale_rows = read_rows(out_dir / "cgra-status-summary.csv")
    stale_data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        stale_data,
        "app",
        {
            "total": 122,
            "pass": 1,
            "fail": 0,
            "blocked": 121,
            "unsupported": 0,
            "missing_status": 0,
        },
    )
    assert_app_cgra_pass_row(repo, stale_rows, "vecsum", expected_hardware="shared_reduction_adg")
    dotproduct = one_row(stale_rows, "app", "dotproduct")
    if dotproduct["status"] == "pass" or dotproduct["dfg_report"]:
        raise AssertionError(f"app sweep mode should not reuse stale dotproduct evidence: {dotproduct}")
    dotprod = one_row(stale_rows, "app", "dotprod")
    if dotprod["status"] == "pass" or dotprod["dfg_report"]:
        raise AssertionError(f"app sweep mode should not reuse stale dotprod evidence: {dotprod}")


def assert_loombench_cgra_pass_row(
    repo: Path,
    rows: list[dict[str, str]],
    case: str,
    *,
    expected_hardware: str,
) -> None:
    row = one_row(rows, "loombench", case)
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["hardware_system"] != expected_hardware
        or row["manifest_case"] != case
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
    ):
        raise AssertionError(f"LoomBench row should expose real CGRA-sim evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)


def assert_app_seed_batch_mode(repo: Path, out_dir: Path) -> None:
    legacy_root = out_dir / "legacy-loombench"
    write_legacy_case(legacy_root, "byte_swap")
    write_legacy_case(legacy_root, "xor_block")
    write_legacy_case(legacy_root, "vecmul")
    write_legacy_case(legacy_root, "vecscale")
    write_legacy_case(legacy_root, "downsample")
    write_legacy_case(legacy_root, "delta_encode")
    write_legacy_case(legacy_root, "delta_decode")
    write_legacy_case(legacy_root, "pack_bits")
    write_legacy_case(legacy_root, "partition")
    write_legacy_case(legacy_root, "prefix_sum_exclusive")
    write_legacy_case(legacy_root, "database_join")
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-seed-batch",
            "--jobs",
            "8",
        ],
    )
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "app",
        {
            "total": 122,
            "pass": 30,
            "fail": 0,
            "blocked": 92,
            "unsupported": 0,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "loombench",
        {
            "total": 11,
            "pass": 11,
            "fail": 0,
            "blocked": 0,
            "unsupported": 0,
            "missing_status": 0,
        },
    )
    seed_rows = (
        ("byte_swap", "shared_vector_alu_adg"),
        ("xor_block", "shared_vector_alu_adg"),
        ("vecmul", "shared_vector_alu_adg"),
        ("vecscale", "shared_vector_alu_adg"),
        ("downsample", "shared_reduction_adg"),
        ("delta_encode", "shared_reduction_adg"),
        ("delta_decode", "shared_reduction_adg"),
        ("pack_bits", "shared_reduction_adg"),
        ("partition", "shared_reduction_adg"),
        ("prefix_sum_exclusive", "shared_reduction_adg"),
        ("database_join", "shared_memory_reduction_adg"),
        ("vecsum", "shared_reduction_adg"),
        ("axpy", "shared_vector_alu_adg"),
        ("dotproduct", "shared_reduction_adg"),
        ("crc32", "shared_reduction_adg"),
        ("autocorrelation", "shared_reduction_adg"),
        ("unpack_bits", "shared_reduction_adg"),
        ("mmtile", "shared_memory_reduction_adg"),
        ("histogram", "shared_memory_reduction_adg"),
        ("histogram_strided", "shared_memory_reduction_adg"),
        ("outer", "shared_reduction_adg"),
        ("transpose", "shared_reduction_adg"),
        ("clz", "shared_memory_reduction_adg"),
        ("ctz", "shared_memory_reduction_adg"),
        ("binary_search", "shared_memory_reduction_adg"),
        ("find_first_set", "shared_memory_reduction_adg"),
        ("lower_bound", "shared_memory_reduction_adg"),
        ("upper_bound", "shared_memory_reduction_adg"),
        ("parity", "shared_memory_reduction_adg"),
        ("popcount", "shared_memory_reduction_adg"),
    )
    for case, hardware in seed_rows:
        assert_app_cgra_pass_row(repo, rows, case, expected_hardware=hardware)
    assert_loombench_cgra_pass_row(repo, rows, "byte_swap", expected_hardware="shared_vector_alu_adg")
    assert_loombench_cgra_pass_row(repo, rows, "xor_block", expected_hardware="shared_vector_alu_adg")
    assert_loombench_cgra_pass_row(repo, rows, "vecmul", expected_hardware="shared_vector_alu_adg")
    assert_loombench_cgra_pass_row(repo, rows, "vecscale", expected_hardware="shared_vector_alu_adg")
    assert_loombench_cgra_pass_row(repo, rows, "downsample", expected_hardware="shared_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "delta_encode", expected_hardware="shared_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "delta_decode", expected_hardware="shared_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "pack_bits", expected_hardware="shared_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "partition", expected_hardware="shared_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "prefix_sum_exclusive", expected_hardware="shared_reduction_adg")
    assert_loombench_cgra_pass_row(repo, rows, "database_join", expected_hardware="shared_memory_reduction_adg")
    for case, _hardware in seed_rows:
        for suffix in ("dfg.report.json", "mapping.json", "cgra.report.json", "sim-comparison-report.json"):
            artifact = out_dir / "current-sim-cycle" / f"{case}.{suffix}"
            if not artifact.is_file():
                raise AssertionError(f"app seed batch should emit {artifact}")
    assert_seed_batch_candidate_evidence(out_dir / "current-sim-cycle")


def assert_seed_batch_candidate_evidence(evidence_dir: Path) -> None:
    cgra_sweep.assert_autocorrelation_dfg_evidence(evidence_dir)
    cgra_sweep.assert_crc32_evidence(evidence_dir)
    cgra_sweep.assert_unpack_bits_evidence(evidence_dir)
    cgra_sweep.assert_mmtile_evidence(evidence_dir)
    cgra_sweep.assert_histogram_evidence(evidence_dir)
    cgra_sweep.assert_histogram_strided_evidence(evidence_dir)
    cgra_sweep.assert_outer_evidence(evidence_dir)
    cgra_sweep.assert_transpose_evidence(evidence_dir)
    cgra_sweep.assert_binary_search_evidence(evidence_dir)
    cgra_sweep.assert_popcount_evidence(evidence_dir)
    cgra_sweep.assert_pack_bits_evidence(evidence_dir)
    cgra_sweep.assert_partition_evidence(evidence_dir)
    cgra_sweep.assert_prefix_sum_exclusive_evidence(evidence_dir)
    run(
        Path(__file__).resolve().parents[2],
        ["python3", "test/artifacts/assert_database_join_cgra_evidence.py", str(evidence_dir)],
    )
    cgra_sweep.assert_mapping_uses_switch_multihop(evidence_dir, "vecmul")
    cgra_sweep.assert_mapping_uses_switch_multihop(evidence_dir, "vecscale")
    cgra_sweep.assert_mapping_uses_switch_multihop(evidence_dir, "downsample")
    cgra_sweep.assert_mapping_uses_switch_multihop(evidence_dir, "delta_decode")
    cgra_sweep.assert_mapping_edges_use_switch_multihop(
        evidence_dir,
        "delta_encode",
        {
            "arith.subi#0.result0->dataflow.store#0.operand2",
            "dataflow.load#0.result0->arith.subi#0.operand0",
            "llvm.load#0.result0->arith.subi#0.operand1",
        },
    )
    cgra_sweep.assert_delta_decode_evidence(evidence_dir)
    cgra_sweep.assert_bound_search_evidence(
        evidence_dir,
        "lower_bound",
        graph="g_t__ZN12_GLOBAL__N_121lower_bound_candidateEPKfS1_Pjjj_0_0",
        expected_output=[
            "i32:1",
            "i32:0",
            "i32:5",
            "i32:10",
            "i32:3",
            "i32:6",
            "i32:9",
            "i32:10",
        ],
        case_route_edges={
            "arith.addi#0.result0->arith.select#1.operand2",
            "arith.addi#2.result0->arith.select#0.operand1",
        },
    )
    cgra_sweep.assert_bound_search_evidence(
        evidence_dir,
        "upper_bound",
        graph="g_t__ZN12_GLOBAL__N_121upper_bound_candidateEPKfS1_Pjjj_0_0",
        expected_output=[
            "i32:3",
            "i32:0",
            "i32:5",
            "i32:10",
            "i32:4",
            "i32:7",
            "i32:10",
            "i32:10",
        ],
        case_route_edges={
            "arith.addi#0.result0->arith.select#1.operand1",
            "arith.addi#2.result0->arith.select#0.operand2",
        },
    )
    cgra_sweep.assert_bit_scan_evidence(
        evidence_dir,
        "clz",
        graph="g_t__ZN12_GLOBAL__N_113clz_candidateEPKjPjj_0_0",
        output_arg="arg7",
        event_count=1490,
        dfg_cycles=1690,
        cgra_cycles=1739,
        placed_records=9,
        routed_edges=8,
        config_records=189,
        route_segments=30,
        operation_fire_counts={
            "arith.addi": 317,
            "arith.andi": 317,
            "arith.cmpi": 380,
            "arith.shrui": 317,
            "dataflow.load": 32,
            "dataflow.store": 32,
            "dataflow.sync": 32,
            "scf.if": 63,
        },
        expected_route_edges={
            "arith.addi#0.result0->dataflow.store#0.operand2",
            "arith.andi#0.result0->arith.cmpi#2.operand0",
            "arith.shrui#0.result0->arith.andi#0.operand0",
            "dataflow.load#0.result0->arith.andi#0.operand1",
            "dataflow.load#0.result0->arith.cmpi#0.operand0",
            "dataflow.load#0.result0->arith.cmpi#1.operand0",
            "dataflow.load#0.result1->dataflow.sync#0.operand0",
            "dataflow.store#0.result0->dataflow.sync#0.operand1",
        },
    )
    cgra_sweep.assert_bit_scan_evidence(
        evidence_dir,
        "ctz",
        graph="g_t__ZN12_GLOBAL__N_113ctz_candidateEPKjPjj_0_0",
        output_arg="arg6",
        event_count=929,
        dfg_cycles=1129,
        cgra_cycles=1175,
        placed_records=10,
        routed_edges=7,
        config_records=178,
        route_segments=27,
        operation_fire_counts={
            "arith.addi": 169,
            "arith.andi": 200,
            "arith.cmpi": 232,
            "arith.shrui": 169,
            "dataflow.load": 32,
            "dataflow.store": 32,
            "dataflow.sync": 32,
            "scf.if": 63,
        },
        expected_route_edges={
            "arith.addi#0.result0->dataflow.store#0.operand2",
            "arith.andi#0.result0->arith.cmpi#1.operand0",
            "arith.andi#1.result0->arith.cmpi#2.operand0",
            "dataflow.load#0.result0->arith.andi#0.operand0",
            "dataflow.load#0.result0->arith.cmpi#0.operand0",
            "dataflow.load#0.result1->dataflow.sync#0.operand0",
            "dataflow.store#0.result0->dataflow.sync#0.operand1",
        },
    )
    cgra_sweep.assert_bit_scan_evidence(
        evidence_dir,
        "find_first_set",
        graph="g_t__ZN12_GLOBAL__N_124find_first_set_candidateEPKjPjj_0_0",
        output_arg="arg5",
        event_count=525,
        dfg_cycles=725,
        cgra_cycles=771,
        placed_records=10,
        routed_edges=7,
        config_records=178,
        route_segments=27,
        operation_fire_counts={
            "arith.addi": 68,
            "arith.andi": 99,
            "arith.cmpi": 131,
            "arith.shrui": 68,
            "dataflow.load": 32,
            "dataflow.store": 32,
            "dataflow.sync": 32,
            "scf.if": 63,
        },
        expected_route_edges={
            "arith.addi#0.result0->dataflow.store#0.operand2",
            "arith.andi#0.result0->arith.cmpi#1.operand0",
            "arith.andi#1.result0->arith.cmpi#2.operand0",
            "dataflow.load#0.result0->arith.andi#0.operand0",
            "dataflow.load#0.result0->arith.cmpi#0.operand0",
            "dataflow.load#0.result1->dataflow.sync#0.operand0",
            "dataflow.store#0.result0->dataflow.sync#0.operand1",
        },
    )
    cgra_sweep.assert_bit_scan_evidence(
        evidence_dir,
        "parity",
        graph="g_t_parity_0_0",
        output_arg="arg4",
        event_count=3648,
        dfg_cycles=3848,
        cgra_cycles=3891,
        placed_records=8,
        routed_edges=6,
        config_records=152,
        route_segments=24,
        operation_fire_counts={
            "arith.andi": 872,
            "arith.cmpi": 904,
            "arith.shrui": 872,
            "arith.xori": 872,
            "dataflow.load": 32,
            "dataflow.store": 32,
            "dataflow.sync": 32,
            "scf.if": 32,
        },
        expected_route_edges={
            "arith.andi#0.result0->arith.xori#0.operand1",
            "arith.shrui#0.result0->arith.cmpi#1.operand0",
            "arith.xori#0.result0->dataflow.store#0.operand2",
            "dataflow.load#0.result0->arith.cmpi#0.operand0",
            "dataflow.load#0.result1->dataflow.sync#0.operand0",
            "dataflow.store#0.result0->dataflow.sync#0.operand1",
        },
    )


SHARED_APP_BLOCKER_DIAGNOSTICS = {
    "edge_update": (
        "primary workload graph is partial: edge_update lowering covers the input-to-output copy loop "
        "while the CSR lookup and update loop remains outside dataflow"
    ),
    "edge_update_batch": (
        "primary workload graph is partial: edge_update_batch lowering covers the input-to-output copy loop "
        "while the batched CSR lookup and update loops remain outside dataflow"
    ),
    "col2im": (
        "primary workload graph absent: col2im_kernel remains a residual call target outside "
        "the discovered dataflow graphs; no discovered graph ids were emitted, so DFG-sim cannot "
        "observe the kernel return value"
    ),
    "sort_insertion": (
        "primary workload graph is partial: sort_insertion lowering covers the copy loop "
        "while the insertion-sort compare-and-shift loop remains outside dataflow"
    ),
    "sort_merge": (
        "primary workload graph is partial: sort_merge lowering covers copy and remainder-copy slices "
        "while the merge compare loop remains outside dataflow"
    ),
    "sort_quick": (
        "primary workload graph is partial: sort_quick lowering covers copy and partition slices "
        "while iterative stack control remains outside dataflow"
    ),
    "spmspm": (
        "primary workload graph is partial: spmspm lowering covers final nonzero compression "
        "while sparse multiply-accumulate loops remain outside dataflow"
    ),
    "string_compare": (
        "primary workload graph absent: string_compare_kernel remains a residual call target outside "
        "the discovered dataflow graphs; discovered graph ids include g_t_main_0_0,g_t_main_1_0,"
        "g_t_main_2_0, so DFG-sim cannot observe the kernel return value"
    ),
}
EMPTY_DISCOVERED_GRAPH_IDS = "__empty__"

SHARED_APP_MAPPING_FAILURE_DIAGNOSTICS: dict[str, str] = {}

SHARED_APP_MAPPING_FAILURE_EVIDENCE: dict[str, dict[str, object]] = {}

SHARED_APP_MAPPING_BLOCKED_DIAGNOSTICS: dict[str, str] = {}

SHARED_APP_MAPPING_BLOCKED_EVIDENCE: dict[str, dict[str, object]] = {}

SHARED_APP_MAPPING_UNSUPPORTED_DIAGNOSTICS: dict[str, str] = {}

SHARED_APP_MAPPING_UNSUPPORTED_EVIDENCE: dict[str, dict[str, object]] = {}


def assert_shared_app_blocker_rows(repo: Path, rows: list[dict[str, str]], sim_evidence: Path) -> None:
    for case, diagnostic in SHARED_APP_BLOCKER_DIAGNOSTICS.items():
        row = one_row(rows, "app", case)
        if (
            row["status"] != "unsupported"
            or row["diagnostic_class"] != "dfg_report_unsupported"
            or row["owner"] != "sim_report"
            or row["blocking_prerequisite"] != "dfg_report"
            or row["dfg_status"] != "unsupported"
            or row["mapping_status"] != "unsupported"
            or row["cgra_status"] != "blocked"
            or row["comparison_status"] != "blocked"
            or row["hardware_system"] != "shared_reduction_adg"
            or row["final_outputs_present"] != "false"
            or row["final_memory_state_present"] != "false"
            or diagnostic not in row["diagnostic"]
        ):
            raise AssertionError(f"attempted app row should expose structured shared-ADG blocker: {row}")
        for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
            assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
            artifact = sim_evidence / Path(row[key]).name
            if not artifact.is_file():
                raise AssertionError(f"attempt manifest should emit {artifact}")
    for case, diagnostic in SHARED_APP_MAPPING_FAILURE_DIAGNOSTICS.items():
        row = one_row(rows, "app", case)
        if (
            row["status"] != "fail"
            or row["diagnostic_class"] != "mapping_artifact_failed"
            or row["owner"] != "sim_report"
            or row["blocking_prerequisite"] != "mapping_artifact"
            or row["dfg_status"] != "pass"
            or row["mapping_status"] != "fail"
            or row["cgra_status"] != "blocked"
            or row["comparison_status"] != "blocked"
            or row["hardware_system"] != "shared_reduction_adg"
            or row["graph_ids"] != SHARED_APP_MAPPING_FAILURE_EVIDENCE[case]["graph"]
            or row["final_outputs_present"] != "true"
            or row["final_memory_state_present"] != "true"
            or diagnostic not in row["diagnostic"]
        ):
            raise AssertionError(f"attempted app row should expose structured mapping failure: {row}")
        for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
            assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
            artifact = sim_evidence / Path(row[key]).name
            if not artifact.is_file():
                raise AssertionError(f"attempt manifest should emit {artifact}")
        dfg_report = json.loads((repo / row["dfg_report"]).read_text())
        expected = SHARED_APP_MAPPING_FAILURE_EVIDENCE[case]
        if (
            dfg_report.get("status") != "pass"
            or dfg_report.get("graph") != expected["graph"]
            or dfg_report.get("dynamic_work_items") != expected["dynamic_work_items"]
        ):
            raise AssertionError(f"{case} should preserve real DFG evidence before mapping failure: {dfg_report}")
        if "final_output_suffix" in expected:
            expected_suffix = expected["final_output_suffix"]
            if dfg_report.get("final_outputs", [])[-len(expected_suffix) :] != expected_suffix:
                raise AssertionError(
                    f"{case} should preserve expected DFG final outputs before mapping failure: {dfg_report}"
                )
        if "operation_fire_counts" in expected:
            for op_name, expected_count in expected["operation_fire_counts"].items():
                actual_count = dfg_report.get("operation_fire_counts", {}).get(op_name)
                if actual_count != expected_count:
                    raise AssertionError(
                        f"{case} {op_name} fire count should be {expected_count}, got {actual_count}: {dfg_report}"
                    )
        if "final_memory_state" in expected:
            for argument, expected_values in expected["final_memory_state"].items():
                actual_values = dfg_report.get("final_memory_state", {}).get(argument)
                if actual_values != expected_values:
                    raise AssertionError(
                        f"{case} should preserve expected DFG final memory before mapping failure: {dfg_report}"
                    )
    for case, diagnostic in SHARED_APP_MAPPING_BLOCKED_DIAGNOSTICS.items():
        row = one_row(rows, "app", case)
        if (
            row["status"] != "blocked"
            or row["diagnostic_class"] != "mapping_artifact_blocked"
            or row["owner"] != "sim_report"
            or row["blocking_prerequisite"] != "mapping_artifact"
            or row["dfg_status"] != "pass"
            or row["mapping_status"] != "blocked"
            or row["cgra_status"] != "blocked"
            or row["comparison_status"] != "blocked"
            or row["hardware_system"] != "shared_reduction_adg"
            or row["graph_ids"] != SHARED_APP_MAPPING_BLOCKED_EVIDENCE[case]["graph"]
            or row["final_outputs_present"] != "true"
            or row["final_memory_state_present"] != "true"
            or diagnostic not in row["diagnostic"]
        ):
            raise AssertionError(f"attempted app row should expose structured mapping blocker: {row}")
        for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
            assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
            artifact = sim_evidence / Path(row[key]).name
            if not artifact.is_file():
                raise AssertionError(f"attempt manifest should emit {artifact}")
        dfg_report = json.loads((repo / row["dfg_report"]).read_text())
        expected = SHARED_APP_MAPPING_BLOCKED_EVIDENCE[case]
        if (
            dfg_report.get("status") != "pass"
            or dfg_report.get("dynamic_work_items") != expected["dynamic_work_items"]
        ):
            raise AssertionError(f"{case} should preserve real DFG evidence before mapping blocker: {dfg_report}")
        if dfg_report.get("final_outputs", []) != expected["final_output_suffix"]:
            raise AssertionError(f"{case} should preserve expected DFG final outputs before mapping blocker: {dfg_report}")
        for op_name, expected_count in expected["operation_fire_counts"].items():
            actual_count = dfg_report.get("operation_fire_counts", {}).get(op_name)
            if actual_count != expected_count:
                raise AssertionError(
                    f"{case} {op_name} fire count should be {expected_count}, got {actual_count}: {dfg_report}"
                )
        for argument, expected_values in expected["final_memory_state"].items():
            actual_values = dfg_report.get("final_memory_state", {}).get(argument)
            if actual_values != expected_values:
                raise AssertionError(f"{case} should preserve expected DFG final memory before mapping blocker: {dfg_report}")
    for case, diagnostic in SHARED_APP_MAPPING_UNSUPPORTED_DIAGNOSTICS.items():
        row = one_row(rows, "app", case)
        if (
            row["status"] != "blocked"
            or row["diagnostic_class"] != "mapping_artifact_unsupported"
            or row["owner"] != "sim_report"
            or row["blocking_prerequisite"] != "mapping_artifact"
            or row["dfg_status"] != "pass"
            or row["mapping_status"] != "unsupported"
            or row["cgra_status"] != "blocked"
            or row["comparison_status"] != "blocked"
            or row["hardware_system"] != "shared_reduction_adg"
            or row["graph_ids"] != SHARED_APP_MAPPING_UNSUPPORTED_EVIDENCE[case]["graph"]
            or row["final_outputs_present"] != "true"
            or row["final_memory_state_present"] != "true"
            or diagnostic not in row["diagnostic"]
        ):
            raise AssertionError(f"attempted app row should expose structured mapping unsupported blocker: {row}")
        for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
            assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
            artifact = sim_evidence / Path(row[key]).name
            if not artifact.is_file():
                raise AssertionError(f"attempt manifest should emit {artifact}")
        dfg_report = json.loads((repo / row["dfg_report"]).read_text())
        expected = SHARED_APP_MAPPING_UNSUPPORTED_EVIDENCE[case]
        if (
            dfg_report.get("status") != "pass"
            or dfg_report.get("dynamic_work_items") != expected["dynamic_work_items"]
            or dfg_report.get("final_outputs") != expected["final_outputs"]
        ):
            raise AssertionError(f"{case} should preserve real DFG evidence before mapping unsupported: {dfg_report}")
        for op_name, expected_count in expected["operation_fire_counts"].items():
            actual_count = dfg_report.get("operation_fire_counts", {}).get(op_name)
            if actual_count != expected_count:
                raise AssertionError(
                    f"{case} {op_name} fire count should be {expected_count}, got {actual_count}: {dfg_report}"
                )
        if "final_memory_state" in expected:
            for argument, expected_values in expected["final_memory_state"].items():
                actual_values = dfg_report.get("final_memory_state", {}).get(argument)
                if actual_values != expected_values:
                    raise AssertionError(
                        f"{case} should preserve expected DFG final memory before mapping unsupported: {dfg_report}"
                    )


def assert_app_attempt_manifest_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-attempt-manifest",
            "test/app/shared-cgra-blocker-batch.json",
        ],
    )
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "app",
        {
            "total": 122,
            "pass": 18,
            "fail": 0,
            "blocked": 96,
            "unsupported": 8,
            "missing_status": 0,
        },
    )
    assert_app_cgra_pass_row(repo, rows, "crc32", expected_hardware="shared_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "autocorrelation", expected_hardware="shared_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "unpack_bits", expected_hardware="shared_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "mmtile", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "outer", expected_hardware="shared_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "transpose", expected_hardware="shared_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "clz", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "ctz", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "binary_search", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "bitonic_stage-modified", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "histogram", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "hist_bin", expected_hardware="shared_signal_window_adg")
    assert_app_cgra_pass_row(repo, rows, "histogram_strided", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "find_first_set", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "lower_bound", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "upper_bound", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "parity", expected_hardware="shared_memory_reduction_adg")
    assert_app_cgra_pass_row(repo, rows, "popcount", expected_hardware="shared_memory_reduction_adg")
    assert_shared_app_blocker_rows(repo, rows, out_dir / "current-sim-cycle")


def assert_sort_insertion_attempt_manifest_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    out_dir.mkdir(parents=True)
    attempt_manifest = out_dir / "sort-insertion-attempt.json"
    attempt_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"case": "sort_insertion", "hardware": "shared_reduction_adg"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir / "rollup"),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-attempt-manifest",
            str(attempt_manifest),
        ],
    )
    rows = read_rows(out_dir / "rollup" / "cgra-status-summary.csv")
    row = one_row(rows, "app", "sort_insertion")
    if (
        row["status"] != "unsupported"
        or row["diagnostic_class"] != "dfg_report_unsupported"
        or row["owner"] != "sim_report"
        or row["blocking_prerequisite"] != "dfg_report"
        or row["dfg_status"] != "unsupported"
        or row["mapping_status"] != "unsupported"
        or row["cgra_status"] != "blocked"
        or row["comparison_status"] != "blocked"
        or row["hardware_system"] != "shared_reduction_adg"
        or row["final_outputs_present"] != "false"
        or row["final_memory_state_present"] != "false"
        or SHARED_APP_BLOCKER_DIAGNOSTICS["sort_insertion"] not in row["diagnostic"]
    ):
        raise AssertionError(f"sort_insertion attempt should publish structured lowering-boundary evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)


def assert_primary_graph_absence_attempt_mode(
    repo: Path,
    out_dir: Path,
    legacy_root: Path,
    *,
    case: str,
    expected_primary_graph_token: str,
    expected_discovered_graph: str = "",
    expected_residual_call: str = "",
) -> None:
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-case",
            case,
        ],
    )
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    row = one_row(rows, "app", case)
    expected_diagnostic = SHARED_APP_BLOCKER_DIAGNOSTICS.get(
        case,
        f"primary workload graph absent: expected token {expected_primary_graph_token}",
    )
    if (
        row["status"] != "unsupported"
        or row["diagnostic_class"] != "dfg_report_unsupported"
        or row["owner"] != "sim_report"
        or row["blocking_prerequisite"] != "dfg_report"
        or row["dfg_status"] != "unsupported"
        or row["mapping_status"] != "unsupported"
        or row["cgra_status"] != "blocked"
        or row["comparison_status"] != "blocked"
        or row["hardware_system"] != "shared_reduction_adg"
        or row["final_outputs_present"] != "false"
        or row["final_memory_state_present"] != "false"
        or expected_diagnostic not in row["diagnostic"]
    ):
        raise AssertionError(f"primary graph absence attempt should publish structured evidence: {row}")
    stale_diagnostic = f"primary workload graph absent: expected token {expected_primary_graph_token}"
    if stale_diagnostic != expected_diagnostic and stale_diagnostic in row["diagnostic"]:
        raise AssertionError(f"primary graph absence attempt should not keep stale diagnostic: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
        artifact = out_dir / "current-sim-cycle" / Path(row[key]).name
        if key == "comparison_report":
            artifact = out_dir / "cgra-status-comparisons" / Path(row[key]).name
        if not artifact.is_file():
            raise AssertionError(f"primary graph absence attempt should emit {artifact}")
    dfg_report = out_dir / "current-sim-cycle" / Path(row["dfg_report"]).name
    dfg_data = json.loads(dfg_report.read_text())
    diagnostics = dfg_data.get("diagnostics")
    if not isinstance(diagnostics, list) or expected_diagnostic not in diagnostics:
        raise AssertionError(f"primary graph absence attempt should emit exact DFG diagnostic: {dfg_data}")
    if stale_diagnostic != expected_diagnostic and stale_diagnostic in diagnostics:
        raise AssertionError(f"primary graph absence DFG report should not keep stale diagnostic: {dfg_data}")
    graph_ids = dfg_data.get("discovered_graph_ids")
    if not isinstance(graph_ids, list) or any(
        expected_primary_graph_token in str(graph_id) for graph_id in graph_ids
    ):
        raise AssertionError(f"primary graph absence attempt should not expose primary graph: {dfg_data}")
    if expected_discovered_graph == EMPTY_DISCOVERED_GRAPH_IDS and graph_ids:
        raise AssertionError(f"primary graph absence attempt should not expose any graph ids: {dfg_data}")
    if (
        expected_discovered_graph
        and expected_discovered_graph != EMPTY_DISCOVERED_GRAPH_IDS
        and expected_discovered_graph not in graph_ids
    ):
        raise AssertionError(
            f"primary graph absence attempt should prove graph {expected_discovered_graph}: {dfg_data}"
        )
    residual_calls = dfg_data.get("residual_call_targets")
    if expected_residual_call and (
        not isinstance(residual_calls, list) or expected_residual_call not in residual_calls
    ):
        raise AssertionError(
            f"primary graph absence attempt should prove residual call {expected_residual_call}: {dfg_data}"
        )


def assert_primary_graph_absence_empty_graph_guard(repo: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dfg_mlir = out_dir / "helper-with-residual-call.mlir"
    dfg_mlir.write_text(
        "\n".join(
            [
                "module {",
                "  dataflow.graph.func @g_t_helper_0_0() {",
                "    dataflow.return",
                "  }",
                "  func.func @main() {",
                "    call @col2im_kernel() : () -> ()",
                "    return",
                "  }",
                "}",
                "",
            ]
        )
    )
    proc = run(
        repo,
        [
            "python3",
            "test/e2e/emit_primary_graph_absence_artifacts.py",
            "--workload",
            "col2im",
            "--dfg-mlir",
            str(dfg_mlir),
            "--expected-graph-token",
            "col2im_kernel",
            "--require-empty-discovered-graphs",
            "--required-residual-call",
            "col2im_kernel",
            "--hardware",
            "shared_reduction_adg",
            "--dfg-output",
            str(out_dir / "col2im.dfg.report.json"),
            "--dfg-cycle-output",
            str(out_dir / "col2im.dfg-cycle.csv"),
            "--mapping-output",
            str(out_dir / "col2im.mapping.json"),
            "--mapping-summary-output",
            str(out_dir / "col2im.mapping.csv"),
        ],
        expect_success=False,
    )
    combined = proc.stdout + proc.stderr
    if "discovered graph ids should be empty" not in combined:
        raise AssertionError(f"empty graph guard should explain discovered graph ids: {combined}")


def assert_no_dfg_app_direct_attempt_mode(
    repo: Path,
    out_dir: Path,
    legacy_root: Path,
    *,
    case: str,
    expected_primary_graph_token: str,
) -> None:
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-case",
            case,
        ],
    )
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    row = one_row(rows, "app", case)
    artifact_keys = ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report")
    if any(row[key] for key in artifact_keys):
        expected_diagnostic = f"primary workload graph absent: expected token {expected_primary_graph_token}"
        if (
            row["status"] != "blocked"
            or row["diagnostic_class"] != "dfg_report_unsupported"
            or row["owner"] != "sim_report"
            or row["blocking_prerequisite"] != "dfg_report"
            or row["dfg_status"] != "unsupported"
            or row["mapping_status"] != "unsupported"
            or row["cgra_status"] != "blocked"
            or row["comparison_status"] != "blocked"
            or row["hardware_system"] != "shared_reduction_adg"
            or row["final_outputs_present"] != "false"
            or row["final_memory_state_present"] != "false"
            or expected_diagnostic not in row["diagnostic"]
        ):
            raise AssertionError(f"no-DFG app attempt should publish structured probe blocker: {row}")
        for key in artifact_keys:
            assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
            artifact = out_dir / "current-sim-cycle" / Path(row[key]).name
            if key == "comparison_report":
                artifact = out_dir / "cgra-status-comparisons" / Path(row[key]).name
            if not artifact.is_file():
                raise AssertionError(f"no-DFG app attempt should emit {artifact}")
        return

    if (
        row["status"] != "blocked"
        or row["diagnostic_class"] != "app_dataflow_tier_missing"
        or row["owner"] != "compiler_pipeline"
        or row["blocking_prerequisite"] != "dataflow"
        or row["required_slice_count"] != "0"
        or row["graph_ids"] != ""
        or row["dfg_status"] != "not_run"
        or row["mapping_status"] != "not_run"
        or row["cgra_status"] != "not_run"
        or row["comparison_status"] != "not_run"
        or row["hardware_system"] != ""
        or row["final_outputs_present"] != "false"
        or row["final_memory_state_present"] != "false"
        or f"app manifest has no dfg tier for {case}" not in row["diagnostic"]
    ):
        raise AssertionError(f"no-DFG app attempt should publish structured lowering-boundary evidence: {row}")
    for key in artifact_keys:
        if row[key] or row[f"{key}_fingerprint"]:
            raise AssertionError(f"no-DFG app attempt must not carry {key} evidence: {row}")


def assert_cmsis_sim_default_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--cmsis-sim-default",
        ],
    )
    sim_evidence = out_dir / "current-sim-cycle"
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "cmsis-dsp",
        {
            "total": 16,
            "pass": 14,
            "fail": 0,
            "blocked": 0,
            "unsupported": 2,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "cmsis-nn",
        {
            "total": 18,
            "pass": 12,
            "fail": 0,
            "blocked": 0,
            "unsupported": 6,
            "missing_status": 0,
        },
    )
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c", "arm_add_q15")
    assert_cmsis_cgra_pass_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-dsp",
        "FilteringFunctions/arm_biquad_cascade_df1_f32.c",
        "arm_biquad_cascade_df1_f32",
    )
    assert_cmsis_biquad_shared_adg_evidence(sim_evidence)
    assert_cmsis_mat_mult_f32_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c", "arm_relu_q15"
    )
    assert_cmsis_relu_q7_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_relu6_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_concat_w_memcpy_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_concat_memcpy_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_reshape_memcpy_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_vector_sum_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_minimum_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_maximum_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_softmax_u8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_depthwise_conv_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_max_pool_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_dfg_unsupported_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-nn",
        "FullyConnectedFunctions/arm_fully_connected_s8.c",
        "arm_fully_connected_s8",
        "g_t_arm_fully_connected_s8_red_0_0",
        "unsupported op: llvm.call @arm_nn_vec_mat_mult_t_s8",
        expected_callee="@arm_nn_vec_mat_mult_t_s8",
    )
    assert_cmsis_unsupported_row(
        repo,
        rows,
        "cmsis-nn",
        "FullyConnectedFunctions/arm_fully_connected_s8.c",
        "unsupported op: llvm.call @arm_nn_vec_mat_mult_t_s8",
    )
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
        ],
    )
    dfg_only_rows = read_rows(out_dir / "cgra-status-summary.csv")
    dfg_only_data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_cmsis_dfg_only_counts(dfg_only_data)
    assert_no_cmsis_pass(dfg_only_rows)
    assert_no_sim_stage_evidence(one_row(dfg_only_rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c"))
    assert_no_sim_stage_evidence(one_row(dfg_only_rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c"))
    assert_no_sim_stage_evidence(one_row(dfg_only_rows, "cmsis-nn", "ActivationFunctions/arm_relu6_s8.c"))


def assert_cmsis_dfg_report(
    sim_evidence: Path,
    stem: str,
    workload: str,
    fired_op: str,
    final_memory_state: dict[str, list[str]],
) -> None:
    report = sim_evidence / f"{stem}.dfg.report.json"
    if not report.is_file():
        raise AssertionError(f"CMSIS DFG-sim evidence mode should emit {report}")
    report_data = json.loads(report.read_text())
    if (
        report_data.get("kind") != "dfg_sim_report"
        or report_data.get("workload") != workload
        or report_data.get("status") != "pass"
        or report_data.get("dynamic_work_items") != 4
        or report_data.get("operation_fire_counts", {}).get(fired_op) != 4
    ):
        raise AssertionError(f"unexpected CMSIS DFG-sim report: {report_data}")
    for key, expected in final_memory_state.items():
        if report_data.get("final_memory_state", {}).get(key) != expected:
            raise AssertionError(f"unexpected CMSIS DFG-sim memory for {stem}: {report_data}")
    input_fingerprints = report_data.get("input_artifact_fingerprints")
    if not isinstance(input_fingerprints, dict) or f"{stem}.dfg" not in input_fingerprints:
        raise AssertionError(f"CMSIS DFG-sim report should fingerprint its input DFG MLIR: {report_data}")


def assert_cmsis_cgra_pass_row(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
    suite: str,
    case: str,
    stem: str,
    expected_hardware: str = "shared_reduction_adg",
) -> None:
    row = one_row(rows, suite, case)
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["blocking_prerequisite"] != ""
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["hardware_system"] != expected_hardware
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
    ):
        raise AssertionError(f"CMSIS row should expose real CGRA-sim evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
    for artifact in (
        sim_evidence / f"{stem}.dfg.report.json",
        sim_evidence / f"{stem}.mapping.json",
        sim_evidence / f"{stem}.cgra.report.json",
    ):
        if not artifact.is_file():
            raise AssertionError(f"CMSIS evidence mode should emit {artifact}")


def assert_cmsis_dfg_unsupported_row(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
    suite: str,
    case: str,
    stem: str,
    graph: str,
    diagnostic: str,
    expected_callee: str | None = None,
) -> None:
    row = one_row(rows, suite, case)
    if (
        row["status"] != "unsupported"
        or row["diagnostic_class"] != "dfg_report_unsupported"
        or row["blocking_prerequisite"] != "dfg_report"
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "unsupported"
        or row["mapping_status"] != "not_run"
        or row["cgra_status"] != "not_run"
        or row["comparison_status"] != "not_run"
        or row["mapping_artifact"] != ""
        or row["mapping_artifact_fingerprint"] != ""
        or row["cgra_report"] != ""
        or row["cgra_report_fingerprint"] != ""
        or row["comparison_report"] != ""
        or row["comparison_report_fingerprint"] != ""
        or row["final_outputs_present"] != "false"
        or row["final_memory_state_present"] != "false"
        or diagnostic not in row["diagnostic"]
    ):
        raise AssertionError(f"CMSIS row should expose a real DFG unsupported blocker: {row}")
    assert_sha256_file(row["dfg_report"], row["dfg_report_fingerprint"], repo)
    report_path = sim_evidence / f"{stem}.dfg.report.json"
    if not report_path.is_file():
        raise AssertionError(f"CMSIS evidence mode should emit {report_path}")
    for suffix in ("mapping.csv", "mapping.json", "cgra.report.json"):
        artifact = sim_evidence / f"{stem}.{suffix}"
        if artifact.exists():
            raise AssertionError(f"CMSIS unsupported DFG row must not emit {artifact}")
    report_data = json.loads(report_path.read_text())
    if (
        report_data.get("kind") != "dfg_sim_report"
        or report_data.get("workload") != case
        or report_data.get("graph") != graph
        or report_data.get("status") != "unsupported"
        or report_data.get("dynamic_work_items") != 0
        or report_data.get("final_outputs") != []
        or report_data.get("final_memory_state") != {}
        or diagnostic not in report_data.get("diagnostics", [])
    ):
        raise AssertionError(f"unexpected CMSIS unsupported DFG report: {report_data}")
    if expected_callee is not None:
        lowered_dfg = sim_evidence / f"{stem}.lowered.dfg.mlir"
        if not lowered_dfg.is_file():
            raise AssertionError(f"CMSIS unsupported DFG row should emit {lowered_dfg}")
        lowered_text = lowered_dfg.read_text()
        if "llvm.call" not in lowered_text or expected_callee not in lowered_text:
            raise AssertionError(
                f"CMSIS unsupported DFG row should preserve call to {expected_callee}: {lowered_dfg}"
            )


def assert_cmsis_mat_mult_f32_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    case = "MatrixFunctions/arm_mat_mult_f32.c"
    stem = "arm_mat_mult_f32"
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", case, stem)
    row = one_row(rows, "cmsis-dsp", case)
    report_path = sim_evidence / f"{stem}.dfg.report.json"
    if not report_path.is_file():
        raise AssertionError(f"CMSIS evidence mode should emit {report_path}")
    expected_memory = {
        "arg4": ["f32:66", "f32:72", "f32:78"],
        "arg9": [
            "f32:7",
            "f32:8",
            "f32:9",
            "f32:10",
            "f32:11",
            "f32:12",
            "f32:13",
            "f32:14",
            "f32:15",
        ],
        "arg10": ["f32:1", "f32:2", "f32:3"],
    }
    report_data = json.loads((repo / row["dfg_report"]).read_text())
    if (
        report_data.get("kind") != "dfg_sim_report"
        or report_data.get("workload") != case
        or report_data.get("graph") != "g_t_arm_mat_mult_f32_red_0_0"
        or report_data.get("status") != "pass"
        or report_data.get("dynamic_work_items") != 3
        or report_data.get("operation_fire_counts", {}).get("dataflow.load") != 18
        or report_data.get("operation_fire_counts", {}).get("llvm.intr.fmuladd") != 9
        or report_data.get("operation_fire_counts", {}).get("dataflow.store") != 3
        or report_data.get("final_outputs") != ["none", "i32:3"]
        or report_data.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected arm_mat_mult_f32 DFG evidence: {report_data}")
    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    if (
        mapping_artifact.get("kind") != "pnr_mapping"
        or mapping_artifact.get("workload") != case
        or mapping_artifact.get("hardware") != "shared_reduction_adg"
        or mapping_artifact.get("graph") != "g_t_arm_mat_mult_f32_red_0_0"
        or mapping_artifact.get("status") != "pass"
        or mapping_artifact.get("unrouted_edges") != 0
    ):
        raise AssertionError(f"unexpected arm_mat_mult_f32 mapping evidence: {mapping_artifact}")
    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        cgra_report.get("status") != "pass"
        or cgra_report.get("hardware") != "shared_reduction_adg"
        or cgra_report.get("final_outputs") != ["none", "i32:3"]
        or cgra_report.get("final_memory_state") != expected_memory
        or comparison_report.get("status") != "pass"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
    ):
        raise AssertionError(f"unexpected arm_mat_mult_f32 CGRA comparison evidence: {cgra_report} {comparison_report}")


def assert_cmsis_concat_memcpy_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
    *,
    case: str = "ConcatenationFunctions/arm_concatenation_s8_x.c",
    stem: str = "arm_concatenation_s8_x",
    graph: str = "g_t_arm_concatenation_s8_x_red_0_0",
    expected_hardware: str = "shared_reduction_adg",
    expected_memory: dict[str, list[str]] | None = None,
) -> None:
    if expected_memory is None:
        expected_memory = {
            "arg6": ["i8:1", "i8:2", "i8:3", "i8:4"],
            "arg7": ["i8:1", "i8:2", "i8:3", "i8:4", "i8:0", "i8:0"],
        }
    row = one_row(rows, "cmsis-nn", case)
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["blocking_prerequisite"] != ""
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["hardware_system"] != expected_hardware
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
        or row["diagnostic"] != "DFG-sim, mapping, CGRA-sim, and simulation comparison evidence passed"
    ):
        raise AssertionError(f"CMSIS concat row should expose real CGRA-sim evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
    if not (sim_evidence / f"{stem}.dfg.report.json").is_file():
        raise AssertionError(f"CMSIS evidence mode should emit {stem} DFG report")
    if not (sim_evidence / f"{stem}.mapping.json").is_file():
        raise AssertionError(f"CMSIS evidence mode should emit {stem} mapping artifact")
    if not (sim_evidence / f"{stem}.cgra.report.json").is_file():
        raise AssertionError(f"CMSIS evidence mode should emit {stem} CGRA report")

    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != case
        or dfg_report.get("graph") != graph
        or dfg_report.get("status") != "pass"
        or dfg_report.get("dynamic_work_items") != 4
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 4
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.store") != 4
        or "llvm.intr.memcpy" in dfg_report.get("operation_fire_counts", {})
        or dfg_report.get("final_outputs") != ["none"]
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected CMSIS concat DFG stream evidence: {dfg_report}")

    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    if (
        mapping_artifact.get("kind") != "pnr_mapping"
        or mapping_artifact.get("workload") != case
        or mapping_artifact.get("hardware") != expected_hardware
        or mapping_artifact.get("graph") != graph
        or mapping_artifact.get("status") != "pass"
        or mapping_artifact.get("unrouted_edges") != 0
    ):
        raise AssertionError(f"unexpected CMSIS concat PnR evidence: {mapping_artifact}")
    placements = mapping_artifact.get("placements", [])
    if not any(
        placement.get("operation") == "dataflow.load"
        and placement.get("resource_kind") == "fabric.mem.load"
        for placement in placements
    ):
        raise AssertionError(f"CMSIS concat mapping should place a real dataflow.load: {mapping_artifact}")
    if not any(
        placement.get("operation") == "dataflow.store"
        and placement.get("resource_kind") == "fabric.mem.store"
        for placement in placements
    ):
        raise AssertionError(f"CMSIS concat mapping should place a real dataflow.store: {mapping_artifact}")
    if (
        any(placement.get("resource_kind") == "fabric.mem.copy" for placement in placements)
        or "fabric.mem.copy" in json.dumps(mapping_artifact, sort_keys=True)
        or "memory_copy_binding" in json.dumps(mapping_artifact, sort_keys=True)
    ):
        raise AssertionError(f"CMSIS concat mapping must not use copy resources: {mapping_artifact}")

    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        cgra_report.get("status") != "pass"
        or cgra_report.get("hardware") != expected_hardware
        or cgra_report.get("final_memory_state") != expected_memory
        or comparison_report.get("status") != "pass"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
    ):
        raise AssertionError(
            f"unexpected CMSIS concat CGRA comparison evidence: {cgra_report} {comparison_report}"
        )


def assert_cmsis_concat_w_memcpy_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    assert_cmsis_concat_memcpy_cgra_evidence(
        repo,
        rows,
        sim_evidence,
        case="ConcatenationFunctions/arm_concatenation_s8_w.c",
        stem="arm_concatenation_s8_w",
        graph="g_arm_concatenation_s8_w_0",
        expected_hardware="shared_signal_window_adg",
        expected_memory={
            "arg1": ["i8:1", "i8:2", "i8:3", "i8:4"],
            "arg6": ["i8:0", "i8:0", "i8:1", "i8:2", "i8:3", "i8:4"],
        },
    )


def assert_cmsis_reshape_memcpy_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    row = one_row(rows, "cmsis-nn", "ReshapeFunctions/arm_reshape_s8.c")
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["blocking_prerequisite"] != ""
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["hardware_system"] != "shared_reduction_adg"
        or row["graph_ids"] != "g_arm_reshape_s8_0"
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
        or row["diagnostic"] != "DFG-sim, mapping, CGRA-sim, and simulation comparison evidence passed"
    ):
        raise AssertionError(f"CMSIS reshape row should expose real CGRA-sim evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
    if not (sim_evidence / "arm_reshape_s8.dfg.report.json").is_file():
        raise AssertionError("CMSIS evidence mode should emit arm_reshape_s8 DFG report")
    if not (sim_evidence / "arm_reshape_s8.mapping.json").is_file():
        raise AssertionError("CMSIS evidence mode should emit arm_reshape_s8 mapping artifact")
    if not (sim_evidence / "arm_reshape_s8.cgra.report.json").is_file():
        raise AssertionError("CMSIS evidence mode should emit arm_reshape_s8 CGRA report")

    expected_memory = {
        "arg1": ["i8:1", "i8:2", "i8:3", "i8:4"],
        "arg2": ["i8:1", "i8:2", "i8:3", "i8:4"],
    }
    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != "ReshapeFunctions/arm_reshape_s8.c"
        or dfg_report.get("graph") != "g_arm_reshape_s8_0"
        or dfg_report.get("status") != "pass"
        or dfg_report.get("dynamic_work_items") != 4
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 4
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.store") != 4
        or "llvm.intr.memcpy" in dfg_report.get("operation_fire_counts", {})
        or dfg_report.get("final_outputs") != ["none"]
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected CMSIS reshape DFG stream evidence: {dfg_report}")

    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    if (
        mapping_artifact.get("kind") != "pnr_mapping"
        or mapping_artifact.get("workload") != "ReshapeFunctions/arm_reshape_s8.c"
        or mapping_artifact.get("graph") != "g_arm_reshape_s8_0"
        or mapping_artifact.get("status") != "pass"
        or mapping_artifact.get("unrouted_edges") != 0
    ):
        raise AssertionError(f"unexpected CMSIS reshape PnR evidence: {mapping_artifact}")
    placements = mapping_artifact.get("placements", [])
    if not any(
        placement.get("operation") == "dataflow.load"
        and placement.get("resource_kind") == "fabric.mem.load"
        for placement in placements
    ):
        raise AssertionError(f"CMSIS reshape mapping should place a real dataflow.load: {mapping_artifact}")
    if not any(
        placement.get("operation") == "dataflow.store"
        and placement.get("resource_kind") == "fabric.mem.store"
        for placement in placements
    ):
        raise AssertionError(f"CMSIS reshape mapping should place a real dataflow.store: {mapping_artifact}")
    if (
        any(placement.get("resource_kind") == "fabric.mem.copy" for placement in placements)
        or "fabric.mem.copy" in json.dumps(mapping_artifact, sort_keys=True)
        or "memory_copy_binding" in json.dumps(mapping_artifact, sort_keys=True)
    ):
        raise AssertionError(f"CMSIS reshape mapping must not use copy resources: {mapping_artifact}")

    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        cgra_report.get("status") != "pass"
        or cgra_report.get("final_memory_state") != expected_memory
        or comparison_report.get("status") != "pass"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
    ):
        raise AssertionError(
            f"unexpected CMSIS reshape CGRA comparison evidence: {cgra_report} {comparison_report}"
        )


def assert_resource_pressure_record(
    mapping: dict[str, object],
    *,
    resource_kind: str,
    operation: str,
    required: int,
    available: int,
    placed: int,
    missing: int,
    label: str,
) -> None:
    records = mapping.get("resource_pressure")
    if not isinstance(records, list):
        raise AssertionError(f"{label} should expose resource_pressure records: {mapping}")
    for record in records:
        if not isinstance(record, dict):
            continue
        if (
            record.get("resource_kind") != resource_kind
            or record.get("operation") != operation
        ):
            continue
        expected = {
            "required": required,
            "available": available,
            "placed": placed,
            "missing": missing,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise AssertionError(
                    f"{label} resource_pressure {key}={record.get(key)!r}, expected {value!r}: {mapping}"
                )
        return
    raise AssertionError(f"{label} missing resource_pressure for {resource_kind}/{operation}: {mapping}")


def assert_cmsis_cfft_component_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    row = one_row(rows, "cmsis-dsp", "TransformFunctions/arm_cfft_f32.c")
    expected_graphs = {
        "g_t_arm_cfft_f32_red_0_0",
        "g_t_arm_cfft_f32_red_1_0",
        "g_t_arm_cfft_f32_red_2_0",
        "g_t_arm_cfft_f32_red_3_0",
    }
    if set(row["graph_ids"].split(",")) != expected_graphs:
        raise AssertionError(f"arm_cfft_f32 row should keep all component graph ids: {row}")
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["blocking_prerequisite"] != ""
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["required_slice_count"] != "4"
        or row["hardware_system"] != "shared_signal_window_adg"
        or row["diagnostic"] != "DFG-sim, mapping, CGRA-sim, and simulation comparison evidence passed"
    ):
        raise AssertionError(f"arm_cfft_f32 should be row-complete CGRA-sim pass evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)

    required_artifacts = (
        "arm_cfft_f32.red0.dfg.report.json",
        "arm_cfft_f32.red0.mapping.json",
        "arm_cfft_f32.red0.cgra.report.json",
        "arm_cfft_f32.red1.dfg.report.json",
        "arm_cfft_f32.red1.mapping.json",
        "arm_cfft_f32.red1.cgra.report.json",
        "arm_cfft_f32.red2.dfg.report.json",
        "arm_cfft_f32.red2.mapping.json",
        "arm_cfft_f32.red2.cgra.report.json",
        "arm_cfft_f32.red3.dfg.report.json",
        "arm_cfft_f32.red3.mapping.json",
        "arm_cfft_f32.red3.cgra.report.json",
        "arm_cfft_f32.dfg.report.json",
        "arm_cfft_f32.mapping.json",
        "arm_cfft_f32.cgra.report.json",
    )
    for artifact_name in required_artifacts:
        artifact = sim_evidence / artifact_name
        if not artifact.is_file():
            raise AssertionError(f"arm_cfft_f32 component evidence should emit {artifact}")
    red0_memory = {"arg4": ["f32:-1", "f32:2", "f32:-3", "f32:4", "f32:5", "f32:6", "f32:7", "f32:8"]}
    red0_dfg = json.loads((sim_evidence / "arm_cfft_f32.red0.dfg.report.json").read_text())
    if (
        red0_dfg.get("workload") != "TransformFunctions/arm_cfft_f32.c"
        or red0_dfg.get("graph") != "g_t_arm_cfft_f32_red_0_0"
        or red0_dfg.get("status") != "pass"
        or red0_dfg.get("dynamic_work_items") != 3
        or red0_dfg.get("operation_fire_counts", {}).get("dataflow.load") != 2
        or red0_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 2
        or red0_dfg.get("operation_fire_counts", {}).get("llvm.fneg") != 2
        or red0_dfg.get("final_memory_state") != red0_memory
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red0 DFG evidence: {red0_dfg}")
    red0_mapping = json.loads((sim_evidence / "arm_cfft_f32.red0.mapping.json").read_text())
    if (
        red0_mapping.get("status") != "pass"
        or red0_mapping.get("hardware") != "shared_signal_window_adg"
        or red0_mapping.get("placed_records") != 6
        or red0_mapping.get("routed_edges") != 6
        or red0_mapping.get("unrouted_edges") != 0
        or red0_mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red0 mapping evidence: {red0_mapping}")
    red0_cgra = json.loads((sim_evidence / "arm_cfft_f32.red0.cgra.report.json").read_text())
    if (
        red0_cgra.get("status") != "pass"
        or red0_cgra.get("dfg_cycles") != 49
        or red0_cgra.get("hardware_aware_cycles") != 94
        or red0_cgra.get("final_outputs") != ["none"]
        or red0_cgra.get("final_memory_state") != red0_memory
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red0 CGRA evidence: {red0_cgra}")

    red1_memory = {
        "arg4": ["f32:5", "f32:7", "f32:9", "f32:11", "f32:5", "f32:15", "f32:17", "f32:19", "f32:9", "f32:10", "f32:11", "f32:12"],
        "arg5": ["f32:5", "f32:7", "f32:9", "f32:11", "f32:6", "f32:15", "f32:17", "f32:19", "f32:10", "f32:11", "f32:12", "f32:13"],
        "arg6": ["f32:1", "f32:11", "f32:1", "f32:15", "f32:7", "f32:8", "f32:1", "f32:23", "f32:11", "f32:12", "f32:13", "f32:14"],
        "arg7": ["f32:-33", "f32:3", "f32:-45", "f32:3", "f32:8", "f32:9", "f32:-69", "f32:3", "f32:12", "f32:13", "f32:14", "f32:15"],
        "arg8": ["f32:5", "f32:6", "f32:7", "f32:8", "f32:9", "f32:10", "f32:11", "f32:12", "f32:13", "f32:14", "f32:15", "f32:16"],
    }
    red1_dfg = json.loads((sim_evidence / "arm_cfft_f32.red1.dfg.report.json").read_text())
    if (
        red1_dfg.get("workload") != "TransformFunctions/arm_cfft_f32.c"
        or red1_dfg.get("graph") != "g_t_arm_cfft_f32_red_1_0"
        or red1_dfg.get("status") != "pass"
        or red1_dfg.get("dynamic_work_items") != 2
        or red1_dfg.get("optimistic_cycles") != 503
        or red1_dfg.get("operation_fire_counts", {}).get("llvm.load") != 30
        or red1_dfg.get("operation_fire_counts", {}).get("llvm.store") != 22
        or red1_dfg.get("operation_fire_counts", {}).get("arith.mulf") != 26
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 4
        or red1_dfg.get("final_outputs") != ["none"]
        or red1_dfg.get("final_memory_state") != red1_memory
        or red1_dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red1 DFG evidence: {red1_dfg}")
    red1_mapping = json.loads((sim_evidence / "arm_cfft_f32.red1.mapping.json").read_text())
    if (
        red1_mapping.get("status") != "pass"
        or red1_mapping.get("hardware") != "shared_signal_window_adg"
        or red1_mapping.get("placed_records") != 79
        or red1_mapping.get("unplaced_records") != 0
        or red1_mapping.get("routed_edges") != 114
        or red1_mapping.get("unrouted_edges") != 0
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red1 mapping evidence: {red1_mapping}")
    red1_cgra = json.loads((sim_evidence / "arm_cfft_f32.red1.cgra.report.json").read_text())
    if (
        red1_cgra.get("status") != "pass"
        or red1_cgra.get("dfg_cycles") != 503
        or red1_cgra.get("hardware_aware_cycles") != 1209
        or red1_cgra.get("final_outputs") != ["none"]
        or red1_cgra.get("final_memory_state") != red1_memory
        or red1_cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red1 CGRA evidence: {red1_cgra}")

    red2_dfg = json.loads((sim_evidence / "arm_cfft_f32.red2.dfg.report.json").read_text())
    red2_memory = red2_dfg.get("final_memory_state", {})
    red2_expected_prefixes = {
        "arg4": ["f32:400", "f32:401", "f32:402", "f32:403", "f32:404", "f32:405", "f32:406", "f32:407", "f32:2832", "f32:409", "f32:410", "f32:2844"],
        "arg6": ["f32:600", "f32:601", "f32:602", "f32:603", "f32:604", "f32:605", "f32:606", "f32:607", "f32:-907100", "f32:609", "f32:610", "f32:611"],
        "arg12": ["f32:1200", "f32:1201", "f32:1202", "f32:1203", "f32:1204", "f32:5420", "f32:1206", "f32:5428", "f32:5432", "f32:1209", "f32:1210", "f32:1211"],
        "arg13": ["f32:1300", "f32:1301", "f32:1302", "f32:1303", "f32:1304", "f32:1305", "f32:1306", "f32:-644400", "f32:203200", "f32:1309", "f32:1310", "f32:1311"],
        "arg21": ["f32:2100", "f32:2101", "f32:2102", "f32:2103", "f32:2104", "f32:2105", "f32:2106", "f32:2107", "f32:2108", "f32:2109", "f32:2110", "f32:68200"],
    }
    if (
        red2_dfg.get("workload") != "TransformFunctions/arm_cfft_f32.c"
        or red2_dfg.get("graph") != "g_t_arm_cfft_f32_red_2_0"
        or red2_dfg.get("status") != "pass"
        or red2_dfg.get("dynamic_work_items") != 2
        or red2_dfg.get("optimistic_cycles") != 641
        or red2_dfg.get("operation_fire_counts", {}).get("llvm.load") != 30
        or red2_dfg.get("operation_fire_counts", {}).get("llvm.store") != 10
        or red2_dfg.get("operation_fire_counts", {}).get("arith.addf") != 29
        or red2_dfg.get("operation_fire_counts", {}).get("arith.subf") != 35
        or red2_dfg.get("operation_fire_counts", {}).get("arith.mulf") != 26
        or red2_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 8
        or red2_dfg.get("final_outputs") != ["none"]
        or red2_dfg.get("diagnostics") != []
        or any(red2_memory.get(arg, [])[:12] != expected for arg, expected in red2_expected_prefixes.items())
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red2 DFG evidence: {red2_dfg}")
    red2_mapping = json.loads((sim_evidence / "arm_cfft_f32.red2.mapping.json").read_text())
    if (
        red2_mapping.get("status") != "pass"
        or red2_mapping.get("hardware") != "shared_signal_window_adg"
        or red2_mapping.get("placed_records") != 128
        or red2_mapping.get("unplaced_records") != 0
        or red2_mapping.get("routed_edges") != 212
        or red2_mapping.get("unrouted_edges") != 0
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red2 mapping evidence: {red2_mapping}")
    red2_cgra = json.loads((sim_evidence / "arm_cfft_f32.red2.cgra.report.json").read_text())
    if (
        red2_cgra.get("status") != "pass"
        or red2_cgra.get("dfg_cycles") != 641
        or red2_cgra.get("hardware_aware_cycles") != 1839
        or red2_cgra.get("final_outputs") != ["none"]
        or any(red2_cgra.get("final_memory_state", {}).get(arg, [])[:12] != expected for arg, expected in red2_expected_prefixes.items())
        or red2_cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red2 CGRA evidence: {red2_cgra}")

    red3_memory = {"arg5": ["f32:0.500000", "f32:-1", "f32:1.500000", "f32:-2", "f32:5", "f32:6", "f32:7", "f32:8"]}
    red3_dfg = json.loads((sim_evidence / "arm_cfft_f32.red3.dfg.report.json").read_text())
    if (
        red3_dfg.get("workload") != "TransformFunctions/arm_cfft_f32.c"
        or red3_dfg.get("graph") != "g_t_arm_cfft_f32_red_3_0"
        or red3_dfg.get("status") != "pass"
        or red3_dfg.get("dynamic_work_items") != 3
        or red3_dfg.get("operation_fire_counts", {}).get("dataflow.load") != 2
        or red3_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 2
        or red3_dfg.get("operation_fire_counts", {}).get("llvm.fneg") != 2
        or red3_dfg.get("operation_fire_counts", {}).get("llvm.store") != 2
        or red3_dfg.get("final_memory_state") != red3_memory
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red3 DFG evidence: {red3_dfg}")
    red3_mapping = json.loads((sim_evidence / "arm_cfft_f32.red3.mapping.json").read_text())
    if (
        red3_mapping.get("status") != "pass"
        or red3_mapping.get("hardware") != "shared_signal_window_adg"
        or red3_mapping.get("placed_records") != 11
        or red3_mapping.get("routed_edges") != 12
        or red3_mapping.get("unplaced_records") != 0
        or red3_mapping.get("unrouted_edges") != 0
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red3 mapping evidence: {red3_mapping}")
    red3_placements = red3_mapping.get("placements", [])
    if sum(1 for placement in red3_placements if placement.get("operation") == "arith.mulf") != 2:
        raise AssertionError(f"arm_cfft_f32 red3 should place both FP multiplies: {red3_mapping}")
    red3_cgra = json.loads((sim_evidence / "arm_cfft_f32.red3.cgra.report.json").read_text())
    red3_dfg_cycles = red3_cgra.get("dfg_cycles")
    red3_cgra_cycles = red3_cgra.get("hardware_aware_cycles")
    if (
        red3_cgra.get("status") != "pass"
        or red3_cgra.get("dfg_cycles") != 88
        or red3_cgra.get("hardware_aware_cycles") != 179
        or red3_cgra.get("final_outputs") != ["none"]
        or red3_cgra.get("final_memory_state") != red3_memory
        or not isinstance(red3_dfg_cycles, int)
        or not isinstance(red3_cgra_cycles, int)
        or red3_cgra_cycles < red3_dfg_cycles
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 red3 CGRA evidence: {red3_cgra}")

    aggregate_dfg = json.loads((sim_evidence / "arm_cfft_f32.dfg.report.json").read_text())
    if (
        aggregate_dfg.get("status") != "pass"
        or aggregate_dfg.get("graph") != "workload_graph_set"
        or set(aggregate_dfg.get("component_graphs", [])) != expected_graphs
        or aggregate_dfg.get("dynamic_work_items") != 10
        or aggregate_dfg.get("optimistic_cycles") != 1281
        or aggregate_dfg.get("operation_fire_counts", {}).get("arith.addf") != 49
        or aggregate_dfg.get("operation_fire_counts", {}).get("arith.subf") != 55
        or aggregate_dfg.get("operation_fire_counts", {}).get("arith.mulf") != 56
        or aggregate_dfg.get("operation_fire_counts", {}).get("llvm.load") != 62
        or aggregate_dfg.get("operation_fire_counts", {}).get("llvm.store") != 34
        or aggregate_dfg.get("final_outputs") != ["none", "none", "none", "none"]
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 aggregate DFG evidence: {aggregate_dfg}")
    aggregate_mapping = json.loads((sim_evidence / "arm_cfft_f32.mapping.json").read_text())
    if (
        aggregate_mapping.get("status") != "pass"
        or aggregate_mapping.get("hardware") != "shared_signal_window_adg"
        or aggregate_mapping.get("graph") != "workload_graph_set"
        or set(aggregate_mapping.get("component_graphs", [])) != expected_graphs
        or aggregate_mapping.get("placed_records") != 224
        or aggregate_mapping.get("routed_edges") != 344
        or aggregate_mapping.get("unplaced_records") != 0
        or aggregate_mapping.get("unrouted_edges") != 0
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 aggregate mapping evidence: {aggregate_mapping}")
    aggregate_cgra = json.loads((sim_evidence / "arm_cfft_f32.cgra.report.json").read_text())
    if (
        aggregate_cgra.get("status") != "pass"
        or aggregate_cgra.get("hardware") != "shared_signal_window_adg"
        or aggregate_cgra.get("graph") != "workload_graph_set"
        or set(aggregate_cgra.get("component_graphs", [])) != expected_graphs
        or aggregate_cgra.get("dfg_cycles") != 1281
        or aggregate_cgra.get("hardware_aware_cycles") != 3321
        or aggregate_cgra.get("final_outputs") != ["none", "none", "none", "none"]
        or aggregate_cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
    ):
        raise AssertionError(f"unexpected arm_cfft_f32 aggregate CGRA evidence: {aggregate_cgra}")


def assert_cmsis_fir_component_pass_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    row = one_row(rows, "cmsis-dsp", "FilteringFunctions/arm_fir_f32.c")
    expected_graphs = {
        "g_t_arm_fir_f32_red_0_0",
        "g_t_arm_fir_f32_red_1_0",
    }
    if set(row["graph_ids"].split(",")) != expected_graphs:
        raise AssertionError(f"arm_fir_f32 row should keep both component graph ids: {row}")
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["blocking_prerequisite"] != ""
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["required_slice_count"] != "2"
        or row["hardware_system"] != "shared_reduction_adg"
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
        or "DFG-sim, mapping, CGRA-sim, and simulation comparison evidence passed"
        not in row["diagnostic"]
    ):
        raise AssertionError(f"arm_fir_f32 should expose workload-level CGRA-sim pass evidence: {row}")
    expected_row_artifacts = {
        "dfg_report": "arm_fir_f32.dfg.report.json",
        "mapping_artifact": "arm_fir_f32.mapping.json",
        "cgra_report": "arm_fir_f32.cgra.report.json",
        "comparison_report": "arm_fir_f32.c.sim-comparison-report.json",
    }
    for key, expected_name in expected_row_artifacts.items():
        if Path(row[key]).name != expected_name:
            raise AssertionError(f"arm_fir_f32 row {key} should reference {expected_name}: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)

    required_artifacts = (
        "arm_fir_f32.dfg.report.json",
        "arm_fir_f32.mapping.json",
        "arm_fir_f32.mapping.csv",
        "arm_fir_f32.cgra.report.json",
        "arm_fir_f32.red0.dfg.report.json",
        "arm_fir_f32.red1.dfg.report.json",
        "arm_fir_f32.red0.mapping.json",
        "arm_fir_f32.red1.mapping.json",
        "arm_fir_f32.red0.cgra.report.json",
        "arm_fir_f32.red1.cgra.report.json",
    )
    for artifact_name in required_artifacts:
        artifact = sim_evidence / artifact_name
        if not artifact.is_file():
            raise AssertionError(f"arm_fir_f32 component evidence should emit {artifact}")
    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    for data, kind, status in (
        (dfg_report, "dfg_sim_report", "pass"),
        (mapping_artifact, "pnr_mapping", "pass"),
        (cgra_report, "cgra_sim_report", "pass"),
    ):
        if (
            data.get("kind") != kind
            or data.get("aggregation_kind") != "workload_graph_set"
            or data.get("component_graphs") != sorted(expected_graphs)
            or data.get("status") != status
        ):
            raise AssertionError(f"unexpected arm_fir_f32 aggregate evidence: {data}")
    source_identity = Path(row["dfg_mlir"]).stem
    input_fingerprints = dfg_report.get("input_artifact_fingerprints", {})
    if input_fingerprints.get(source_identity) != row["dfg_mlir_fingerprint"]:
        raise AssertionError(f"arm_fir_f32 aggregate DFG should retain source MLIR fingerprint: {dfg_report}")
    red0_mapping = json.loads((sim_evidence / "arm_fir_f32.red0.mapping.json").read_text())
    red1_mapping = json.loads((sim_evidence / "arm_fir_f32.red1.mapping.json").read_text())
    red0_cgra = json.loads((sim_evidence / "arm_fir_f32.red0.cgra.report.json").read_text())
    red1_cgra = json.loads((sim_evidence / "arm_fir_f32.red1.cgra.report.json").read_text())
    if (
        red0_mapping.get("status") != "pass"
        or red1_mapping.get("status") != "pass"
        or red0_cgra.get("status") != "pass"
        or red1_cgra.get("status") != "pass"
    ):
        raise AssertionError(
            f"arm_fir_f32 component evidence should preserve red0 and red1 pass records: "
            f"{red0_mapping} {red1_mapping} {red0_cgra} {red1_cgra}"
        )

    red0_dfg = json.loads((sim_evidence / "arm_fir_f32.red0.dfg.report.json").read_text())
    red0_memory = {
        "arg8": ["f32:1", "f32:2", "f32:3", "f32:4"],
        "arg9": ["f32:1", "f32:2", "f32:3", "f32:4"],
        "arg10": ["f32:0", "f32:0", "f32:0", "f32:0"],
    }
    if (
        red0_dfg.get("workload") != "FilteringFunctions/arm_fir_f32.c"
        or red0_dfg.get("graph") != "g_t_arm_fir_f32_red_0_0"
        or red0_dfg.get("status") != "pass"
        or red0_dfg.get("dynamic_work_items") != 4
        or red0_dfg.get("operation_fire_counts", {}).get("llvm.getelementptr") != 16
        or red0_dfg.get("final_outputs") != ["none"]
        or red0_dfg.get("final_memory_state") != red0_memory
    ):
        raise AssertionError(f"unexpected arm_fir_f32 red0 DFG evidence: {red0_dfg}")

    red1_memory = {
        "arg4": ["f32:1", "f32:2", "f32:3", "f32:4"],
        "arg5": ["f32:1", "f32:2", "f32:3", "f32:4"],
    }
    red1_dfg = json.loads((sim_evidence / "arm_fir_f32.red1.dfg.report.json").read_text())
    if (
        red1_dfg.get("workload") != "FilteringFunctions/arm_fir_f32.c"
        or red1_dfg.get("graph") != "g_t_arm_fir_f32_red_1_0"
        or red1_dfg.get("status") != "pass"
        or red1_dfg.get("dynamic_work_items") != 4
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.load") != 4
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 4
        or red1_dfg.get("final_outputs") != ["none"]
        or red1_dfg.get("final_memory_state") != red1_memory
    ):
        raise AssertionError(f"unexpected arm_fir_f32 red1 DFG evidence: {red1_dfg}")

    component_expectations = (
        ("red0", red0_mapping, red0_cgra, red0_dfg, red0_memory),
        ("red1", red1_mapping, red1_cgra, red1_dfg, red1_memory),
    )
    for label, mapping, cgra, dfg, memory in component_expectations:
        placed_records = mapping.get("placed_records")
        routed_edges = mapping.get("routed_edges")
        route_records = mapping.get("routes", [])
        config_records = mapping.get("config_records")
        if (
            mapping.get("status") != "pass"
            or mapping.get("hardware") != "shared_reduction_adg"
            or not isinstance(placed_records, int)
            or placed_records <= 0
            or not isinstance(routed_edges, int)
            or routed_edges <= 0
            or mapping.get("unrouted_edges") != 0
            or mapping.get("unplaced_records") != 0
            or not isinstance(route_records, list)
            or len(route_records) != routed_edges
            or not isinstance(config_records, int)
            or config_records <= 0
        ):
            raise AssertionError(f"unexpected arm_fir_f32 {label} mapping evidence: {mapping}")

        dfg_cycles = cgra.get("dfg_cycles")
        cgra_cycles = cgra.get("hardware_aware_cycles")
        cgra_route_segments = cgra.get("route_segments")
        cgra_config_records = cgra.get("config_records")
        if (
            cgra.get("status") != "pass"
            or not isinstance(dfg_cycles, int)
            or dfg_cycles != dfg.get("optimistic_cycles")
            or not isinstance(cgra_cycles, int)
            or cgra_cycles < dfg_cycles
            or not isinstance(cgra_route_segments, int)
            or cgra_route_segments <= 0
            or not isinstance(cgra_config_records, int)
            or cgra_config_records <= 0
            or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
            or cgra.get("final_outputs") != ["none"]
            or cgra.get("final_memory_state") != memory
        ):
            raise AssertionError(f"unexpected arm_fir_f32 {label} CGRA evidence: {cgra}")

    aggregate_dfg_cycles = cgra_report.get("dfg_cycles")
    aggregate_cgra_cycles = cgra_report.get("hardware_aware_cycles")
    if (
        mapping_artifact.get("placed_records") != red0_mapping.get("placed_records") + red1_mapping.get("placed_records")
        or mapping_artifact.get("unrouted_edges") != 0
        or mapping_artifact.get("unplaced_records") != 0
        or not isinstance(aggregate_dfg_cycles, int)
        or aggregate_dfg_cycles != red0_cgra.get("dfg_cycles") + red1_cgra.get("dfg_cycles")
        or not isinstance(aggregate_cgra_cycles, int)
        or aggregate_cgra_cycles < aggregate_dfg_cycles
    ):
        raise AssertionError(f"unexpected arm_fir_f32 aggregate mapping/CGRA evidence: {mapping_artifact} {cgra_report}")

    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        comparison_report.get("status") != "pass"
        or comparison_report.get("workload") != "FilteringFunctions/arm_fir_f32.c"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
        or comparison_report.get("performance_comparison_status") != "pass"
        or comparison_report.get("cgra_sim_cycles") != aggregate_cgra_cycles
        or comparison_report.get("dfg_sim_cycles") != aggregate_dfg_cycles
        or comparison_report.get("cgra_sim_cycles") < comparison_report.get("dfg_sim_cycles")
    ):
        raise AssertionError(f"unexpected arm_fir_f32 comparison evidence: {comparison_report}")


def assert_app_cgra_pass_row(
    repo: Path,
    rows: list[dict[str, str]],
    case: str,
    *,
    expected_hardware: str,
) -> None:
    row = one_row(rows, "app", case)
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["hardware_system"] != expected_hardware
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
    ):
        raise AssertionError(f"app row should expose real CGRA-sim evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)


def assert_cmsis_dfg_blocker_row(
    repo: Path,
    rows: list[dict[str, str]],
    suite: str,
    case: str,
    *,
    dfg_status: str,
    diagnostic_class: str,
    diagnostic_substring: str,
) -> None:
    row = one_row(rows, suite, case)
    if (
        row["status"] != "blocked"
        or row["diagnostic_class"] != diagnostic_class
        or row["blocking_prerequisite"] != "dfg_report"
        or row["owner"] != "sim_report"
        or row["dfg_status"] != dfg_status
        or row["mapping_status"] != "not_run"
        or row["cgra_status"] != "not_run"
        or diagnostic_substring not in row["diagnostic"]
    ):
        raise AssertionError(f"CMSIS row should expose exact DFG-sim blocker evidence: {row}")
    assert_sha256_file(row["dfg_report"], row["dfg_report_fingerprint"], repo)


def assert_cmsis_dfg_ready_for_mapping_row(
    repo: Path,
    rows: list[dict[str, str]],
    suite: str,
    case: str,
) -> None:
    row = one_row(rows, suite, case)
    if (
        row["status"] != "blocked"
        or row["diagnostic_class"] != "missing_mapping_artifact"
        or row["blocking_prerequisite"] != "mapping_artifact"
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "not_run"
        or row["cgra_status"] != "not_run"
        or row["comparison_status"] != "not_run"
        or row["mapping_artifact"]
        or row["cgra_report"]
        or row["comparison_report"]
        or "PnR mapping artifact is absent" not in row["diagnostic"]
    ):
        raise AssertionError(f"CMSIS row should expose DFG pass evidence and a missing mapping artifact: {row}")
    assert_sha256_file(row["dfg_report"], row["dfg_report_fingerprint"], repo)


def assert_cmsis_mapping_blocker_row(
    repo: Path,
    rows: list[dict[str, str]],
    suite: str,
    case: str,
    *,
    diagnostic_substring: str,
) -> None:
    row = one_row(rows, suite, case)
    if (
        row["status"] != "fail"
        or row["diagnostic_class"] != "mapping_artifact_failed"
        or row["blocking_prerequisite"] != "mapping_artifact"
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "fail"
        or row["cgra_status"] != "blocked"
        or row["comparison_status"] != "blocked"
        or not row["mapping_artifact"]
        or not row["cgra_report"]
        or not row["comparison_report"]
        or row["hardware_system"] != "shared_reduction_adg"
        or diagnostic_substring not in row["diagnostic"]
    ):
        raise AssertionError(f"CMSIS row should expose exact PnR mapping blocker evidence: {row}")
    assert_sha256_file(row["dfg_report"], row["dfg_report_fingerprint"], repo)
    assert_sha256_file(row["mapping_artifact"], row["mapping_artifact_fingerprint"], repo)
    assert_sha256_file(row["cgra_report"], row["cgra_report_fingerprint"], repo)
    assert_sha256_file(row["comparison_report"], row["comparison_report_fingerprint"], repo)
    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    mapping_diagnostics = " ".join(mapping_artifact.get("diagnostics", []))
    if (
        mapping_artifact.get("kind") != "pnr_mapping"
        or mapping_artifact.get("workload") != case
        or mapping_artifact.get("hardware") != "shared_reduction_adg"
        or mapping_artifact.get("status") != "fail"
        or mapping_artifact.get("unrouted_edges", 0) <= 0
        or diagnostic_substring not in mapping_diagnostics
        or not mapping_artifact.get("unrouted_edge_details")
    ):
        raise AssertionError(f"unexpected CMSIS mapping blocker artifact: {mapping_artifact}")
    cgra_diagnostics = " ".join(cgra_report.get("diagnostics", []))
    if (
        cgra_report.get("kind") != "cgra_sim_report"
        or cgra_report.get("workload") != case
        or cgra_report.get("hardware") != "shared_reduction_adg"
        or cgra_report.get("status") != "blocked"
        or diagnostic_substring not in cgra_diagnostics
    ):
        raise AssertionError(f"unexpected blocked CMSIS CGRA report: {cgra_report}")
    if (
        comparison_report.get("kind") != "sim_comparison_report"
        or comparison_report.get("workload") != case
        or comparison_report.get("status") != "blocked"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
        or comparison_report.get("performance_comparison_status") != "blocked"
    ):
        raise AssertionError(f"unexpected blocked CMSIS comparison report: {comparison_report}")


def assert_cmsis_vector_sum_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    assert_cmsis_cgra_pass_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-nn",
        "FullyConnectedFunctions/arm_vector_sum_s8.c",
        "arm_vector_sum_s8",
    )
    row = one_row(rows, "cmsis-nn", "FullyConnectedFunctions/arm_vector_sum_s8.c")
    expected_memory = {
        "arg8": ["i32:3", "i32:7"],
        "arg9": ["i8:1", "i8:2", "i8:3", "i8:4"],
    }

    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != "FullyConnectedFunctions/arm_vector_sum_s8.c"
        or dfg_report.get("graph") != "g_t_arm_vector_sum_s8_red_0_0"
        or dfg_report.get("status") != "pass"
        or dfg_report.get("optimistic_cycles") != 86
        or dfg_report.get("dynamic_work_items") != 2
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 6
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.store") != 2
        or dfg_report.get("operation_fire_counts", {}).get("arith.addi") != 8
        or dfg_report.get("operation_fire_counts", {}).get("arith.muli") != 2
        or dfg_report.get("operation_fire_counts", {}).get("llvm.sext") != 4
        or dfg_report.get("final_outputs") != ["none"]
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected arm_vector_sum_s8 DFG evidence: {dfg_report}")

    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    expected_mapping = {
        "kind": "pnr_mapping",
        "workload": "FullyConnectedFunctions/arm_vector_sum_s8.c",
        "graph": "g_t_arm_vector_sum_s8_red_0_0",
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "placed_records": 11,
        "routed_edges": 10,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 264,
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(
                f"arm_vector_sum_s8 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}"
            )
    routes_by_edge = {
        route.get("edge_ref"): route
        for route in mapping_artifact.get("routes", [])
        if isinstance(route, dict)
    }
    required_routes = {
        "arith.addi#0.result0->arith.addi#1.operand0": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.operand0",
        ),
        "arith.addi#1.result0->arith.muli#0.operand0": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.operand0",
        ),
        "arith.muli#0.result0->arith.addi#2.operand1": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.operand1",
        ),
        "dataflow.load#1.result0->arith.addi#2.operand0": (
            "shared_reduction_adg::mem.load#1.result0",
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.operand0",
        ),
        "arith.addi#2.result0->dataflow.store#0.operand2": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            "shared_reduction_adg::mem.store#0.operand1",
        ),
    }
    for edge_ref, (source_endpoint, sink_endpoint) in required_routes.items():
        route = routes_by_edge.get(edge_ref)
        if route is None:
            raise AssertionError(f"arm_vector_sum_s8 mapping missed route {edge_ref}: {mapping_artifact}")
        assert_routed_endpoint_shape(
            route,
            edge_ref,
            source_endpoint=source_endpoint,
            sink_endpoint=sink_endpoint,
            label="arm_vector_sum_s8",
        )

    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        cgra_report.get("kind") != "cgra_sim_report"
        or cgra_report.get("workload") != "FullyConnectedFunctions/arm_vector_sum_s8.c"
        or cgra_report.get("hardware") != "shared_reduction_adg"
        or cgra_report.get("status") != "pass"
        or cgra_report.get("dfg_cycles") != 86
        or cgra_report.get("hardware_aware_cycles") != 160
        or cgra_report.get("fidelity_level") != "mapping_constraint_estimate"
        or cgra_report.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra_report.get("final_outputs") != ["none"]
        or cgra_report.get("final_memory_state") != expected_memory
        or comparison_report.get("status") != "pass"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
        or comparison_report.get("performance_comparison_status") != "pass"
    ):
        raise AssertionError(
            f"unexpected arm_vector_sum_s8 CGRA comparison evidence: {cgra_report} {comparison_report}"
        )


def assert_cmsis_minmax_s8_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
    *,
    case: str,
    stem: str,
    graph: str,
    intrinsic: str,
    expected_output: list[str],
) -> None:
    row = one_row(rows, "cmsis-nn", case)
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["blocking_prerequisite"] != ""
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["hardware_system"] != "shared_memory_reduction_adg"
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
    ):
        raise AssertionError(f"{stem} row should expose real shared-memory CGRA evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
    for key in ("dfg_report", "mapping_artifact", "cgra_report"):
        artifact = sim_evidence / Path(row[key]).name
        if not artifact.is_file():
            raise AssertionError(f"attempt manifest should emit {artifact}")
    expected_memory = {
        "arg40": ["i8:3", "i8:-4", "i8:7"],
        "arg41": ["i8:2", "i8:5", "i8:-9"],
        "arg42": expected_output,
    }

    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != case
        or dfg_report.get("graph") != graph
        or dfg_report.get("status") != "pass"
        or dfg_report.get("dynamic_work_items") != 3
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 6
        or dfg_report.get("operation_fire_counts", {}).get(intrinsic) != 3
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.store") != 3
        or dfg_report.get("final_outputs") != ["none"]
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected {stem} DFG evidence: {dfg_report}")

    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    placements = [
        placement
        for placement in mapping_artifact.get("placements", [])
        if isinstance(placement, dict)
    ]
    placement_counts: dict[str, int] = {}
    for placement in placements:
        operation = placement.get("operation")
        if isinstance(operation, str):
            placement_counts[operation] = placement_counts.get(operation, 0) + 1
    if (
        mapping_artifact.get("kind") != "pnr_mapping"
        or mapping_artifact.get("workload") != case
        or mapping_artifact.get("graph") != graph
        or mapping_artifact.get("hardware") != "shared_memory_reduction_adg"
        or mapping_artifact.get("status") != "pass"
        or mapping_artifact.get("unplaced_records") != 0
        or mapping_artifact.get("unrouted_edges") != 0
        or mapping_artifact.get("placed_records") != 84
        or mapping_artifact.get("routed_edges") != 55
        or placement_counts.get("dataflow.load") != 18
        or placement_counts.get("dataflow.store") != 9
        or placement_counts.get(intrinsic) != 9
        or "resource_pressure" in mapping_artifact
    ):
        raise AssertionError(f"unexpected {stem} mapping evidence: {mapping_artifact}")

    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        cgra_report.get("kind") != "cgra_sim_report"
        or cgra_report.get("workload") != case
        or cgra_report.get("hardware") != "shared_memory_reduction_adg"
        or cgra_report.get("status") != "pass"
        or cgra_report.get("dfg_cycles") != 92
        or cgra_report.get("hardware_aware_cycles") != 521
        or cgra_report.get("fidelity_level") != "mapping_constraint_estimate"
        or cgra_report.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra_report.get("final_outputs") != ["none"]
        or cgra_report.get("final_memory_state") != expected_memory
        or comparison_report.get("status") != "pass"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
        or comparison_report.get("performance_comparison_status") != "pass"
        or comparison_report.get("cgra_sim_cycles") != 521
        or comparison_report.get("dfg_sim_cycles") != 92
        or comparison_report.get("cgra_sim_cycles") < comparison_report.get("dfg_sim_cycles")
    ):
        raise AssertionError(
            f"unexpected {stem} CGRA comparison evidence: {cgra_report} {comparison_report}"
        )


def assert_cmsis_minimum_s8_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    assert_cmsis_minmax_s8_cgra_evidence(
        repo,
        rows,
        sim_evidence,
        case="BasicMathFunctions/arm_minimum_s8.c",
        stem="arm_minimum_s8",
        graph="g_t_arm_minimum_s8_red_0_0",
        intrinsic="llvm.intr.smin",
        expected_output=["i8:2", "i8:-4", "i8:-9"],
    )


def assert_cmsis_maximum_s8_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    assert_cmsis_minmax_s8_cgra_evidence(
        repo,
        rows,
        sim_evidence,
        case="BasicMathFunctions/arm_maximum_s8.c",
        stem="arm_maximum_s8",
        graph="g_t_arm_maximum_s8_red_0_0",
        intrinsic="llvm.intr.smax",
        expected_output=["i8:3", "i8:5", "i8:7"],
    )


def assert_cmsis_softmax_u8_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    case = "SoftmaxFunctions/arm_softmax_u8.c"
    stem = "arm_softmax_u8"
    graph = "g_t_arm_softmax_u8_red_0_0"
    hardware = "shared_quantized_window_adg"
    assert_cmsis_cgra_pass_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-nn",
        case,
        stem,
        expected_hardware=hardware,
    )
    row = one_row(rows, "cmsis-nn", case)
    for key in ("dfg_report", "mapping_artifact", "cgra_report"):
        artifact = sim_evidence / Path(row[key]).name
        if not artifact.is_file():
            raise AssertionError(f"attempt manifest should emit {artifact}")
    comparison_artifact = repo / row["comparison_report"]
    if not comparison_artifact.is_file():
        raise AssertionError(f"rollup should emit {comparison_artifact}")
    expected_memory = {
        "arg45": ["i8:1", "i8:2", "i8:3"],
        "arg46": ["i8:0", "i8:0", "i8:0"],
    }

    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != case
        or dfg_report.get("graph") != graph
        or dfg_report.get("status") != "pass"
        or dfg_report.get("dynamic_work_items") != 3
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 9
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.mux") != 6
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.store") != 3
        or dfg_report.get("operation_fire_counts", {}).get("llvm.intr.ctlz") != 1
        or dfg_report.get("final_outputs") != ["none"]
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected {stem} DFG evidence: {dfg_report}")

    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    expected_mapping_id = (
        "SoftmaxFunctions%2Farm_softmax_u8%2Ec__"
        "g_t_arm_softmax_u8_red_0_0__shared_quantized_window_adg"
    )
    if (
        mapping_artifact.get("kind") != "pnr_mapping"
        or mapping_artifact.get("workload") != case
        or mapping_artifact.get("graph") != graph
        or mapping_artifact.get("hardware") != hardware
        or mapping_artifact.get("mapping_id") != expected_mapping_id
        or mapping_artifact.get("status") != "pass"
        or mapping_artifact.get("placed_records") != 224
        or mapping_artifact.get("unplaced_records") != 0
        or mapping_artifact.get("routed_edges") != 281
        or mapping_artifact.get("unrouted_edges") != 0
        or mapping_artifact.get("config_records") != 7590
        or mapping_artifact.get("resource_pressure") not in (None, [])
    ):
        raise AssertionError(f"unexpected {stem} mapping evidence: {mapping_artifact}")
    placement_counts: dict[str, int] = {}
    for placement in mapping_artifact.get("placements", []):
        if isinstance(placement, dict) and placement.get("resource_kind") == "fabric.op":
            operation = placement.get("operation")
            if isinstance(operation, str):
                placement_counts[operation] = placement_counts.get(operation, 0) + 1
    for operation, expected_count in {
        "arith.cmpi": 40,
        "arith.divsi": 17,
        "arith.muli": 17,
        "arith.select": 36,
        "arith.shli": 5,
        "arith.shrsi": 1,
        "arith.shrui": 2,
        "dataflow.mux": 2,
        "llvm.intr.ctlz": 1,
        "llvm.intr.umax": 1,
        "llvm.sext": 17,
        "llvm.trunc": 18,
        "llvm.zext": 6,
    }.items():
        if placement_counts.get(operation) != expected_count:
            raise AssertionError(
                f"{stem} should place {expected_count} {operation} ops on shared fabric: "
                f"{placement_counts}"
            )
    routes = mapping_artifact.get("routes")
    if not isinstance(routes, list) or len(routes) != 281:
        raise AssertionError(f"{stem} should expose routed edge records: {mapping_artifact}")
    if any(not isinstance(route, dict) or route.get("status") != "routed" for route in routes):
        raise AssertionError(f"{stem} should not contain unrouted route records: {mapping_artifact}")

    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        cgra_report.get("kind") != "cgra_sim_report"
        or cgra_report.get("workload") != case
        or cgra_report.get("hardware") != hardware
        or cgra_report.get("mapping_id") != expected_mapping_id
        or cgra_report.get("status") != "pass"
        or cgra_report.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra_report.get("final_outputs") != ["none"]
        or cgra_report.get("final_memory_state") != expected_memory
        or cgra_report.get("dfg_cycles") != 1198
        or cgra_report.get("hardware_aware_cycles") != 2795
        or cgra_report.get("performance_delta_cycles") != 1597
        or cgra_report.get("route_segments") != 1433
        or cgra_report.get("config_records") != 7590
        or comparison_report.get("kind") != "sim_comparison_report"
        or comparison_report.get("workload") != case
        or comparison_report.get("status") != "pass"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
        or comparison_report.get("performance_comparison_status") != "pass"
        or comparison_report.get("dfg_sim_cycles") != cgra_report.get("dfg_cycles")
        or comparison_report.get("cgra_sim_cycles") != cgra_report.get("hardware_aware_cycles")
        or comparison_report.get("performance_delta_cycles") != cgra_report.get("performance_delta_cycles")
    ):
        raise AssertionError(
            f"unexpected {stem} CGRA comparison evidence: {cgra_report} {comparison_report}"
        )


def assert_cmsis_depthwise_conv_s8_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    case = "ConvolutionFunctions/arm_depthwise_conv_s8.c"
    stem = "arm_depthwise_conv_s8"
    hardware = "shared_quantized_window_adg"
    expected_graphs = [
        "g_t_arm_depthwise_conv_s8_red_0_0",
        "g_t_arm_depthwise_conv_s8_red_1_0",
    ]
    expected_memory = {
        "g_t_arm_depthwise_conv_s8_red_1_0:arg21": ["i32:0", "i32:0", "i32:0", "i32:0"],
        "g_t_arm_depthwise_conv_s8_red_1_0:arg23": ["i8:1", "i8:1", "i8:1", "i8:1"],
        "g_t_arm_depthwise_conv_s8_red_1_0:arg27": ["i32:1", "i32:1", "i32:1", "i32:1"],
        "g_t_arm_depthwise_conv_s8_red_1_0:arg28": ["i32:0", "i32:0", "i32:0", "i32:0"],
        "g_t_arm_depthwise_conv_s8_red_1_0:arg34": ["i8:0", "i8:0", "i8:0", "i8:0"],
        "g_t_arm_depthwise_conv_s8_red_1_0:arg42": ["i8:0", "i8:0", "i8:0", "i8:0"],
    }
    assert_cmsis_cgra_pass_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-nn",
        case,
        stem,
        expected_hardware=hardware,
    )
    row = one_row(rows, "cmsis-nn", case)
    if (
        row["required_slice_count"] != "2"
        or row["graph_ids"].split(",") != expected_graphs
        or Path(row["dfg_report"]).name != f"{stem}.dfg.report.json"
        or Path(row["mapping_artifact"]).name != f"{stem}.mapping.json"
        or Path(row["cgra_report"]).name != f"{stem}.cgra.report.json"
    ):
        raise AssertionError(f"{stem} should expose row-complete aggregate evidence: {row}")
    for artifact_name in (
        f"{stem}.red0.dfg.report.json",
        f"{stem}.red0.mapping.json",
        f"{stem}.red0.cgra.report.json",
        f"{stem}.red1.dfg.report.json",
        f"{stem}.red1.mapping.json",
        f"{stem}.red1.cgra.report.json",
        f"{stem}.dfg.report.json",
        f"{stem}.mapping.json",
        f"{stem}.mapping.csv",
        f"{stem}.cgra.report.json",
    ):
        artifact = sim_evidence / artifact_name
        if not artifact.is_file():
            raise AssertionError(f"{stem} evidence should emit {artifact}")

    red0_dfg = json.loads((sim_evidence / f"{stem}.red0.dfg.report.json").read_text())
    red0_mapping = json.loads((sim_evidence / f"{stem}.red0.mapping.json").read_text())
    red0_cgra = json.loads((sim_evidence / f"{stem}.red0.cgra.report.json").read_text())
    if (
        red0_dfg.get("kind") != "dfg_sim_report"
        or red0_dfg.get("workload") != case
        or red0_dfg.get("graph") != expected_graphs[0]
        or red0_dfg.get("status") != "pass"
        or red0_dfg.get("dynamic_work_items") != 1
        or red0_dfg.get("operation_fire_counts", {}).get("llvm.intr.smax") != 1
        or red0_dfg.get("final_outputs") != ["none", "i32:0"]
        or red0_mapping.get("status") != "pass"
        or red0_mapping.get("hardware") != hardware
        or red0_mapping.get("placed_records") != 163
        or red0_mapping.get("routed_edges") != 167
        or red0_mapping.get("unrouted_edges") != 0
        or red0_mapping.get("unplaced_records") != 0
        or red0_mapping.get("config_records") != 4424
        or red0_cgra.get("status") != "pass"
        or red0_cgra.get("dfg_cycles") != 8
        or red0_cgra.get("hardware_aware_cycles") != 995
        or red0_cgra.get("final_outputs") != ["none", "i32:0"]
    ):
        raise AssertionError(f"unexpected {stem} red0 evidence: {red0_dfg} {red0_mapping} {red0_cgra}")

    red1_dfg = json.loads((sim_evidence / f"{stem}.red1.dfg.report.json").read_text())
    red1_mapping = json.loads((sim_evidence / f"{stem}.red1.mapping.json").read_text())
    red1_cgra = json.loads((sim_evidence / f"{stem}.red1.cgra.report.json").read_text())
    red1_memory = {
        key.split(":", 1)[1]: value for key, value in expected_memory.items()
    }
    if (
        red1_dfg.get("kind") != "dfg_sim_report"
        or red1_dfg.get("workload") != case
        or red1_dfg.get("graph") != expected_graphs[1]
        or red1_dfg.get("status") != "pass"
        or red1_dfg.get("dynamic_work_items") != 1
        or red1_dfg.get("operation_fire_counts", {}).get("arith.divsi") != 4
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.load") != 5
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.mux") != 4
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 1
        or red1_dfg.get("final_outputs") != ["none", "i32:1"]
        or red1_dfg.get("final_memory_state") != red1_memory
        or red1_mapping.get("status") != "pass"
        or red1_mapping.get("hardware") != hardware
        or red1_mapping.get("placed_records") != 86
        or red1_mapping.get("routed_edges") != 93
        or red1_mapping.get("unrouted_edges") != 0
        or red1_mapping.get("unplaced_records") != 0
        or red1_mapping.get("config_records") != 2476
        or red1_cgra.get("status") != "pass"
        or red1_cgra.get("dfg_cycles") != 224
        or red1_cgra.get("hardware_aware_cycles") != 766
        or red1_cgra.get("final_outputs") != ["none", "i32:1"]
        or red1_cgra.get("final_memory_state") != red1_memory
    ):
        raise AssertionError(f"unexpected {stem} red1 evidence: {red1_dfg} {red1_mapping} {red1_cgra}")

    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != case
        or dfg_report.get("graph") != "workload_graph_set"
        or dfg_report.get("aggregation_kind") != "workload_graph_set"
        or dfg_report.get("component_graphs") != expected_graphs
        or dfg_report.get("status") != "pass"
        or dfg_report.get("optimistic_cycles") != 232
        or dfg_report.get("dynamic_work_items") != 2
        or dfg_report.get("operation_fire_counts", {}).get("arith.divsi") != 4
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 5
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.store") != 1
        or dfg_report.get("final_outputs") != ["none", "i32:0", "none", "i32:1"]
        or dfg_report.get("final_memory_state") != expected_memory
        or mapping_artifact.get("kind") != "pnr_mapping"
        or mapping_artifact.get("workload") != case
        or mapping_artifact.get("hardware") != hardware
        or mapping_artifact.get("graph") != "workload_graph_set"
        or mapping_artifact.get("aggregation_kind") != "workload_graph_set"
        or mapping_artifact.get("component_graphs") != expected_graphs
        or mapping_artifact.get("status") != "pass"
        or mapping_artifact.get("placed_records") != 249
        or mapping_artifact.get("routed_edges") != 260
        or mapping_artifact.get("unrouted_edges") != 0
        or mapping_artifact.get("unplaced_records") != 0
        or mapping_artifact.get("config_records") != 6900
        or mapping_artifact.get("route_segments") != 1246
        or cgra_report.get("kind") != "cgra_sim_report"
        or cgra_report.get("workload") != case
        or cgra_report.get("hardware") != hardware
        or cgra_report.get("graph") != "workload_graph_set"
        or cgra_report.get("aggregation_kind") != "workload_graph_set"
        or cgra_report.get("component_graphs") != expected_graphs
        or cgra_report.get("status") != "pass"
        or cgra_report.get("fidelity_level") != "mapping_constraint_estimate"
        or cgra_report.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
        or cgra_report.get("dfg_cycles") != 232
        or cgra_report.get("hardware_aware_cycles") != 1761
        or cgra_report.get("performance_delta_cycles") != 1529
        or cgra_report.get("route_segments") != 1246
        or cgra_report.get("config_records") != 6900
        or cgra_report.get("final_outputs") != ["none", "i32:0", "none", "i32:1"]
        or cgra_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(
            f"unexpected {stem} aggregate evidence: {dfg_report} {mapping_artifact} {cgra_report}"
        )
    comparison = json.loads((repo / row["comparison_report"]).read_text())
    if (
        comparison.get("kind") != "sim_comparison_report"
        or comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 232
        or comparison.get("cgra_sim_cycles") != 1761
        or comparison.get("performance_delta_cycles") != 1529
    ):
        raise AssertionError(f"{stem} comparison should pass: {comparison}")


def assert_cmsis_max_pool_s8_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    case = "PoolingFunctions/arm_max_pool_s8.c"
    stem = "arm_max_pool_s8"
    graph = "g_t_arm_max_pool_s8_red_0_0"
    hardware = "shared_quantized_window_adg"
    expected_memory = {"arg38": ["i8:0", "i8:0", "i8:0", "i8:0"]}
    assert_cmsis_cgra_pass_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-nn",
        case,
        stem,
        expected_hardware=hardware,
    )
    row = one_row(rows, "cmsis-nn", case)
    for artifact_name in (
        f"{stem}.dfg.report.json",
        f"{stem}.mapping.json",
        f"{stem}.cgra.report.json",
    ):
        artifact = sim_evidence / artifact_name
        if not artifact.is_file():
            raise AssertionError(f"{stem} evidence should emit {artifact}")

    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != case
        or dfg_report.get("graph") != graph
        or dfg_report.get("status") != "pass"
        or dfg_report.get("dynamic_work_items") != 1
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 1
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.store") != 1
        or dfg_report.get("operation_fire_counts", {}).get("scf.if") != 3
        or "llvm.intr.memcpy" in dfg_report.get("operation_fire_counts", {})
        or dfg_report.get("final_outputs") != ["none"]
        or dfg_report.get("final_memory_state") != expected_memory
        or dfg_report.get("optimistic_cycles") != 35
    ):
        raise AssertionError(f"unexpected {stem} DFG evidence: {dfg_report}")

    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    mapping_text = json.dumps(mapping_artifact, sort_keys=True)
    routes = mapping_artifact.get("routes")
    if (
        mapping_artifact.get("kind") != "pnr_mapping"
        or mapping_artifact.get("workload") != case
        or mapping_artifact.get("graph") != graph
        or mapping_artifact.get("hardware") != hardware
        or mapping_artifact.get("status") != "pass"
        or mapping_artifact.get("routed_edges") != 121
        or mapping_artifact.get("unrouted_edges") != 0
        or mapping_artifact.get("unplaced_records") != 0
        or mapping_artifact.get("placed_records") != 120
        or mapping_artifact.get("config_records") != 3164
        or mapping_artifact.get("resource_pressure") not in (None, [])
        or mapping_artifact.get("unrouted_edge_details") not in (None, [])
        or not isinstance(routes, list)
        or len(routes) != 121
        or "missing hardware resource for software op dataflow.load" in mapping_text
        or "llvm.intr.memcpy" in mapping_text
        or "fabric.mem.copy" in mapping_text
        or "memory_copy_binding" in mapping_text
    ):
        raise AssertionError(f"unexpected {stem} mapping pass evidence: {mapping_artifact}")
    if not any(
        isinstance(route, dict)
        and any(
            isinstance(segment, dict)
            and segment.get("segment_kind") == "resource_edge"
            and str(segment.get("source_endpoint", "")).startswith(
                f"{hardware}::fabric.pe#"
            )
            and f"{hardware}::fabric.switch#" in str(segment.get("sink_endpoint", ""))
            for segment in route.get("segments", [])
        )
        for route in routes
    ):
        raise AssertionError(f"{stem} should expose real routed fabric endpoints: {mapping_artifact}")
    if not any(
        isinstance(record, dict)
        and record.get("register") == "operation"
        and record.get("value") == "llvm.intr.smax"
        and str(record.get("target", "")).startswith(f"{hardware}::fabric.op#")
        for record in mapping_artifact.get("config_bitstream", [])
    ):
        raise AssertionError(f"{stem} should configure a real smax fabric op: {mapping_artifact}")

    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    if (
        cgra_report.get("kind") != "cgra_sim_report"
        or cgra_report.get("workload") != case
        or cgra_report.get("hardware") != hardware
        or cgra_report.get("status") != "pass"
        or cgra_report.get("fidelity_level") != "mapping_constraint_estimate"
        or cgra_report.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra_report.get("dfg_cycles") != 35
        or cgra_report.get("hardware_aware_cycles") != 715
        or cgra_report.get("performance_delta_cycles") != 680
        or cgra_report.get("route_segments") != 559
        or cgra_report.get("config_records") != 3164
        or cgra_report.get("final_outputs") != ["none"]
        or cgra_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected {stem} CGRA evidence: {cgra_report}")

    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        comparison_report.get("kind") != "sim_comparison_report"
        or comparison_report.get("workload") != case
        or comparison_report.get("status") != "pass"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
        or comparison_report.get("performance_comparison_status") != "pass"
        or comparison_report.get("dfg_sim_cycles") != 35
        or comparison_report.get("cgra_sim_cycles") != 715
        or comparison_report.get("performance_delta_cycles") != 680
    ):
        raise AssertionError(f"unexpected {stem} comparison evidence: {comparison_report}")


def assert_cmsis_max_pool_mapping_blocker_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    case = "PoolingFunctions/arm_max_pool_s8.c"
    stem = "arm_max_pool_s8"
    graph = "g_t_arm_max_pool_s8_red_0_0"
    diagnostic = "missing hardware resource for software op dataflow.load"
    assert_cmsis_mapping_blocker_row(
        repo,
        rows,
        "cmsis-nn",
        case,
        diagnostic_substring=diagnostic,
    )
    row = one_row(rows, "cmsis-nn", case)
    for key in ("dfg_report", "mapping_artifact", "cgra_report"):
        artifact = sim_evidence / Path(row[key]).name
        if not artifact.is_file():
            raise AssertionError(f"attempt manifest should emit {artifact}")

    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != case
        or dfg_report.get("graph") != graph
        or dfg_report.get("status") != "pass"
        or dfg_report.get("dynamic_work_items") != 1
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 1
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.store") != 1
        or dfg_report.get("operation_fire_counts", {}).get("scf.if") != 3
        or "llvm.intr.memcpy" in dfg_report.get("operation_fire_counts", {})
        or dfg_report.get("final_outputs") != ["none"]
        or dfg_report.get("final_memory_state") != {
            "arg38": ["i8:0", "i8:0", "i8:0", "i8:0"]
        }
    ):
        raise AssertionError(f"unexpected {stem} DFG evidence: {dfg_report}")
    if (
        mapping_artifact.get("kind") != "pnr_mapping"
        or mapping_artifact.get("workload") != case
        or mapping_artifact.get("graph") != graph
        or mapping_artifact.get("hardware") != "shared_reduction_adg"
        or mapping_artifact.get("status") != "fail"
        or mapping_artifact.get("unrouted_edges") != 8
        or mapping_artifact.get("unplaced_records") != 85
        or diagnostic not in " ".join(mapping_artifact.get("diagnostics", []))
        or "llvm.intr.memcpy" in json.dumps(mapping_artifact, sort_keys=True)
        or "fabric.mem.copy" in json.dumps(mapping_artifact, sort_keys=True)
        or "memory_copy_binding" in json.dumps(mapping_artifact, sort_keys=True)
    ):
        raise AssertionError(f"unexpected {stem} mapping blocker evidence: {mapping_artifact}")
    placements = mapping_artifact.get("placements", [])
    if not any(
        placement.get("operation") == "dataflow.load"
        and placement.get("resource_kind") == "fabric.mem.load"
        for placement in placements
    ):
        raise AssertionError(f"{stem} mapping should place a real dataflow.load: {mapping_artifact}")
    if not any(
        placement.get("operation") == "dataflow.store"
        and placement.get("resource_kind") == "fabric.mem.store"
        for placement in placements
    ):
        raise AssertionError(f"{stem} mapping should place a real dataflow.store: {mapping_artifact}")
    assert_resource_pressure_record(
        mapping_artifact,
        resource_kind="fabric.mem.load",
        operation="dataflow.load",
        required=7,
        available=6,
        placed=6,
        missing=1,
        label=stem,
    )
    assert_resource_pressure_record(
        mapping_artifact,
        resource_kind="fabric.mem.store",
        operation="dataflow.store",
        required=5,
        available=2,
        placed=2,
        missing=3,
        label=stem,
    )
    if (
        cgra_report.get("kind") != "cgra_sim_report"
        or cgra_report.get("workload") != case
        or cgra_report.get("hardware") != "shared_reduction_adg"
        or cgra_report.get("status") != "blocked"
        or diagnostic not in " ".join(cgra_report.get("diagnostics", []))
    ):
        raise AssertionError(f"unexpected {stem} CGRA blocker evidence: {cgra_report}")
    if (
        comparison_report.get("kind") != "sim_comparison_report"
        or comparison_report.get("workload") != case
        or comparison_report.get("status") != "blocked"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
        or comparison_report.get("performance_comparison_status") != "blocked"
    ):
        raise AssertionError(f"unexpected {stem} comparison evidence: {comparison_report}")


def assert_cmsis_relu_q7_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    row = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu_q7.c")
    expected_graphs = {
        "g_t_arm_relu_q7_red_0_0",
        "g_t_arm_relu_q7_red_1_0",
    }
    if set(row["graph_ids"].split(",")) != expected_graphs:
        raise AssertionError(f"arm_relu_q7 row should keep both component graph ids: {row}")
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["blocking_prerequisite"] != ""
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["hardware_system"] != "shared_reduction_adg"
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
        or Path(row["dfg_report"]).name != "arm_relu_q7.dfg.report.json"
        or Path(row["mapping_artifact"]).name != "arm_relu_q7.mapping.json"
        or Path(row["cgra_report"]).name != "arm_relu_q7.cgra.report.json"
        or Path(row["comparison_report"]).name != "arm_relu_q7.c.sim-comparison-report.json"
        or "DFG-sim, mapping, CGRA-sim, and simulation comparison evidence passed" not in row["diagnostic"]
    ):
        raise AssertionError(f"arm_relu_q7 should expose aggregate CGRA-sim pass evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)

    for artifact_name in (
        "arm_relu_q7.red0.dfg.report.json",
        "arm_relu_q7.red0.mapping.json",
        "arm_relu_q7.red0.cgra.report.json",
        "arm_relu_q7.red1.dfg.report.json",
        "arm_relu_q7.red1.mapping.json",
        "arm_relu_q7.red1.cgra.report.json",
        "arm_relu_q7.dfg.report.json",
        "arm_relu_q7.mapping.json",
        "arm_relu_q7.cgra.report.json",
        "arm_relu_q7.mapping.csv",
    ):
        artifact = sim_evidence / artifact_name
        if not artifact.is_file():
            raise AssertionError(f"arm_relu_q7 evidence should emit {artifact}")

    red1_memory = {"arg5": ["i8:0", "i8:2", "i8:0"]}
    red1_dfg = json.loads((sim_evidence / "arm_relu_q7.red1.dfg.report.json").read_text())
    if (
        red1_dfg.get("kind") != "dfg_sim_report"
        or red1_dfg.get("workload") != "ActivationFunctions/arm_relu_q7.c"
        or red1_dfg.get("graph") != "g_t_arm_relu_q7_red_1_0"
        or red1_dfg.get("status") != "pass"
        or red1_dfg.get("optimistic_cycles") != 101
        or red1_dfg.get("dynamic_work_items") != 3
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.load") != 3
        or red1_dfg.get("operation_fire_counts", {}).get("arith.cmpi") != 3
        or red1_dfg.get("operation_fire_counts", {}).get("arith.select") != 3
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 3
        or red1_dfg.get("final_outputs") != ["none"]
        or red1_dfg.get("final_memory_state") != red1_memory
    ):
        raise AssertionError(f"unexpected arm_relu_q7 residual DFG report: {red1_dfg}")

    red1_mapping = json.loads((sim_evidence / "arm_relu_q7.red1.mapping.json").read_text())
    expected_red1_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "g_t_arm_relu_q7_red_1_0",
        "placed_records": 13,
        "routed_edges": 18,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 402,
        "status": "pass",
    }
    for key, value in expected_red1_mapping.items():
        if red1_mapping.get(key) != value:
            raise AssertionError(f"arm_relu_q7 red1 mapping {key}={red1_mapping.get(key)!r}, expected {value!r}")
    routes = red1_mapping.get("routes", [])
    if len(routes) != 18:
        raise AssertionError(f"arm_relu_q7 red1 mapping should expose every routed edge: {red1_mapping}")
    routes_by_edge = {route.get("edge_ref"): route for route in routes if isinstance(route, dict)}
    required_routes = {
        "dataflow.invariant#0.result0->arith.select#0.operand1": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.operand1",
        ),
        "dataflow.constant#2.result0->dataflow.store#0.operand1": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            "shared_reduction_adg::mem.store#0.operand0",
        ),
        "arith.select#0.result0->dataflow.store#0.operand2": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            "shared_reduction_adg::mem.store#0.operand1",
        ),
    }
    for edge_ref, (source_endpoint, sink_endpoint) in required_routes.items():
        route = routes_by_edge.get(edge_ref)
        if route is None:
            raise AssertionError(f"arm_relu_q7 red1 mapping missed route {edge_ref}: {red1_mapping}")
        assert_routed_endpoint_shape(
            route,
            edge_ref,
            source_endpoint=source_endpoint,
            sink_endpoint=sink_endpoint,
            label="arm_relu_q7 red1",
        )
    sync_route = routes_by_edge.get("dataflow.store#0.result0->dataflow.sync#0.operand1")
    if sync_route is None:
        raise AssertionError(f"arm_relu_q7 red1 mapping missed store-to-sync route: {red1_mapping}")
    sync_segments = sync_route.get("segments", [])
    if (
        not sync_segments
        or not isinstance(sync_segments[0], dict)
        or sync_segments[0].get("source_endpoint") != "shared_reduction_adg::mem.store#0.result0"
        or not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in sync_segments)
    ):
        raise AssertionError(f"arm_relu_q7 red1 store-to-sync route should leave the store through Fabric paths: {sync_route}")

    red1_cgra = json.loads((sim_evidence / "arm_relu_q7.red1.cgra.report.json").read_text())
    if (
        red1_cgra.get("status") != "pass"
        or red1_cgra.get("dfg_cycles") != 101
        or red1_cgra.get("hardware_aware_cycles") != 200
        or red1_cgra.get("fidelity_level") != "mapping_constraint_estimate"
        or red1_cgra.get("config_records") != 402
        or red1_cgra.get("route_segments") != 72
        or red1_cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
        or red1_cgra.get("final_outputs") != ["none"]
        or red1_cgra.get("final_memory_state") != red1_memory
    ):
        raise AssertionError(f"unexpected arm_relu_q7 red1 CGRA report: {red1_cgra}")

    aggregate_memory = {
        "g_t_arm_relu_q7_red_0_0:arg8": ["i32:0", "i32:2130706433"],
        "g_t_arm_relu_q7_red_1_0:arg5": ["i8:0", "i8:2", "i8:0"],
    }
    expected_graph_list = sorted(expected_graphs)
    aggregate_dfg = json.loads((sim_evidence / "arm_relu_q7.dfg.report.json").read_text())
    if (
        aggregate_dfg.get("kind") != "dfg_sim_report"
        or aggregate_dfg.get("workload") != "ActivationFunctions/arm_relu_q7.c"
        or aggregate_dfg.get("graph") != "workload_graph_set"
        or aggregate_dfg.get("aggregation_kind") != "workload_graph_set"
        or aggregate_dfg.get("component_graphs") != expected_graph_list
        or aggregate_dfg.get("status") != "pass"
        or aggregate_dfg.get("optimistic_cycles") != 188
        or aggregate_dfg.get("dynamic_work_items") != 5
        or aggregate_dfg.get("operation_fire_counts", {}).get("dataflow.load") != 5
        or aggregate_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 5
        or aggregate_dfg.get("final_outputs") != ["none", "none"]
        or aggregate_dfg.get("final_memory_state") != aggregate_memory
    ):
        raise AssertionError(f"unexpected arm_relu_q7 aggregate DFG report: {aggregate_dfg}")

    aggregate_mapping = json.loads((sim_evidence / "arm_relu_q7.mapping.json").read_text())
    expected_aggregate_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "workload_graph_set",
        "aggregation_kind": "workload_graph_set",
        "placed_records": 31,
        "routed_edges": 44,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 991,
        "route_segments": 180,
        "status": "pass",
    }
    for key, value in expected_aggregate_mapping.items():
        if aggregate_mapping.get(key) != value:
            raise AssertionError(f"arm_relu_q7 aggregate mapping {key}={aggregate_mapping.get(key)!r}, expected {value!r}")
    if aggregate_mapping.get("component_graphs") != expected_graph_list or len(aggregate_mapping.get("routes", [])) != 44:
        raise AssertionError(f"arm_relu_q7 aggregate mapping should preserve component graph routes: {aggregate_mapping}")

    aggregate_cgra = json.loads((sim_evidence / "arm_relu_q7.cgra.report.json").read_text())
    if (
        aggregate_cgra.get("status") != "pass"
        or aggregate_cgra.get("component_graphs") != expected_graph_list
        or aggregate_cgra.get("aggregation_kind") != "workload_graph_set"
        or aggregate_cgra.get("fidelity_level") != "mapping_constraint_estimate"
        or aggregate_cgra.get("dfg_cycles") != 188
        or aggregate_cgra.get("hardware_aware_cycles") != 425
        or aggregate_cgra.get("performance_delta_cycles") != 237
        or aggregate_cgra.get("routed_edges") != 44
        or aggregate_cgra.get("config_records") != 991
        or aggregate_cgra.get("route_segments") != 180
        or aggregate_cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
        or aggregate_cgra.get("final_outputs") != ["none", "none"]
        or aggregate_cgra.get("final_memory_state") != aggregate_memory
    ):
        raise AssertionError(f"unexpected arm_relu_q7 aggregate CGRA report: {aggregate_cgra}")
    comparison = json.loads((repo / row["comparison_report"]).read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
    ):
        raise AssertionError(f"arm_relu_q7 comparison should pass: {comparison}")


def assert_cmsis_relu6_s8_cgra_evidence(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    case = "ActivationFunctions/arm_relu6_s8.c"
    stem = "arm_relu6_s8"
    graph = "g_t_arm_relu6_s8_0_0"
    hardware = "shared_quantized_window_adg"
    expected_memory = {"arg1": ["i8:0", "i8:2", "i8:6"]}
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-nn", case, stem, expected_hardware=hardware)
    row = one_row(rows, "cmsis-nn", case)
    if row["graph_ids"] != graph or row["required_slice_count"] != "1":
        raise AssertionError(f"arm_relu6_s8 row should name its single dataflow graph: {row}")
    if Path(row["comparison_report"]).name != "arm_relu6_s8.c.sim-comparison-report.json":
        raise AssertionError(f"arm_relu6_s8 row should reference its comparison artifact: {row}")
    if not (sim_evidence / "arm_relu6_s8.mapping.csv").is_file():
        raise AssertionError(f"arm_relu6_s8 evidence should emit mapping CSV under {sim_evidence}")

    dfg_report = json.loads((repo / row["dfg_report"]).read_text())
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != case
        or dfg_report.get("graph") != graph
        or dfg_report.get("status") != "pass"
        or dfg_report.get("optimistic_cycles") != 16
        or dfg_report.get("dynamic_work_items") != 1
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 1
        or dfg_report.get("operation_fire_counts", {}).get("llvm.intr.smax") != 1
        or dfg_report.get("operation_fire_counts", {}).get("llvm.intr.umin") != 1
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.store") != 1
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.sync") != 1
        or dfg_report.get("final_outputs") != ["none"]
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected arm_relu6_s8 DFG evidence: {dfg_report}")

    mapping_artifact = json.loads((repo / row["mapping_artifact"]).read_text())
    expected_mapping = {
        "kind": "pnr_mapping",
        "workload": case,
        "graph": graph,
        "hardware": hardware,
        "status": "pass",
        "placed_records": 5,
        "routed_edges": 5,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 112,
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_relu6_s8 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if not isinstance(routes, list) or len(routes) != 5:
        raise AssertionError(f"arm_relu6_s8 mapping should expose every routed edge: {mapping_artifact}")
    routes_by_edge = {route.get("edge_ref"): route for route in routes if isinstance(route, dict)}
    required_routes = {
        "dataflow.load#0.result0->llvm.intr.smax#0.operand0": (
            "shared_quantized_window_adg::mem.load#0.result0",
            r"re:shared_quantized_window_adg::fabric\.op#[0-9]+\.operand0",
        ),
        "llvm.intr.smax#0.result0->llvm.intr.umin#0.operand0": (
            r"re:shared_quantized_window_adg::fabric\.op#[0-9]+\.result0",
            r"re:shared_quantized_window_adg::fabric\.op#[0-9]+\.operand0",
        ),
        "llvm.intr.umin#0.result0->dataflow.store#0.operand2": (
            r"re:shared_quantized_window_adg::fabric\.op#[0-9]+\.result0",
            "shared_quantized_window_adg::mem.store#0.operand1",
        ),
    }
    for edge_ref, (source_endpoint, sink_endpoint) in required_routes.items():
        route = routes_by_edge.get(edge_ref)
        if route is None:
            raise AssertionError(f"arm_relu6_s8 mapping missed route {edge_ref}: {mapping_artifact}")
        assert_routed_endpoint_shape(
            route,
            edge_ref,
            source_endpoint=source_endpoint,
            sink_endpoint=sink_endpoint,
            label="arm_relu6_s8",
        )

    cgra_report = json.loads((repo / row["cgra_report"]).read_text())
    if (
        cgra_report.get("kind") != "cgra_sim_report"
        or cgra_report.get("workload") != case
        or cgra_report.get("hardware") != hardware
        or cgra_report.get("status") != "pass"
        or cgra_report.get("fidelity_level") != "mapping_constraint_estimate"
        or cgra_report.get("dfg_cycles") != 16
        or cgra_report.get("hardware_aware_cycles") != 51
        or cgra_report.get("performance_delta_cycles") != 35
        or cgra_report.get("placed_records") != 5
        or cgra_report.get("routed_edges") != 5
        or cgra_report.get("config_records") != 112
        or cgra_report.get("route_segments") != 19
        or cgra_report.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra_report.get("final_outputs") != ["none"]
        or cgra_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected arm_relu6_s8 CGRA evidence: {cgra_report}")

    comparison_report = json.loads((repo / row["comparison_report"]).read_text())
    if (
        comparison_report.get("kind") != "sim_comparison_report"
        or comparison_report.get("workload") != case
        or comparison_report.get("status") != "pass"
        or comparison_report.get("functional_comparison_status") != "pass"
        or comparison_report.get("memory_comparison_status") != "pass"
        or comparison_report.get("performance_comparison_status") != "pass"
        or comparison_report.get("dfg_sim_cycles") != 16
        or comparison_report.get("cgra_sim_cycles") != 51
    ):
        raise AssertionError(f"unexpected arm_relu6_s8 comparison evidence: {comparison_report}")


def assert_cgra_status_audit_rejects_bad_relu_q7_mapping(
    repo: Path,
    out_dir: Path,
    legacy_root: Path,
) -> None:
    mapping = out_dir / "cmsis-sim-evidence" / "arm_relu_q7.mapping.json"
    original = mapping.read_text()
    data = json.loads(original)
    data["status"] = "fail"
    try:
        mapping.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        result = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(out_dir / "cgra-status-summary.csv"),
                "--json-input",
                str(out_dir / "cgra-status-summary.json"),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
    finally:
        mapping.write_text(original)
    combined = result.stdout + result.stderr
    if "referenced mapping_artifact JSON status is not pass" not in combined:
        raise AssertionError(f"CGRA status audit should reject stale arm_relu_q7 mapping evidence: {combined}")

    try:
        mapping.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        result = run(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "generic-audit-bad-relu-q7-mapping.json"),
                str(out_dir / "cgra-status-summary.csv"),
            ],
            expect_success=False,
        )
    finally:
        mapping.write_text(original)
    combined = result.stdout + result.stderr
    audit_data = json.loads((out_dir / "generic-audit-bad-relu-q7-mapping.json").read_text())
    audit_diagnostics = "\n".join(str(item) for item in audit_data.get("diagnostics", []))
    if "referenced mapping_artifact JSON status is not pass" not in combined + audit_diagnostics:
        raise AssertionError(f"generic audit should reject stale arm_relu_q7 mapping evidence: {combined} {audit_data}")


def assert_cgra_status_audit_rejects_bad_aggregate_graphs(
    repo: Path,
    out_dir: Path,
    legacy_root: Path,
) -> None:
    for report_name in ("arm_var_f32.dfg.report.json", "arm_var_f32.cgra.report.json"):
        report = out_dir / "cmsis-sim-evidence" / report_name
        original = report.read_text()
        data = json.loads(original)
        data["component_graphs"] = ["g_t_arm_var_f32_red_0_0"]
        try:
            report.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            result = run(
                repo,
                [
                    "bash",
                    "test/e2e/run_cgra_status_audit.sh",
                    "--input",
                    str(out_dir / "cgra-status-summary.csv"),
                    "--json-input",
                    str(out_dir / "cgra-status-summary.json"),
                    "--legacy-loombench-root",
                    str(legacy_root),
                ],
                expect_success=False,
            )
        finally:
            report.write_text(original)
        combined = result.stdout + result.stderr
        if "aggregate component_graphs do not exactly match row graph_ids" not in combined:
            raise AssertionError(f"CGRA status audit should reject stale aggregate graph coverage: {combined}")

        data = json.loads(original)
        data.pop("aggregation_kind", None)
        try:
            report.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            result = run(
                repo,
                [
                    "bash",
                    "test/e2e/run_cgra_status_audit.sh",
                    "--input",
                    str(out_dir / "cgra-status-summary.csv"),
                    "--json-input",
                    str(out_dir / "cgra-status-summary.json"),
                    "--legacy-loombench-root",
                    str(legacy_root),
                ],
                expect_success=False,
            )
        finally:
            report.write_text(original)
        combined = result.stdout + result.stderr
        if "aggregate lacks workload_graph_set aggregation_kind" not in combined:
            raise AssertionError(f"CGRA status audit should reject pseudo aggregate evidence: {combined}")


def assert_generic_artifact_audit_rejects_bad_aggregate_graphs(repo: Path, out_dir: Path) -> None:
    report = out_dir / "cmsis-sim-evidence" / "arm_var_f32.cgra.report.json"
    original = report.read_text()
    data = json.loads(original)
    data["component_graphs"] = ["g_t_arm_var_f32_red_0_0"]
    try:
        report.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        result = run(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "generic-audit-bad-aggregate.json"),
                str(out_dir / "cgra-status-summary.csv"),
            ],
            expect_success=False,
        )
    finally:
        report.write_text(original)
    combined = result.stdout + result.stderr
    audit_data = json.loads((out_dir / "generic-audit-bad-aggregate.json").read_text())
    audit_diagnostics = "\n".join(str(item) for item in audit_data.get("diagnostics", []))
    if "aggregate component_graphs do not exactly match row graph_ids" not in combined + audit_diagnostics:
        raise AssertionError(f"generic audit should reject stale aggregate graph coverage: {combined} {audit_data}")


def assert_cmsis_add_q15_shared_adg_evidence(sim_evidence: Path) -> None:
    mapping_artifact = json.loads((sim_evidence / "arm_add_q15.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "placed_records": 14,
        "routed_edges": 19,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 430,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_add_q15 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 19:
        raise AssertionError(f"arm_add_q15 mapping should expose every routed edge: {mapping_artifact}")
    required_edges = {
        "dataflow.load#0.result0->llvm.sext#0.operand0",
        "dataflow.load#1.result0->llvm.sext#1.operand0",
        "llvm.sext#0.result0->llvm.arm.qadd16#0.operand0",
        "llvm.sext#1.result0->llvm.arm.qadd16#0.operand1",
        "llvm.arm.qadd16#0.result0->llvm.trunc#0.operand0",
        "llvm.trunc#0.result0->dataflow.store#0.operand2",
    }
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_add_q15 mapping missed q15 datapath route evidence: {mapping_artifact}")
    for route in routes:
        if route.get("edge_ref") not in required_edges:
            continue
        segments = route.get("segments", [])
        if not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in segments):
            raise AssertionError(f"arm_add_q15 route should traverse Fabric paths: {route}")

    cgra_report = json.loads((sim_evidence / "arm_add_q15.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "hardware_aware_cycles": 239,
        "performance_delta_cycles": 111,
        "width_adapter_latency_cycles": 1,
        "route_segments": 77,
        "config_records": 430,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_add_q15 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")


def assert_cmsis_offset_shared_adg_evidence(sim_evidence: Path) -> None:
    mapping_artifact = json.loads((sim_evidence / "arm_offset_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "placed_records": 11,
        "routed_edges": 15,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_offset_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 15:
        raise AssertionError(f"arm_offset_f32 mapping should expose every routed edge: {mapping_artifact}")
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    required_edges = {
        "dataflow.carry#0.result0->arith.addi#0.operand0",
        "arith.addi#0.result0->dataflow.carry#0.operand2",
        "dataflow.carry#0.result0->dataflow.load#0.operand1",
        "dataflow.carry#0.result0->dataflow.store#0.operand1",
    }
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_offset_f32 mapping missed explicit index-carry route evidence: {mapping_artifact}")
    if any("llvm.getelementptr" in str(edge) for edge in actual_edges):
        raise AssertionError(f"arm_offset_f32 mapping must not hide GEP as a routed edge: {mapping_artifact}")
    config_records = mapping_artifact.get("config_records")
    if not isinstance(config_records, int) or config_records <= 0:
        raise AssertionError(f"arm_offset_f32 mapping should emit non-empty configuration evidence: {mapping_artifact}")

    cgra_report = json.loads((sim_evidence / "arm_offset_f32.cgra.report.json").read_text())
    if cgra_report.get("config_records") != config_records or cgra_report.get("route_segments", 0) <= 0:
        raise AssertionError(f"arm_offset_f32 CGRA report should consume mapping evidence: {cgra_report}")


def assert_cmsis_fill_shared_adg_evidence(sim_evidence: Path) -> None:
    mapping_artifact = json.loads((sim_evidence / "arm_fill_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "placed_records": 9,
        "routed_edges": 11,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 246,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_fill_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 11:
        raise AssertionError(f"arm_fill_f32 mapping should expose every routed edge: {mapping_artifact}")
    required_edges = {
        "dataflow.carry#0.result0->dataflow.store#0.operand1",
        "dataflow.invariant#0.result0->dataflow.store#0.operand2",
        "dataflow.store#0.result0->dataflow.sync#0.operand0",
    }
    actual_edges = {
        route.get("edge_ref")
        for route in routes
        if isinstance(route, dict)
    }
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_fill_f32 mapping missed store route evidence: {mapping_artifact}")
    multi_hop_routes = [
        route
        for route in routes
        if route.get("edge_ref") in required_edges
        and any(
            segment.get("segment_kind") == "module_path"
            for segment in route.get("segments", [])
            if isinstance(segment, dict)
        )
    ]
    if len(multi_hop_routes) != len(required_edges):
        raise AssertionError(f"arm_fill_f32 mapping should route store edges through Fabric paths: {mapping_artifact}")

    cgra_report = json.loads((sim_evidence / "arm_fill_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "hardware_aware_cycles": 135,
        "performance_delta_cycles": 60,
        "route_segments": 43,
        "config_records": 246,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_fill_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")


def assert_cmsis_mean_shared_adg_evidence(sim_evidence: Path) -> None:
    mapping_artifact = json.loads((sim_evidence / "arm_mean_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "placed_records": 10,
        "routed_edges": 13,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 296,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_mean_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 13:
        raise AssertionError(f"arm_mean_f32 mapping should expose every routed edge: {mapping_artifact}")
    required_edges = {
        "arith.addi#0.result0->dataflow.carry#1.operand2",
        "dataflow.carry#1.result0->arith.addi#0.operand0",
        "dataflow.carry#1.result0->dataflow.load#0.operand1",
        "dataflow.constant#0.result0->dataflow.carry#1.operand1",
    }
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_mean_f32 mapping missed index-carry route evidence: {mapping_artifact}")
    expected_endpoints = {
        "arith.addi#0.result0->dataflow.carry#1.operand2": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.operand2",
        ),
        "dataflow.carry#1.result0->arith.addi#0.operand0": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.operand0",
        ),
        "dataflow.carry#1.result0->dataflow.load#0.operand1": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            "shared_reduction_adg::mem.load#0.operand0",
        ),
        "dataflow.constant#0.result0->dataflow.carry#1.operand1": (
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.result0",
            r"re:shared_reduction_adg::fabric\.op#[0-9]+\.operand1",
        ),
    }
    routes_by_edge = {route.get("edge_ref"): route for route in routes if isinstance(route, dict)}
    for edge_ref, (source_endpoint, sink_endpoint) in expected_endpoints.items():
        route = routes_by_edge.get(edge_ref)
        if route is None:
            raise AssertionError(f"arm_mean_f32 mapping missed route {edge_ref}: {mapping_artifact}")
        assert_routed_endpoint_shape(
            route,
            edge_ref,
            source_endpoint=source_endpoint,
            sink_endpoint=sink_endpoint,
            label="arm_mean_f32 index-carry",
        )

    cgra_report = json.loads((sim_evidence / "arm_mean_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "hardware_aware_cycles": 146,
        "performance_delta_cycles": 68,
        "route_segments": 53,
        "config_records": 296,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_mean_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")
    if cgra_report.get("final_outputs") != ["none", "f32:3.750000"]:
        raise AssertionError(f"arm_mean_f32 CGRA report should carry final reduction output: {cgra_report}")


def assert_cmsis_max_shared_adg_evidence(sim_evidence: Path) -> None:
    dfg_report = json.loads((sim_evidence / "arm_max_f32.dfg.report.json").read_text())
    expected_outputs = ["none", "i32:3", "f32:4.250000"]
    expected_memory = {
        "arg7": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
    }
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != "StatisticsFunctions/arm_max_f32.c"
        or dfg_report.get("graph") != "g_t_arm_max_f32_red_0_0"
        or dfg_report.get("status") != "pass"
        or dfg_report.get("optimistic_cycles") != 105
        or dfg_report.get("dynamic_work_items") != 3
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 3
        or dfg_report.get("operation_fire_counts", {}).get("arith.cmpf") != 3
        or dfg_report.get("final_outputs") != expected_outputs
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected arm_max_f32 DFG report: {dfg_report}")

    mapping_artifact = json.loads((sim_evidence / "arm_max_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "g_t_arm_max_f32_red_0_0",
        "placed_records": 18,
        "routed_edges": 28,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 653,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_max_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 28:
        raise AssertionError(f"arm_max_f32 mapping should expose every routed edge: {mapping_artifact}")
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    required_edges = {
        "dataflow.invariant#2.result0->arith.addi#1.operand1",
        "arith.addi#1.result0->dataflow.load#0.operand1",
        "dataflow.load#0.result0->arith.cmpf#0.operand1",
        "arith.cmpf#0.result0->arith.select#0.operand0",
        "arith.select#0.result0->dataflow.carry#1.operand2",
    }
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_max_f32 mapping missed max-reduction route evidence: {mapping_artifact}")
    for route in routes:
        if route.get("edge_ref") not in required_edges:
            continue
        segments = route.get("segments", [])
        if not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in segments):
            raise AssertionError(f"arm_max_f32 route should traverse Fabric paths: {route}")

    cgra_report = json.loads((sim_evidence / "arm_max_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "dfg_cycles": 105,
        "hardware_aware_cycles": 247,
        "performance_delta_cycles": 142,
        "route_segments": 122,
        "config_records": 653,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_max_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")
    if cgra_report.get("final_outputs") != expected_outputs or cgra_report.get("final_memory_state") != expected_memory:
        raise AssertionError(f"arm_max_f32 CGRA report should carry final max state: {cgra_report}")


def assert_cmsis_biquad_shared_adg_evidence(sim_evidence: Path) -> None:
    dfg_report = json.loads((sim_evidence / "arm_biquad_cascade_df1_f32.dfg.report.json").read_text())
    expected_outputs = ["none", "f32:0.321289", "f32:-0.385681", "f32:-3.500000", "f32:4.250000"]
    expected_memory = {
        "arg9": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
        "arg14": ["f32:0.250000", "f32:1.015625", "f32:0.321289", "f32:-0.385681"],
    }
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != "FilteringFunctions/arm_biquad_cascade_df1_f32.c"
        or dfg_report.get("graph") != "g_t_arm_biquad_cascade_df1_f32_red_0_0"
        or dfg_report.get("status") != "pass"
        or dfg_report.get("optimistic_cycles") != 291
        or dfg_report.get("dynamic_work_items") != 4
        or dfg_report.get("final_outputs") != expected_outputs
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected arm_biquad_cascade_df1_f32 DFG report: {dfg_report}")

    mapping_artifact = json.loads((sim_evidence / "arm_biquad_cascade_df1_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "g_t_arm_biquad_cascade_df1_f32_red_0_0",
        "placed_records": 23,
        "routed_edges": 39,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 898,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_biquad_cascade_df1_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 39:
        raise AssertionError(f"arm_biquad_cascade_df1_f32 mapping should expose every routed edge: {mapping_artifact}")
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    required_edges = {
        "dataflow.carry#4.result0->dataflow.load#0.operand1",
        "dataflow.carry#4.result0->dataflow.store#0.operand1",
        "dataflow.invariant#4.result0->arith.mulf#0.operand0",
        "dataflow.carry#3.result0->arith.mulf#0.operand1",
        "arith.mulf#0.result0->llvm.intr.fmuladd#0.operand2",
        "llvm.intr.fmuladd#0.result0->llvm.intr.fmuladd#1.operand2",
        "llvm.intr.fmuladd#1.result0->llvm.intr.fmuladd#2.operand2",
        "llvm.intr.fmuladd#2.result0->llvm.intr.fmuladd#3.operand2",
        "llvm.intr.fmuladd#3.result0->dataflow.store#0.operand2",
    }
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_biquad_cascade_df1_f32 mapping missed biquad datapath routes: {mapping_artifact}")
    for route in routes:
        if route.get("edge_ref") not in required_edges:
            continue
        segments = route.get("segments", [])
        if not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in segments):
            raise AssertionError(f"arm_biquad_cascade_df1_f32 route should traverse Fabric paths: {route}")

    cgra_report = json.loads((sim_evidence / "arm_biquad_cascade_df1_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "dfg_cycles": 291,
        "hardware_aware_cycles": 498,
        "performance_delta_cycles": 207,
        "route_segments": 171,
        "config_records": 898,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_biquad_cascade_df1_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")
    if cgra_report.get("final_outputs") != expected_outputs or cgra_report.get("final_memory_state") != expected_memory:
        raise AssertionError(f"arm_biquad_cascade_df1_f32 CGRA report should carry final state: {cgra_report}")


def assert_cmsis_var_shared_adg_evidence(sim_evidence: Path) -> None:
    dfg_report = json.loads((sim_evidence / "arm_var_f32.dfg.report.json").read_text())
    expected_graphs = ["g_t_arm_var_f32_red_0_0", "g_t_arm_var_f32_red_1_0"]
    expected_outputs = ["none", "f32:3.750000", "none", "f32:31.796875"]
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != "StatisticsFunctions/arm_var_f32.c"
        or dfg_report.get("graph") != "workload_graph_set"
        or dfg_report.get("aggregation_kind") != "workload_graph_set"
        or dfg_report.get("component_graphs") != expected_graphs
        or dfg_report.get("status") != "pass"
        or dfg_report.get("optimistic_cycles") != 196
        or dfg_report.get("dynamic_work_items") != 8
        or dfg_report.get("final_outputs") != expected_outputs
    ):
        raise AssertionError(f"unexpected arm_var_f32 aggregate DFG report: {dfg_report}")
    expected_memory = {
        "g_t_arm_var_f32_red_0_0:arg4": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
        "g_t_arm_var_f32_red_1_0:arg5": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
    }
    if dfg_report.get("final_memory_state") != expected_memory:
        raise AssertionError(f"unexpected arm_var_f32 aggregate DFG memory: {dfg_report}")

    mapping_artifact = json.loads((sim_evidence / "arm_var_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "workload_graph_set",
        "aggregation_kind": "workload_graph_set",
        "placed_records": 22,
        "routed_edges": 30,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 684,
        "route_segments": 124,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_var_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    if mapping_artifact.get("component_graphs") != expected_graphs:
        raise AssertionError(f"arm_var_f32 mapping should preserve component graph coverage: {mapping_artifact}")
    if len(mapping_artifact.get("routes", [])) != 30:
        raise AssertionError(f"arm_var_f32 mapping should expose every component route: {mapping_artifact}")

    cgra_report = json.loads((sim_evidence / "arm_var_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "graph": "workload_graph_set",
        "status": "pass",
        "aggregation_kind": "workload_graph_set",
        "fidelity_level": "mapping_constraint_estimate",
        "dfg_cycles": 196,
        "hardware_aware_cycles": 353,
        "performance_delta_cycles": 157,
        "route_segments": 124,
        "config_records": 684,
        "functional_state_source": "component_cgra_sim_reports_carried_from_dfg_sim_reports",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_var_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")
    if cgra_report.get("component_graphs") != expected_graphs:
        raise AssertionError(f"arm_var_f32 CGRA report should preserve component graph coverage: {cgra_report}")
    if cgra_report.get("final_outputs") != expected_outputs or cgra_report.get("final_memory_state") != expected_memory:
        raise AssertionError(f"arm_var_f32 CGRA report should carry aggregate final state: {cgra_report}")


def assert_cmsis_dfg_sim_evidence_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    sim_evidence = out_dir / "cmsis-sim-evidence"
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--sim-evidence-dir",
            str(sim_evidence),
        ],
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_abs_f32",
        "BasicMathFunctions/arm_abs_f32.c",
        "llvm.intr.fabs",
        {
            "arg4": ["f32:-1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:1", "f32:2", "f32:3.500000", "f32:4.250000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_mult_f32",
        "BasicMathFunctions/arm_mult_f32.c",
        "arith.mulf",
        {
            "arg4": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:2", "f32:-2", "f32:-10.500000", "f32:2.125000"],
            "arg6": ["f32:2", "f32:-1", "f32:3", "f32:0.500000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_add_q15",
        "BasicMathFunctions/arm_add_q15.c",
        "llvm.arm.qadd16",
        {
            "arg4": ["i16:1000", "i16:20000", "i16:-30000", "i16:32760"],
            "arg5": ["i16:3000", "i16:32767", "i16:-32768", "i16:32767"],
            "arg6": ["i16:2000", "i16:15000", "i16:-10000", "i16:1000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_mat_add_f32",
        "MatrixFunctions/arm_mat_add_f32.c",
        "arith.addf",
        {
            "arg4": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:2", "f32:-1", "f32:3", "f32:0.500000"],
            "arg6": ["f32:3", "f32:1", "f32:-0.500000", "f32:4.750000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_mean_f32",
        "StatisticsFunctions/arm_mean_f32.c",
        "arith.addf",
        {
            "arg5": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_copy_f32",
        "SupportFunctions/arm_copy_f32.c",
        "dataflow.load",
        {
            "arg4": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_fill_f32",
        "SupportFunctions/arm_fill_f32.c",
        "dataflow.store",
        {
            "arg5": ["f32:3.250000", "f32:3.250000", "f32:3.250000", "f32:3.250000"],
        },
    )

    rows = read_rows(out_dir / "cgra-status-summary.csv")
    data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "cmsis-dsp",
        {
            "total": 16,
            "pass": 14,
            "fail": 0,
            "blocked": 0,
            "unsupported": 2,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "cmsis-nn",
        {
            "total": 18,
            "pass": 12,
            "fail": 0,
            "blocked": 0,
            "unsupported": 6,
            "missing_status": 0,
        },
    )
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_abs_f32.c", "arm_abs_f32")
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_mult_f32.c", "arm_mult_f32")
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c", "arm_add_q15")
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_offset_f32.c", "arm_offset_f32"
    )
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "MatrixFunctions/arm_mat_add_f32.c", "arm_mat_add_f32"
    )
    assert_cmsis_mat_mult_f32_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "StatisticsFunctions/arm_mean_f32.c", "arm_mean_f32"
    )
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "StatisticsFunctions/arm_var_f32.c", "arm_var_f32"
    )
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "StatisticsFunctions/arm_max_f32.c", "arm_max_f32"
    )
    assert_cmsis_cgra_pass_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-dsp",
        "FilteringFunctions/arm_biquad_cascade_df1_f32.c",
        "arm_biquad_cascade_df1_f32",
    )
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "SupportFunctions/arm_copy_f32.c", "arm_copy_f32")
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "SupportFunctions/arm_fill_f32.c", "arm_fill_f32")
    assert_cmsis_add_q15_shared_adg_evidence(sim_evidence)
    assert_cmsis_offset_shared_adg_evidence(sim_evidence)
    assert_cmsis_fill_shared_adg_evidence(sim_evidence)
    assert_cmsis_mean_shared_adg_evidence(sim_evidence)
    assert_cmsis_max_shared_adg_evidence(sim_evidence)
    assert_cmsis_biquad_shared_adg_evidence(sim_evidence)
    assert_cmsis_var_shared_adg_evidence(sim_evidence)
    assert_cmsis_cfft_component_evidence(repo, rows, sim_evidence)
    assert_cmsis_fir_component_pass_evidence(repo, rows, sim_evidence)
    assert_cgra_status_audit_rejects_bad_aggregate_graphs(repo, out_dir, legacy_root)
    assert_generic_artifact_audit_rejects_bad_aggregate_graphs(repo, out_dir)
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c", "arm_relu_q15"
    )
    assert_cmsis_relu_q7_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_relu6_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cgra_status_audit_rejects_bad_relu_q7_mapping(repo, out_dir, legacy_root)
    assert_cmsis_concat_w_memcpy_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_concat_memcpy_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_reshape_memcpy_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_vector_sum_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_minimum_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_maximum_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_softmax_u8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_depthwise_conv_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_max_pool_s8_cgra_evidence(repo, rows, sim_evidence)
    assert_cmsis_dfg_unsupported_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-nn",
        "FullyConnectedFunctions/arm_fully_connected_s8.c",
        "arm_fully_connected_s8",
        "g_t_arm_fully_connected_s8_red_0_0",
        "unsupported op: llvm.call @arm_nn_vec_mat_mult_t_s8",
        expected_callee="@arm_nn_vec_mat_mult_t_s8",
    )
    fake_cgra_tool = out_dir / "not-executable-cgra-sim"
    fake_cgra_tool.write_text("#!/bin/sh\nexit 99\n")
    no_cgra_evidence = out_dir / "cmsis-sim-evidence-no-cgra"
    fake_result = run(
        repo,
        [
            sys.executable,
            "test/e2e/run_cmsis_dfg_sim_attempts.py",
            "--cmsis-dsp-dfg-dir",
            str(out_dir / "cmsis-dsp-dfg"),
            "--cmsis-nn-dfg-dir",
            str(out_dir / "cmsis-nn-dfg"),
            "--output-dir",
            str(no_cgra_evidence),
            "--attempt-stem",
            "arm_abs_f32",
            "--loom-cgra-sim",
            str(fake_cgra_tool),
        ],
        expect_success=False,
    )
    if "CGRA-sim" not in fake_result.stderr and "loom-cgra-sim" not in fake_result.stderr:
        raise AssertionError(f"selected CMSIS attempt should fail at unavailable CGRA-sim: {fake_result.stderr}")
    if not (no_cgra_evidence / "arm_abs_f32.mapping.json").is_file():
        raise AssertionError("CMSIS abs should emit mapping evidence before requiring CGRA-sim")
    if (no_cgra_evidence / "arm_abs_f32.cgra.report.json").exists():
        raise AssertionError("CMSIS abs should not emit CGRA evidence from a failing CGRA-sim tool")


def write_legacy_case(root: Path, name: str, *, with_header: bool = True) -> None:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    (case_dir / "main.cpp").write_text("int main() { return 0; }\n")
    (case_dir / f"{name}.cpp").write_text(f'#include "{name}.h"\n')
    if with_header:
        (case_dir / f"{name}.h").write_text("#pragma once\n")


def assert_app_default_batch_manifest_fail_fast(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    out_dir.mkdir(parents=True)
    invalid_manifest = out_dir / "invalid-default-cgra-sim-batch.json"
    invalid_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"case": "vecsum", "hardware": "not_a_real_shared_adg"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    result = run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir / "invalid-default-batch"),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-default-batch",
        ],
        env={**os.environ, "LOOM_DEFAULT_CGRA_SIM_BATCH": str(invalid_manifest)},
        expect_success=False,
    )
    if "unsupported hardware" not in result.stderr:
        raise AssertionError(f"default batch manifest failure should be reported by the shell wrapper: {result.stderr}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "cmsis-cgra-status-rollup-") as raw_out_dir:
        out_dir = Path(raw_out_dir)
        assert_cmsis_attempt_guard_rejects_bad_relu_q7_report(repo)
        assert_no_legacy_mode(repo, out_dir / "no-legacy")
        legacy_root = out_dir / "legacy-loombench"
        write_legacy_case(legacy_root, "legacy_missing")
        write_legacy_case(legacy_root, "vecadd")
        write_legacy_case(legacy_root, "cdma")
        write_legacy_case(legacy_root, "line_intersect")
        write_legacy_case(legacy_root, "database_join")
        write_legacy_case(legacy_root, "depthwise_conv")
        write_legacy_case(legacy_root, "normalize")
        write_legacy_case(legacy_root, "spmm")
        write_legacy_case(legacy_root, "rle_decode")
        write_legacy_case(legacy_root, "blocked_case", with_header=False)
        assert_default_legacy_root_mode(repo, out_dir / "default-legacy")
        assert_explicit_legacy_root_must_exist(repo, out_dir / "missing-explicit-legacy")
        assert_app_default_batch_manifest_fail_fast(repo, out_dir / "manifest-fail-fast", legacy_root)
        assert_app_seed_batch_mode(repo, out_dir / "app-seed-batch")
        assert_app_attempt_manifest_mode(repo, out_dir / "app-attempt-manifest", legacy_root)
        assert_sort_insertion_attempt_manifest_mode(repo, out_dir / "sort-insertion-attempt", legacy_root)
        assert_primary_graph_absence_attempt_mode(
            repo,
            out_dir / "no-dfg-app-attempt-col2im",
            legacy_root,
            case="col2im",
            expected_primary_graph_token="col2im_kernel",
            expected_discovered_graph=EMPTY_DISCOVERED_GRAPH_IDS,
            expected_residual_call="col2im_kernel",
        )
        assert_primary_graph_absence_empty_graph_guard(repo, out_dir / "primary-graph-empty-guard")
        assert_primary_graph_absence_attempt_mode(
            repo,
            out_dir / "no-dfg-app-attempt-string-compare",
            legacy_root,
            case="string_compare",
            expected_primary_graph_token="string_compare_kernel",
            expected_discovered_graph="g_t_main_0_0",
            expected_residual_call="string_compare_kernel",
        )
        assert_direct_cmsis_dfg_mode(repo, out_dir / "direct-cmsis-dfg", legacy_root)
        assert_app_cgra_sweep_mode(repo, out_dir / "app-cgra-sweep", legacy_root)
        assert_cmsis_sim_default_mode(repo, out_dir / "cmsis-sim-default", legacy_root)
        assert_cmsis_dfg_sim_evidence_mode(repo, out_dir / "cmsis-dfg-sim-evidence", legacy_root)
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cmsis_cgra_status_rollup.sh",
                "--output-dir",
                str(out_dir),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
        )

        csv_output = out_dir / "cgra-status-summary.csv"
        json_output = out_dir / "cgra-status-summary.json"
        audit_output = out_dir / "cgra-status-generic-audit.json"
        manifest_output = out_dir / "loombench-manifest.json"
        manifest_csv_output = out_dir / "loombench-manifest.csv"
        for artifact in (csv_output, json_output, audit_output, manifest_output, manifest_csv_output):
            if not artifact.is_file():
                raise AssertionError(f"missing expected rollup artifact: {artifact}")
        assert_manifest_projection(manifest_output, manifest_csv_output)

        rows = read_rows(csv_output)
        data = json.loads(json_output.read_text())
        assert_cmsis_dfg_only_counts(data)
        assert_counts(
            data,
            "loombench",
            {
                "total": 10,
                "pass": 0,
                "fail": 0,
                "blocked": 9,
                "unsupported": 1,
                "missing_status": 0,
            },
        )
        assert_no_cmsis_pass(rows)

        dsp_add = one_row(rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        if (
            dsp_add["status"] != "blocked"
            or dsp_add["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or dsp_add["blocking_prerequisite"] != "dfg_sim_report"
            or dsp_add["required_slice_count"] != "1"
            or "g_t_arm_add_q15_red_0_0" not in dsp_add["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-DSP add row should be an exact DFG-sim blocker: {dsp_add}")
        assert_sha256_file(dsp_add["dfg_mlir"], dsp_add["dfg_mlir_fingerprint"], repo)

        dsp_sin = one_row(rows, "cmsis-dsp", "FastMathFunctions/arm_sin_f32.c")
        if (
            dsp_sin["status"] != "unsupported"
            or dsp_sin["diagnostic_class"] != "cmsis_no_dataflow_graph"
            or dsp_sin["blocking_prerequisite"] != "dataflow_graph"
            or dsp_sin["required_slice_count"] != "0"
        ):
            raise AssertionError(f"CMSIS-DSP no-graph row should be structured unsupported: {dsp_sin}")
        assert_sha256_file(dsp_sin["dfg_mlir"], dsp_sin["dfg_mlir_fingerprint"], repo)

        nn_relu = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
        if (
            nn_relu["status"] != "blocked"
            or nn_relu["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or nn_relu["blocking_prerequisite"] != "dfg_sim_report"
            or "g_t_arm_relu_q15_red_0_0" not in nn_relu["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-NN relu row should be an exact DFG-sim blocker: {nn_relu}")
        assert_sha256_file(nn_relu["dfg_mlir"], nn_relu["dfg_mlir_fingerprint"], repo)

        nn_relu6 = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu6_s8.c")
        if (
            nn_relu6["status"] != "blocked"
            or nn_relu6["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or nn_relu6["blocking_prerequisite"] != "dfg_sim_report"
            or nn_relu6["required_slice_count"] != "1"
            or "g_t_arm_relu6_s8_0_0" not in nn_relu6["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-NN relu6 row should be an exact DFG-sim blocker: {nn_relu6}")
        assert_sha256_file(nn_relu6["dfg_mlir"], nn_relu6["dfg_mlir_fingerprint"], repo)

        loombench_vecadd = one_row(rows, "loombench", "vecadd")
        if (
            loombench_vecadd["status"] != "blocked"
            or loombench_vecadd["diagnostic_class"] != "loombench_workload_identity_bridge_ready"
            or loombench_vecadd["blocking_prerequisite"] != "sim_evidence"
            or loombench_vecadd["manifest_case"] != "vecadd"
        ):
            raise AssertionError(f"LoomBench accepted row should expose explicit evidence bridge: {loombench_vecadd}")
        loombench_rle_decode = one_row(rows, "loombench", "rle_decode")
        if (
            loombench_rle_decode["status"] != "blocked"
            or loombench_rle_decode["diagnostic_class"] != "loombench_workload_identity_bridge_ready"
            or loombench_rle_decode["blocking_prerequisite"] != "sim_evidence"
            or loombench_rle_decode["manifest_case"] != "rle_decode"
        ):
            raise AssertionError(f"LoomBench rle_decode row should expose explicit evidence bridge: {loombench_rle_decode}")
        loombench_spmm = one_row(rows, "loombench", "spmm")
        if (
            loombench_spmm["status"] != "blocked"
            or loombench_spmm["diagnostic_class"] != "loombench_workload_identity_bridge_ready"
            or loombench_spmm["blocking_prerequisite"] != "sim_evidence"
            or loombench_spmm["manifest_case"] != "spmm"
        ):
            raise AssertionError(f"LoomBench spmm row should expose explicit evidence bridge: {loombench_spmm}")
        loombench_deferred = one_row(rows, "loombench", "legacy_missing")
        if (
            loombench_deferred["status"] != "blocked"
            or loombench_deferred["diagnostic_class"] != "loombench_import_deferred"
            or loombench_deferred["blocking_prerequisite"] != "app_import"
        ):
            raise AssertionError(f"LoomBench deferred row should be a structured blocker: {loombench_deferred}")
        loombench_excluded = one_row(rows, "loombench", "blocked_case")
        if (
            loombench_excluded["status"] != "unsupported"
            or loombench_excluded["diagnostic_class"] != "loombench_import_excluded"
            or loombench_excluded["blocking_prerequisite"] != "legacy_source"
        ):
            raise AssertionError(f"LoomBench excluded row should be structured unsupported: {loombench_excluded}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
