#!/usr/bin/env python3
"""Regression test for batch CGRA-sim evidence production."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import artifact_test_common


APP_NO_DFG_TIER_COUNT = 38
DEFAULT_SWEEP_CASES = (
    "autocorrelation",
    "vecsum",
    "vecsum-while",
    "dotproduct",
    "dotprod",
    "dot_product_3d",
    "axpy",
    "binary_search",
    "bit_reverse",
    "clz",
    "ctz",
    "downsample",
    "downsample_avg",
    "delta_encode",
    "delta_decode",
    "find_first_set",
    "prefix_sum",
    "cumsum",
    "prefix_sum_inclusive",
    "prefix_sum_exclusive",
    "pack_bits",
    "parity",
    "partition",
    "popcount",
    "unpack_bits",
    "integrate_trapz",
    "reduction",
    "mean",
    "vecnorm_l1",
    "vecnorm_l2",
    "correlation",
    "covariance",
    "compare_swap",
    "compact",
    "hash_mix",
    "string_hash",
    "merge",
    "modmul",
    "spmv",
    "convolve_1d",
    "conv1d",
    "convolve_1d_same",
    "crc32",
    "fir_filter",
    "fir_filter_stateful",
    "gather",
    "gf_mul",
    "gemv",
    "gemm",
    "matmul",
    "mat3x3_mult",
    "spmspv",
    "lower_bound",
    "matvec",
    "moving_avg",
    "newton_iter",
    "outer",
    "byte_swap",
    "scatter_add",
    "xor_block",
    "relu",
    "rotate_bits",
    "runge_kutta_step",
    "sbox_lookup",
    "transpose",
    "upper_bound",
    "upsample",
    "vecadd",
    "vecmul",
    "vecscale",
    "variance",
)
MAPPING_FAILED_SWEEP_CASES: tuple[str, ...] = ()
MAPPING_BLOCKED_SWEEP_CASES: tuple[str, ...] = ()
DFG_BLOCKED_SWEEP_CASES: tuple[str, ...] = ()
DFG_UNSUPPORTED_SWEEP_CASES = (
    "autocorrelation",
    "binary_search",
    "clz",
    "crc32",
    "ctz",
    "find_first_set",
    "gather",
    "lower_bound",
    "merge",
    "moving_avg",
    "outer",
    "pack_bits",
    "parity",
    "popcount",
    "scatter_add",
    "transpose",
    "unpack_bits",
    "upper_bound",
)
PRIMARY_GRAPH_MISSING_SWEEP_CASES = (
    ("binary_search", "binary_search_candidate"),
    ("clz", "clz_candidate"),
    ("ctz", "ctz_candidate"),
    ("find_first_set", "find_first_set_candidate"),
    ("gather", "gather"),
    ("lower_bound", "lower_bound_candidate"),
    ("moving_avg", "moving_avg_kernel"),
    ("outer", "outer_kernel"),
    ("parity", "parity"),
    ("popcount", "popcount_candidate"),
    ("scatter_add", "scatter_add"),
    ("transpose", "transpose"),
    ("upper_bound", "upper_bound_candidate"),
)

HEADER = [
    "suite",
    "case",
    "source_row",
    "manifest_case",
    "software_root",
    "graph_ids",
    "dfg_mlir",
    "dfg_mlir_fingerprint",
    "required_slice_count",
    "hardware_system",
    "spatialcore_template",
    "mapping_id",
    "dfg_report",
    "dfg_report_fingerprint",
    "dfg_status",
    "mapping_artifact",
    "mapping_artifact_fingerprint",
    "mapping_status",
    "cgra_report",
    "cgra_report_fingerprint",
    "cgra_status",
    "comparison_report",
    "comparison_report_fingerprint",
    "comparison_status",
    "final_outputs_present",
    "final_memory_state_present",
    "status",
    "diagnostic_class",
    "owner",
    "blocking_prerequisite",
    "diagnostic",
]


def assert_default_sweep_cases(script: Path) -> None:
    text = script.read_text()
    marker = "if [[ ${#CASES[@]} -eq 0 ]]; then"
    start = text.find(marker)
    if start == -1:
        raise AssertionError(f"default sweep case block is missing: {script}")
    end = text.find("  )", start)
    if end == -1:
        raise AssertionError(f"default sweep case block is malformed: {script}")
    block = text[start:end]
    cases = tuple(
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith(("if ", "CASES=("))
    )
    if cases != DEFAULT_SWEEP_CASES:
        raise AssertionError(f"default sweep cases changed: {cases}")


def run(repo: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed with {result.returncode}: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames != HEADER:
            raise AssertionError(f"unexpected header: {reader.fieldnames}")
        return rows


def assert_default_batch_rejects_non_pass_evidence(repo: Path, out_dir: Path) -> None:
    manifest = out_dir / "invalid-default-cgra-sim-batch.json"
    evidence = out_dir / "invalid-default-evidence"
    evidence.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"case": "delta_encode", "hardware": "shared_reduction_adg"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    artifacts = {
        "dfg.report.json": "dfg_sim_report",
        "mapping.json": "pnr_mapping",
        "cgra.report.json": "cgra_sim_report",
        "sim-comparison-report.json": "sim_comparison_report",
    }
    for suffix, kind in artifacts.items():
        (evidence / f"delta_encode.{suffix}").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": kind,
                    "workload": "delta_encode",
                    "hardware": "shared_reduction_adg",
                    "status": "blocked",
                    "diagnostics": ["fixture non-pass evidence"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    result = subprocess.run(
        [
            "python3",
            "test/app/default_cgra_sim_batch.py",
            "--manifest",
            str(manifest),
            "--validate-evidence-dir",
            str(evidence),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError("default batch validation accepted non-pass simulator evidence")
    if "status" not in result.stderr:
        raise AssertionError(f"default batch non-pass diagnostic missing status detail: {result.stderr}")


def parse_sweep_statuses(stdout: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("[") or "] " not in stripped:
            continue
        case, status = stripped[1:].split("] ", 1)
        if case:
            statuses[case] = status.strip()
    return statuses


def one_row(rows: list[dict[str, str]], case: str) -> dict[str, str]:
    matches = [row for row in rows if row["suite"] == "app" and row["case"] == case]
    if len(matches) != 1:
        raise AssertionError(f"expected one app/{case} row, got {matches}")
    return matches[0]


def app_manifest_no_dfg_cases(repo: Path) -> tuple[str, ...]:
    manifest_path = repo / "test/app/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise AssertionError(f"app manifest cases should be a list: {manifest_path}")
    no_dfg_cases: list[str] = []
    for entry in cases:
        if not isinstance(entry, dict):
            continue
        case = entry.get("case")
        tiers = entry.get("tiers", [])
        if isinstance(case, str) and case and (not isinstance(tiers, list) or "dfg" not in tiers):
            no_dfg_cases.append(case)
    if len(no_dfg_cases) != APP_NO_DFG_TIER_COUNT:
        raise AssertionError(
            f"expected {APP_NO_DFG_TIER_COUNT} app rows without dfg tier, "
            f"got {len(no_dfg_cases)}: {no_dfg_cases}"
        )
    return tuple(no_dfg_cases)


def assert_sweep_artifact(evidence_dir: Path, case: str, suffix: str) -> None:
    path = evidence_dir / f"{case}.{suffix}"
    if not path.is_file():
        raise AssertionError(f"missing sweep artifact: {path}")
    data = json.loads(path.read_text())
    if data.get("workload") != case:
        raise AssertionError(f"sweep artifact has wrong workload identity: {path}: {data}")
    if data.get("status") != "pass":
        raise AssertionError(f"sweep artifact is not pass: {path}: {data}")


def assert_sweep_artifact_status(evidence_dir: Path, case: str, suffix: str, expected_status: str) -> None:
    path = evidence_dir / f"{case}.{suffix}"
    if not path.is_file():
        raise AssertionError(f"missing sweep artifact: {path}")
    data = json.loads(path.read_text())
    if data.get("workload") != case or data.get("status") != expected_status:
        raise AssertionError(f"sweep artifact has wrong identity or status: {path}: {data}")


def artifact_id(path: Path) -> str:
    for suffix in (".csv", ".json"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_comparison_artifact(evidence_dir: Path, case: str, expected_status: str) -> None:
    path = evidence_dir / f"{case}.sim-comparison-report.json"
    if not path.is_file():
        raise AssertionError(f"missing sweep comparison artifact: {path}")
    data = json.loads(path.read_text())
    if data.get("kind") != "sim_comparison_report":
        raise AssertionError(f"sweep comparison artifact has wrong kind: {path}: {data}")
    if data.get("workload") != case or data.get("status") != expected_status:
        raise AssertionError(f"sweep comparison artifact has wrong identity or status: {path}: {data}")
    expected = {
        "dfg_sim_report_identity": artifact_id(evidence_dir / f"{case}.dfg.report.json"),
        "mapping_artifact_identity": artifact_id(evidence_dir / f"{case}.mapping.json"),
        "cgra_sim_report_identity": artifact_id(evidence_dir / f"{case}.cgra.report.json"),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise AssertionError(f"sweep comparison artifact has stale {key}: {path}: {data}")


def assert_mapping_hardware(evidence_dir: Path, case: str, expected_hardware: str) -> None:
    path = evidence_dir / f"{case}.mapping.json"
    data = json.loads(path.read_text())
    if data.get("hardware") != expected_hardware:
        raise AssertionError(f"{case} should map to {expected_hardware}: {path}: {data}")


def assert_mapping_uses_switch_multihop(evidence_dir: Path, case: str) -> None:
    path = evidence_dir / f"{case}.mapping.json"
    data = json.loads(path.read_text())
    routes = data.get("routes")
    if not isinstance(routes, list):
        raise AssertionError(f"{case} mapping lacks routes array: {path}: {data}")
    saw_switch_multihop_route = False
    for route in routes:
        if not isinstance(route, dict):
            continue
        segments = route.get("segments")
        if not isinstance(segments, list):
            continue
        route_uses_switch = False
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            endpoints = (
                str(segment.get("source_endpoint", "")),
                str(segment.get("sink_endpoint", "")),
                str(segment.get("hardware_ref", "")),
            )
            if any("fabric.switch" in endpoint for endpoint in endpoints):
                route_uses_switch = True
            if any(endpoint.endswith(".out") or endpoint.endswith(".in") for endpoint in endpoints):
                raise AssertionError(f"{case} mapping uses placeholder endpoint: {path}: {data}")
        if route_uses_switch and len(segments) >= 3:
            saw_switch_multihop_route = True
    if not saw_switch_multihop_route:
        raise AssertionError(f"{case} should route through real switch multihop paths: {path}: {data}")


def assert_mapping_edges_use_switch_multihop(
    evidence_dir: Path,
    case: str,
    expected_edges: set[str],
) -> None:
    path = evidence_dir / f"{case}.mapping.json"
    data = json.loads(path.read_text())
    routes = data.get("routes")
    if not isinstance(routes, list):
        raise AssertionError(f"{case} mapping lacks routes array: {path}: {data}")
    by_edge = {
        str(route.get("edge_ref")): route
        for route in routes
        if isinstance(route, dict) and route.get("edge_ref") is not None
    }
    for edge_ref in expected_edges:
        route = by_edge.get(edge_ref)
        if route is None:
            raise AssertionError(f"{case} mapping lacks route for {edge_ref}: {path}: {data}")
        if route.get("status") != "routed":
            raise AssertionError(f"{case} route should be routed for {edge_ref}: {path}: {route}")
        segments = route.get("segments")
        if not isinstance(segments, list) or len(segments) < 3:
            raise AssertionError(f"{case} route should be multihop for {edge_ref}: {path}: {route}")
        saw_switch = False
        segment_kinds = set()
        for segment in segments:
            if not isinstance(segment, dict):
                raise AssertionError(f"{case} route has malformed segment for {edge_ref}: {path}: {route}")
            segment_kinds.add(str(segment.get("segment_kind", "")))
            endpoints = (
                str(segment.get("source_endpoint", "")),
                str(segment.get("sink_endpoint", "")),
                str(segment.get("hardware_ref", "")),
            )
            if any("fabric.switch" in endpoint for endpoint in endpoints):
                saw_switch = True
            if any(endpoint.endswith(".out") or endpoint.endswith(".in") for endpoint in endpoints):
                raise AssertionError(f"{case} mapping uses placeholder endpoint for {edge_ref}: {path}: {route}")
        if not saw_switch:
            raise AssertionError(f"{case} route should use a real switch for {edge_ref}: {path}: {route}")
        if not {"module_path", "resource_edge"}.issubset(segment_kinds):
            raise AssertionError(f"{case} route should expose concrete path segments for {edge_ref}: {path}: {route}")


def assert_component_references_resolve(evidence_dir: Path, case: str) -> None:
    aggregate_specs = (
        (
            evidence_dir / f"{case}.dfg.report.json",
            ("component_dfg_sim_report_identities",),
        ),
        (
            evidence_dir / f"{case}.mapping.json",
            ("component_mapping_artifact_identities",),
        ),
        (
            evidence_dir / f"{case}.cgra.report.json",
            ("component_dfg_sim_report_identities", "component_cgra_sim_report_identities"),
        ),
    )
    saw_component = False
    for aggregate_path, identity_fields in aggregate_specs:
        aggregate = json.loads(aggregate_path.read_text())
        fingerprints = aggregate.get("input_artifact_fingerprints")
        if not isinstance(fingerprints, dict):
            raise AssertionError(f"aggregate lacks input artifact fingerprints: {aggregate_path}: {aggregate}")
        for field in identity_fields:
            identities = aggregate.get(field)
            if not isinstance(identities, list):
                continue
            for identity in identities:
                if not isinstance(identity, str) or not identity:
                    raise AssertionError(f"aggregate has invalid component identity: {aggregate_path}: {aggregate}")
                if not identity.startswith(f"{case}."):
                    raise AssertionError(f"component identity is not namespaced: {aggregate_path}: {identity}")
                component_path = evidence_dir / f"{identity}.json"
                if not component_path.is_file():
                    raise AssertionError(f"component identity does not resolve: {aggregate_path}: {identity}")
                if fingerprints.get(identity) != fingerprint(component_path):
                    raise AssertionError(f"component fingerprint mismatch: {aggregate_path}: {identity}")
                saw_component = True
    if not saw_component:
        raise AssertionError(f"expected aggregate component references for {case}")


def assert_promoted_row(repo: Path, rows: list[dict[str, str]], case: str) -> None:
    row = one_row(rows, case)
    if row["status"] != "pass":
        raise AssertionError(f"{case} should be promoted to CGRA-sim pass: {row}")
    for column in ("dfg_status", "mapping_status", "cgra_status", "comparison_status"):
        if row[column] != "pass":
            raise AssertionError(f"{case} should have {column}=pass: {row}")
    if row["final_outputs_present"] != "true":
        raise AssertionError(f"{case} should expose matching final outputs: {row}")
    if row["final_memory_state_present"] != "true":
        raise AssertionError(f"{case} should expose matching final memory state: {row}")
    if row["diagnostic_class"] != "cgra_sim_pass":
        raise AssertionError(f"{case} should have pass diagnostic class: {row}")
    for artifact_column, fingerprint_column in (
        ("dfg_report", "dfg_report_fingerprint"),
        ("mapping_artifact", "mapping_artifact_fingerprint"),
        ("cgra_report", "cgra_report_fingerprint"),
        ("comparison_report", "comparison_report_fingerprint"),
    ):
        path = repo / row[artifact_column]
        if not path.is_file():
            raise AssertionError(f"{case} row references missing artifact {artifact_column}: {row}")
        actual = artifact_test_common.fingerprint(path)
        if actual != row[fingerprint_column]:
            raise AssertionError(f"{case} row has stale {artifact_column} fingerprint: {row}")


def assert_structured_blocker_row(
    repo: Path,
    rows: list[dict[str, str]],
    case: str,
    expected_status: str,
    expected_mapping_status: str,
) -> None:
    row = one_row(rows, case)
    if row["status"] != expected_status:
        raise AssertionError(f"{case} should have status {expected_status}: {row}")
    if row["dfg_status"] != "pass":
        raise AssertionError(f"{case} should preserve DFG evidence before blocking: {row}")
    if row["mapping_status"] != expected_mapping_status:
        raise AssertionError(f"{case} should have mapping_status={expected_mapping_status}: {row}")
    if row["cgra_status"] != "blocked":
        raise AssertionError(f"{case} should have cgra_status=blocked: {row}")
    if row["comparison_status"] != "blocked":
        raise AssertionError(f"{case} should have comparison_status=blocked: {row}")
    if row["hardware_system"] != "shared_reduction_adg":
        raise AssertionError(f"{case} should stay on shared reduction hardware: {row}")
    expected_diagnostic_class = "mapping_artifact_blocked" if expected_status == "blocked" else "mapping_artifact_failed"
    if row["diagnostic_class"] != expected_diagnostic_class:
        raise AssertionError(f"{case} should name the mapping artifact blocker: {row}")
    if row["blocking_prerequisite"] != "mapping_artifact":
        raise AssertionError(f"{case} should block on mapping_artifact: {row}")
    if row["final_outputs_present"] != "true" or row["final_memory_state_present"] != "true":
        raise AssertionError(f"{case} should preserve final-state evidence while blocked: {row}")
    for artifact_column, fingerprint_column in (
        ("dfg_report", "dfg_report_fingerprint"),
        ("mapping_artifact", "mapping_artifact_fingerprint"),
        ("cgra_report", "cgra_report_fingerprint"),
        ("comparison_report", "comparison_report_fingerprint"),
    ):
        path = repo / row[artifact_column]
        if not path.is_file():
            raise AssertionError(f"{case} row references missing artifact {artifact_column}: {row}")
        actual = artifact_test_common.fingerprint(path)
        if actual != row[fingerprint_column]:
            raise AssertionError(f"{case} row has stale {artifact_column} fingerprint: {row}")


def assert_dfg_blocked_row(repo: Path, rows: list[dict[str, str]], case: str) -> None:
    row = one_row(rows, case)
    if row["status"] != "blocked":
        raise AssertionError(f"{case} should stay blocked while DFG-sim reports a runtime blocker: {row}")
    if row["dfg_status"] != "blocked":
        raise AssertionError(f"{case} should have dfg_status=blocked: {row}")
    if row["mapping_status"] != "pass":
        raise AssertionError(f"{case} should preserve mapping evidence after a DFG runtime blocker: {row}")
    if row["cgra_status"] != "blocked":
        raise AssertionError(f"{case} should have cgra_status=blocked: {row}")
    if row["comparison_status"] != "blocked":
        raise AssertionError(f"{case} should have comparison_status=blocked: {row}")
    if row["hardware_system"] != "shared_reduction_adg":
        raise AssertionError(f"{case} should use shared reduction hardware: {row}")
    if row["diagnostic_class"] != "dfg_report_blocked":
        raise AssertionError(f"{case} should block first on runtime DFG-sim evidence: {row}")
    if row["blocking_prerequisite"] != "dfg_report":
        raise AssertionError(f"{case} should name dfg_report as the prerequisite: {row}")
    if row["final_outputs_present"] != "false" or row["final_memory_state_present"] != "false":
        raise AssertionError(f"{case} should not claim complete final-state evidence while DFG is blocked: {row}")
    if "llvm.load address is out of range" not in row["diagnostic"]:
        raise AssertionError(f"{case} should expose the llvm.load runtime blocker: {row}")
    for artifact_column, fingerprint_column in (
        ("dfg_report", "dfg_report_fingerprint"),
        ("mapping_artifact", "mapping_artifact_fingerprint"),
        ("cgra_report", "cgra_report_fingerprint"),
        ("comparison_report", "comparison_report_fingerprint"),
    ):
        path = repo / row[artifact_column]
        if not path.is_file():
            raise AssertionError(f"{case} row references missing artifact {artifact_column}: {row}")
        actual = artifact_test_common.fingerprint(path)
        if actual != row[fingerprint_column]:
            raise AssertionError(f"{case} row has stale {artifact_column} fingerprint: {row}")


def assert_dfg_dynamic_work_items(evidence_dir: Path, case: str, expected_count: int) -> None:
    path = evidence_dir / f"{case}.dfg.report.json"
    data = json.loads(path.read_text())
    if data.get("dynamic_work_items") != expected_count:
        raise AssertionError(f"{case} should have {expected_count} dynamic work items: {path}: {data}")


def assert_operation_fire_counts(case: str, dfg: dict, expected_counts: dict[str, int]) -> None:
    actual_counts = dfg.get("operation_fire_counts", {})
    for op_name, expected in expected_counts.items():
        actual = actual_counts.get(op_name)
        if actual != expected:
            raise AssertionError(f"{case} {op_name} fire count should be {expected}, got {actual}: {dfg}")


def assert_prefix_sum_exclusive_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg4": ["i32:3", "i32:1", "i32:4", "i32:1", "i32:5", "i32:9", "i32:2", "i32:6"],
        "arg5": ["i32:0", "i32:3", "i32:4", "i32:8", "i32:9", "i32:14", "i32:23", "i32:25"],
    }
    dfg_path = evidence_dir / "prefix_sum_exclusive.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    expected_counts = {
        "llvm.load": 7,
        "dataflow.store": 7,
        "dataflow.sync": 7,
        "dataflow.carry": 8,
        "dataflow.stream": 8,
    }
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 7
        or dfg.get("final_outputs") != ["none", "i32:25"]
        or dfg.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"prefix_sum_exclusive DFG evidence should be complete: {dfg_path}: {dfg}")
    assert_operation_fire_counts("prefix_sum_exclusive", dfg, expected_counts)

    cgra_path = evidence_dir / "prefix_sum_exclusive.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("final_outputs") != ["none", "i32:25"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"prefix_sum_exclusive CGRA evidence should carry final state: {cgra_path}: {cgra}")


def assert_delta_decode_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg4": [
            "i32:100",
            "i32:2",
            "i32:3",
            "i32:5",
            "i32:5",
            "i32:7",
            "i32:8",
            "i32:5",
            "i32:7",
            "i32:8",
        ],
        "arg5": [
            "i32:100",
            "i32:102",
            "i32:105",
            "i32:110",
            "i32:115",
            "i32:122",
            "i32:130",
            "i32:135",
            "i32:142",
            "i32:150",
        ],
    }
    expected_counts = {
        "arith.addi": 9,
        "arith.index_cast": 10,
        "dataflow.carry": 10,
        "dataflow.load": 9,
        "dataflow.store": 9,
        "dataflow.stream": 10,
        "dataflow.sync": 9,
    }

    dfg_path = evidence_dir / "delta_decode.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 9
        or dfg.get("final_outputs") != ["none", "i32:150"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"delta_decode DFG evidence should be complete: {dfg_path}: {dfg}")
    assert_operation_fire_counts("delta_decode", dfg, expected_counts)

    cgra_path = evidence_dir / "delta_decode.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("final_outputs") != ["none", "i32:150"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"delta_decode CGRA evidence should carry final state: {cgra_path}: {cgra}")


def assert_spmspv_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg4": ["i32:2", "i32:3", "i32:4", "i32:1", "i32:5", "i32:6", "i32:7", "i32:2", "i32:3"],
        "arg5": ["i32:0", "i32:2", "i32:1", "i32:3", "i32:0", "i32:4", "i32:1", "i32:2", "i32:4"],
        "arg6": ["i32:3", "i32:0", "i32:2", "i32:5", "i32:0"],
    }
    expected_counts = {
        "arith.addi": 3,
        "arith.index_cast": 7,
        "arith.muli": 3,
        "dataflow.carry": 4,
        "dataflow.load": 9,
        "dataflow.stream": 4,
        "dataflow.sync": 3,
    }

    dfg_path = evidence_dir / "spmspv.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 3
        or dfg.get("optimistic_cycles") != 66
        or dfg.get("final_outputs") != ["none", "i32:4"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"spmspv DFG evidence should match the row-3 CSR dot slice: {dfg_path}: {dfg}")
    assert_operation_fire_counts("spmspv", dfg, expected_counts)

    cgra_path = evidence_dir / "spmspv.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 66
        or cgra.get("hardware_aware_cycles") != 124
        or cgra.get("final_outputs") != ["none", "i32:4"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"spmspv CGRA evidence should carry row-3 CSR dot final state: {cgra_path}: {cgra}")


def assert_mat3x3_mult_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg4": ["f32:1", "f32:1.875000", "f32:2.750000"],
        "arg6": [
            "f32:-0.500000",
            "f32:0",
            "f32:0",
            "f32:0.437500",
            "f32:0",
            "f32:0",
            "f32:0.187500",
        ],
    }
    expected_counts = {
        "arith.muli": 4,
        "arith.shrui": 3,
        "dataflow.load": 6,
        "dataflow.stream": 4,
        "dataflow.sync": 3,
        "llvm.intr.fmuladd": 3,
    }

    dfg_path = evidence_dir / "mat3x3_mult.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 3
        or dfg.get("optimistic_cycles") != 91
        or dfg.get("final_outputs") != ["none", "f32:0.835938"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"mat3x3_mult DFG evidence should match the first real matrix dot: {dfg_path}: {dfg}")
    assert_operation_fire_counts("mat3x3_mult", dfg, expected_counts)

    cgra_path = evidence_dir / "mat3x3_mult.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 91
        or cgra.get("hardware_aware_cycles") != 153
        or cgra.get("routed_edges") != 14
        or cgra.get("route_segments") != 54
        or cgra.get("final_outputs") != ["none", "f32:0.835938"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"mat3x3_mult CGRA evidence should carry the first real matrix dot state: {cgra_path}: {cgra}")


def assert_string_hash_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg5": [
            "i32:97",
            "i32:98",
            "i32:99",
            "i32:100",
            "i32:101",
            "i32:102",
            "i32:103",
            "i32:104",
        ],
    }
    expected_counts = {
        "arith.addi": 8,
        "arith.index_cast": 9,
        "arith.remui": 8,
        "arith.shli": 9,
        "dataflow.carry": 9,
        "dataflow.invariant": 20,
        "dataflow.load": 8,
        "dataflow.stream": 9,
        "dataflow.sync": 8,
    }

    dfg_path = evidence_dir / "string_hash.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 168
        or dfg.get("final_outputs") != ["none", "i32:38"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"string_hash DFG evidence should match the first real rolling-hash window: {dfg_path}: {dfg}")
    assert_operation_fire_counts("string_hash", dfg, expected_counts)

    cgra_path = evidence_dir / "string_hash.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 168
        or cgra.get("hardware_aware_cycles") != 220
        or cgra.get("routed_edges") != 12
        or cgra.get("route_segments") != 48
        or cgra.get("final_outputs") != ["none", "i32:38"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(
            f"string_hash CGRA evidence should carry the first real rolling-hash window state: {cgra_path}: {cgra}"
        )


def assert_fir_filter_stateful_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg4": [
            "f32:0.250000",
            "f32:-0.125000",
            "f32:0.500000",
            "f32:0.375000",
            "f32:-0.250000",
        ],
        "arg6": ["f32:4", "f32:3", "f32:2", "f32:1"],
    }
    expected_counts = {
        "arith.index_cast": 11,
        "arith.subi": 5,
        "dataflow.carry": 5,
        "dataflow.invariant": 6,
        "dataflow.load": 8,
        "dataflow.stream": 5,
        "dataflow.sync": 4,
        "llvm.intr.fmuladd": 4,
        "llvm.trunc": 5,
    }

    dfg_path = evidence_dir / "fir_filter_stateful.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 4
        or dfg.get("optimistic_cycles") != 105
        or dfg.get("final_outputs") != ["none", "f32:1.250000"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"fir_filter_stateful DFG evidence should match the first real stateful FIR MAC: {dfg_path}: {dfg}")
    assert_operation_fire_counts("fir_filter_stateful", dfg, expected_counts)

    cgra_path = evidence_dir / "fir_filter_stateful.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 105
        or cgra.get("hardware_aware_cycles", 0) < cgra.get("dfg_cycles", 0)
        or cgra.get("routed_edges") != 13
        or cgra.get("route_segments") != 49
        or cgra.get("final_outputs") != ["none", "f32:1.250000"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"fir_filter_stateful CGRA evidence should carry the first real stateful FIR MAC: {cgra_path}: {cgra}")


def assert_covariance_evidence(evidence_dir: Path) -> None:
    expected_head_x = ["f32:0", "f32:1", "f32:2", "f32:3"]
    expected_tail_x = ["f32:20", "f32:21", "f32:22", "f32:23"]
    expected_head_y = ["f32:0.500000", "f32:2.500000", "f32:4.500000", "f32:6.500000"]
    expected_tail_y = ["f32:40.500000", "f32:42.500000", "f32:44.500000", "f32:46.500000"]

    def check_real_inputs(memory: dict, x_key: str, y_key: str, label: str) -> None:
        x_values = memory.get(x_key)
        y_values = memory.get(y_key)
        if (
            not isinstance(x_values, list)
            or not isinstance(y_values, list)
            or len(x_values) != 1024
            or len(y_values) != 1024
            or x_values[:4] != expected_head_x
            or x_values[-4:] != expected_tail_x
            or y_values[:4] != expected_head_y
            or y_values[-4:] != expected_tail_y
        ):
            raise AssertionError(f"covariance {label} should use source-derived x/y input arrays")

    expected_outputs = ["none", "f32:49776", "f32:50064", "none", "f32:441956.250000"]

    dfg_path = evidence_dir / "covariance.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 2048
        or dfg.get("optimistic_cycles") != 44043
        or dfg.get("final_outputs") != expected_outputs
        or dfg.get("component_graphs") != ["g_t_covariance_kernel_red_0_0", "g_t_covariance_kernel_red_1_0"]
    ):
        raise AssertionError(f"covariance DFG aggregate should carry real two-pass covariance state: {dfg_path}: {dfg}")
    memory = dfg.get("final_memory_state", {})
    check_real_inputs(memory, "g_t_covariance_kernel_red_0_0:arg4", "g_t_covariance_kernel_red_0_0:arg5", "sums")
    check_real_inputs(memory, "g_t_covariance_kernel_red_1_0:arg4", "g_t_covariance_kernel_red_1_0:arg6", "covariance")

    component_identities = dfg.get("component_dfg_sim_report_identities", [])
    components = [json.loads((evidence_dir / f"{identity}.json").read_text()) for identity in component_identities]
    by_graph = {component.get("graph"): component for component in components}
    sums = by_graph.get("g_t_covariance_kernel_red_0_0")
    cov = by_graph.get("g_t_covariance_kernel_red_1_0")
    if not isinstance(sums, dict) or not isinstance(cov, dict):
        raise AssertionError(f"covariance DFG aggregate should reference both component reports: {dfg}")
    if sums.get("final_outputs") != ["none", "f32:49776", "f32:50064"] or sums.get("diagnostics") != []:
        raise AssertionError(f"covariance sums component should report source-derived sums: {sums}")
    if cov.get("final_outputs") != ["none", "f32:441956.250000"] or cov.get("diagnostics") != []:
        raise AssertionError(f"covariance component should report source-derived covariance accumulator: {cov}")
    assert_operation_fire_counts(
        "covariance sums",
        sums,
        {"arith.addf": 2048, "dataflow.load": 2048, "dataflow.sync": 1024},
    )
    assert_operation_fire_counts(
        "covariance covariance",
        cov,
        {"arith.subf": 2048, "dataflow.load": 2048, "dataflow.sync": 1024, "llvm.intr.fmuladd": 1024},
    )

    cgra_path = evidence_dir / "covariance.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 44043
        or cgra.get("hardware_aware_cycles") != 44166
        or cgra.get("routed_edges") != 27
        or cgra.get("route_segments") != 107
        or cgra.get("final_outputs") != expected_outputs
        or cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
    ):
        raise AssertionError(f"covariance CGRA aggregate should carry real two-pass covariance state: {cgra_path}: {cgra}")
    cgra_memory = cgra.get("final_memory_state", {})
    check_real_inputs(cgra_memory, "g_t_covariance_kernel_red_0_0:arg4", "g_t_covariance_kernel_red_0_0:arg5", "CGRA sums")
    check_real_inputs(cgra_memory, "g_t_covariance_kernel_red_1_0:arg4", "g_t_covariance_kernel_red_1_0:arg6", "CGRA covariance")


def assert_modmul_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": [
            "i32:12345",
            "i32:24690",
            "i32:987654321",
            "i32:42",
            "i32:65535",
            "i32:1000000006",
            "i32:314159",
            "i32:271828",
        ],
        "arg2": [
            "i32:67890",
            "i32:13579",
            "i32:123456789",
            "i32:99",
            "i32:65537",
            "i32:1000000006",
            "i32:271828",
            "i32:314159",
        ],
        "arg4": ["i32:838102050", "i32:0", "i32:0", "i32:0", "i32:0", "i32:0", "i32:0", "i32:0"],
    }
    expected_counts = {
        "arith.muli": 1,
        "arith.remui": 1,
        "dataflow.load": 2,
        "dataflow.store": 1,
        "dataflow.sync": 1,
        "llvm.trunc": 1,
        "llvm.zext": 2,
    }

    dfg_path = evidence_dir / "modmul.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 1
        or dfg.get("optimistic_cycles") != 27
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"modmul DFG evidence should match the first real modular product: {dfg_path}: {dfg}")
    assert_operation_fire_counts("modmul", dfg, expected_counts)

    mapping_path = evidence_dir / "modmul.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("placed_records") != 9
        or mapping.get("routed_edges") != 10
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("diagnostics") != ["mapped software graph to fabric resources"]
    ):
        raise AssertionError(f"modmul mapping should route real 64-bit modular arithmetic on the shared ADG: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "modmul.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 27
        or cgra.get("hardware_aware_cycles") != 81
        or cgra.get("routed_edges") != 10
        or cgra.get("route_segments") != 42
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"modmul CGRA-sim should carry the first real modular product state: {cgra_path}: {cgra}")


def assert_newton_iter_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": ["f32:1", "f32:2", "f32:3", "f32:4"],
        "arg2": ["f32:0", "f32:2", "f32:6", "f32:12"],
        "arg3": ["f32:2", "f32:4", "f32:6", "f32:8"],
        "arg4": ["f32:0", "f32:1.500000", "f32:0", "f32:0"],
    }
    expected_counts = {
        "arith.divf": 1,
        "arith.subf": 1,
        "dataflow.load": 3,
        "dataflow.store": 1,
        "dataflow.sync": 1,
    }

    dfg_path = evidence_dir / "newton_iter.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 1
        or dfg.get("optimistic_cycles") != 31
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"newton_iter DFG evidence should match x - f/df real fixture: {dfg_path}: {dfg}")
    assert_operation_fire_counts("newton_iter", dfg, expected_counts)

    mapping_path = evidence_dir / "newton_iter.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("unrouted_edges") != 0
        or mapping.get("routed_edges") != 9
    ):
        raise AssertionError(f"newton_iter should route on shared reduction hardware: {mapping_path}: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_edges = {
        "arith.divf#0.result0->arith.subf#0.operand1",
        "arith.subf#0.result0->dataflow.store#0.operand2",
        "dataflow.store#0.result0->dataflow.sync#0.operand3",
    }
    if not expected_edges.issubset(route_edges):
        raise AssertionError(f"newton_iter mapping should expose div/sub/store/sync route edges: {mapping}")

    cgra_path = evidence_dir / "newton_iter.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 31
        or cgra.get("hardware_aware_cycles") != 78
        or cgra.get("routed_edges") != 9
        or cgra.get("route_segments") != 31
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"newton_iter CGRA evidence should carry x - f/df final state: {cgra_path}: {cgra}")


def assert_runge_kutta_step_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": ["f32:1", "f32:1.100000", "f32:1.200000", "f32:1.300000"],
        "arg2": ["f32:1.100000", "f32:1.200000", "f32:1.300000", "f32:1.400000"],
        "arg4": ["f32:1.200000", "f32:1.300000", "f32:1.400000", "f32:1.500000"],
        "arg5": ["f32:1.300000", "f32:1.400000", "f32:1.500000", "f32:1.600000"],
        "arg6": ["f32:0", "f32:1", "f32:2", "f32:3"],
        "arg8": ["f32:0.115000", "f32:0", "f32:0", "f32:0"],
    }
    expected_counts = {
        "arith.addf": 1,
        "dataflow.load": 5,
        "dataflow.store": 1,
        "dataflow.sync": 1,
        "llvm.intr.fmuladd": 3,
    }

    dfg_path = evidence_dir / "runge_kutta_step.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 1
        or dfg.get("optimistic_cycles") != 51
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"runge_kutta_step DFG evidence should match the first real RK4 update: {dfg_path}: {dfg}")
    assert_operation_fire_counts("runge_kutta_step", dfg, expected_counts)

    mapping_path = evidence_dir / "runge_kutta_step.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("unrouted_edges") != 0
        or mapping.get("routed_edges") != 15
    ):
        raise AssertionError(f"runge_kutta_step should route on shared reduction hardware: {mapping_path}: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_edges = {
        "dataflow.load#0.result0->llvm.intr.fmuladd#0.operand2",
        "dataflow.load#1.result0->llvm.intr.fmuladd#0.operand0",
        "llvm.intr.fmuladd#1.result0->arith.addf#0.operand0",
        "arith.addf#0.result0->llvm.intr.fmuladd#2.operand1",
        "dataflow.load#4.result0->llvm.intr.fmuladd#2.operand2",
        "dataflow.store#0.result0->dataflow.sync#0.operand5",
        "llvm.intr.fmuladd#2.result0->dataflow.store#0.operand2",
    }
    if not expected_edges.issubset(route_edges):
        raise AssertionError(f"runge_kutta_step mapping should expose RK4 FMA/add/store/sync route edges: {mapping}")

    cgra_path = evidence_dir / "runge_kutta_step.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 51
        or cgra.get("hardware_aware_cycles") != 126
        or cgra.get("routed_edges") != 15
        or cgra.get("route_segments") != 51
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"runge_kutta_step CGRA evidence should carry the first real RK4 update state: {cgra_path}: {cgra}")


def assert_gf_mul_evidence(evidence_dir: Path) -> None:
    expected_outputs = ["none", "i32:10433", "i32:0", "i32:20592", "i32:193"]
    expected_counts = {
        "arith.andi": 27,
        "arith.cmpi": 18,
        "arith.select": 18,
        "arith.shli": 9,
        "arith.shrui": 9,
        "arith.xori": 18,
        "dataflow.carry": 30,
        "dataflow.invariant": 50,
        "dataflow.stream": 9,
    }

    dfg_path = evidence_dir / "gf_mul.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 188
        or dfg.get("final_outputs") != expected_outputs
        or dfg.get("final_memory_state") != {}
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"gf_mul DFG evidence should carry the real GF(2^8) product state: {dfg_path}: {dfg}")
    assert_operation_fire_counts("gf_mul", dfg, expected_counts)

    cgra_path = evidence_dir / "gf_mul.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 188
        or cgra.get("hardware_aware_cycles", 0) < cgra.get("dfg_cycles", 0)
        or cgra.get("routed_edges", 0) <= 19
        or cgra.get("route_segments", 0) <= 0
        or cgra.get("final_outputs") != expected_outputs
        or cgra.get("final_memory_state") != {}
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"gf_mul CGRA evidence should carry the real GF(2^8) product state: {cgra_path}: {cgra}")


def assert_compact_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg4": [
            "i32:10",
            "i32:0",
            "i32:20",
            "i32:0",
            "i32:30",
            "i32:40",
            "i32:0",
            "i32:50",
            "i32:0",
            "i32:60",
            "i32:70",
            "i32:0",
        ],
        "arg6": [
            "i32:10",
            "i32:20",
            "i32:30",
            "i32:40",
            "i32:50",
            "i32:60",
            "i32:70",
            "i32:0",
            "i32:0",
            "i32:0",
            "i32:0",
            "i32:0",
        ],
    }
    expected_counts = {
        "arith.addi": 13,
        "arith.cmpi": 12,
        "arith.index_cast": 26,
        "arith.select": 12,
        "dataflow.carry": 13,
        "dataflow.demux": 36,
        "dataflow.invariant": 28,
        "dataflow.load": 12,
        "dataflow.mux": 12,
        "dataflow.store": 7,
        "dataflow.stream": 13,
        "dataflow.sync": 12,
    }

    dfg_path = evidence_dir / "compact.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 12
        or dfg.get("final_outputs") != ["none", "i32:7"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"compact DFG evidence should match the real filtered copy state: {dfg_path}: {dfg}")
    assert_operation_fire_counts("compact", dfg, expected_counts)

    mapping_path = evidence_dir / "compact.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("placed_records") != 14
        or mapping.get("routed_edges") != 25
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("diagnostics") != ["mapped software graph to fabric resources"]
    ):
        raise AssertionError(f"compact should map control tokens on the shared ADG: {mapping_path}: {mapping}")
    route_refs = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_route_refs = {
        "arith.cmpi#0.result0->dataflow.demux#0.operand0",
        "arith.cmpi#0.result0->dataflow.demux#1.operand0",
        "arith.cmpi#0.result0->dataflow.demux#2.operand0",
        "arith.cmpi#0.result0->dataflow.mux#0.operand0",
        "dataflow.demux#2.result1->dataflow.mux#0.operand2",
        "dataflow.store#0.result0->dataflow.mux#0.operand1",
    }
    if not expected_route_refs.issubset(route_refs):
        missing = sorted(expected_route_refs - route_refs)
        raise AssertionError(f"compact should route selector and token mux/demux edges, missing {missing}: {mapping_path}")

    cgra_path = evidence_dir / "compact.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 253
        or cgra.get("hardware_aware_cycles") != 374
        or cgra.get("routed_edges") != 25
        or cgra.get("route_segments") != 113
        or cgra.get("final_outputs") != ["none", "i32:7"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"compact CGRA-sim evidence should carry the real filtered copy state: {cgra_path}: {cgra}")


def assert_partition_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "g_t_partition_red_0_0:arg4": [
            "f32:3",
            "f32:7",
            "f32:1",
            "f32:9",
            "f32:5",
            "f32:2",
            "f32:8",
            "f32:4",
            "f32:6",
            "f32:10",
        ],
        "g_t_partition_red_0_0:arg6": [
            "f32:3",
            "f32:1",
            "f32:5",
            "f32:2",
            "f32:4",
            "f32:0",
            "f32:0",
            "f32:0",
            "f32:0",
            "f32:0",
        ],
        "g_t_partition_red_1_0:arg4": [
            "f32:3",
            "f32:7",
            "f32:1",
            "f32:9",
            "f32:5",
            "f32:2",
            "f32:8",
            "f32:4",
            "f32:6",
            "f32:10",
        ],
        "g_t_partition_red_1_0:arg6": [
            "f32:3",
            "f32:1",
            "f32:5",
            "f32:2",
            "f32:4",
            "f32:7",
            "f32:9",
            "f32:8",
            "f32:6",
            "f32:10",
        ],
    }
    expected_counts = {
        "arith.addi": 22,
        "arith.cmpf": 20,
        "arith.index_cast": 44,
        "arith.select": 20,
        "dataflow.carry": 22,
        "dataflow.demux": 60,
        "dataflow.invariant": 48,
        "dataflow.load": 20,
        "dataflow.mux": 20,
        "dataflow.store": 10,
        "dataflow.stream": 22,
        "dataflow.sync": 20,
    }

    dfg_path = evidence_dir / "partition.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 20
        or dfg.get("optimistic_cycles") != 438
        or dfg.get("final_outputs") != ["none", "i32:5", "none", "i32:10"]
        or dfg.get("final_memory_state") != expected_memory
        or "derived workload graph-set DFG report from component DFG simulator reports" not in dfg.get("diagnostics", [])
    ):
        raise AssertionError(f"partition DFG evidence should preserve the two-sided real partition state: {dfg_path}: {dfg}")
    assert_operation_fire_counts("partition", dfg, expected_counts)

    mapping_path = evidence_dir / "partition.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("placed_records") != 28
        or mapping.get("routed_edges") != 50
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("route_segments") != 226
        or mapping.get("diagnostics") != ["derived workload graph-set mapping artifact from component PnR mapping artifacts"]
    ):
        raise AssertionError(f"partition should aggregate passing component mappings: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "partition.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 438
        or cgra.get("hardware_aware_cycles") != 680
        or cgra.get("routed_edges") != 50
        or cgra.get("route_segments") != 226
        or cgra.get("final_outputs") != ["none", "i32:5", "none", "i32:10"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
        or cgra.get("diagnostics") != ["derived workload graph-set CGRA report from component CGRA simulator reports"]
    ):
        raise AssertionError(f"partition CGRA-sim should carry the two-sided real partition state: {cgra_path}: {cgra}")


def assert_mapping_unrouted_edges(evidence_dir: Path, case: str, expected_edges: set[str]) -> None:
    path = evidence_dir / f"{case}.mapping.json"
    data = json.loads(path.read_text())
    details = data.get("unrouted_edge_details")
    if not isinstance(details, list):
        raise AssertionError(f"{case} mapping lacks unrouted edge details: {path}: {data}")
    actual_edges = {
        str(edge.get("edge_ref"))
        for edge in details
        if isinstance(edge, dict) and edge.get("status") == "unrouted"
    }
    if actual_edges != expected_edges:
        raise AssertionError(f"{case} should expose exact unrouted edges {expected_edges}: {path}: {data}")


def assert_unsupported_operation(
    evidence_dir: Path,
    case: str,
    dfg_operation: str,
    mapping_operation: str | None = None,
    rejected_dfg_operations: tuple[str, ...] = (),
) -> None:
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    mapping_path = evidence_dir / f"{case}.mapping.json"
    dfg = json.loads(dfg_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    expected_dfg = f"unsupported op: {dfg_operation}"
    mapping_operation = mapping_operation or dfg_operation
    expected_mapping = f"unsupported PnR graph operation: {mapping_operation}"
    dfg_diagnostics = dfg.get("diagnostics", [])
    if expected_dfg not in dfg_diagnostics:
        raise AssertionError(f"{case} DFG unsupported diagnostic should be {expected_dfg}: {dfg_path}: {dfg}")
    for rejected in rejected_dfg_operations:
        rejected_diagnostic = f"unsupported op: {rejected}"
        if rejected_diagnostic in dfg_diagnostics:
            raise AssertionError(
                f"{case} DFG unsupported diagnostic should not include stale {rejected_diagnostic}: "
                f"{dfg_path}: {dfg}"
            )
    if expected_mapping not in mapping.get("diagnostics", []):
        raise AssertionError(
            f"{case} mapping unsupported diagnostic should be {expected_mapping}: {mapping_path}: {mapping}"
        )


def assert_mapping_unsupported_operation(evidence_dir: Path, case: str, operation: str) -> None:
    mapping_path = evidence_dir / f"{case}.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    expected_mapping = f"unsupported PnR graph operation: {operation}"
    if expected_mapping not in mapping.get("diagnostics", []):
        raise AssertionError(
            f"{case} mapping unsupported diagnostic should be {expected_mapping}: {mapping_path}: {mapping}"
        )


def assert_primary_graph_missing(evidence_dir: Path, case: str, expected_token: str) -> None:
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    mapping_path = evidence_dir / f"{case}.mapping.json"
    dfg = json.loads(dfg_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    expected = f"primary workload graph absent: expected token {expected_token}"
    for artifact_path, artifact in ((dfg_path, dfg), (mapping_path, mapping)):
        diagnostics = artifact.get("diagnostics")
        if not isinstance(diagnostics, list) or expected not in diagnostics:
            raise AssertionError(f"{case} should report primary graph absence: {artifact_path}: {artifact}")
    graph_ids = dfg.get("discovered_graph_ids")
    if not isinstance(graph_ids, list) or any(expected_token in str(graph_id) for graph_id in graph_ids):
        raise AssertionError(f"{case} should not expose its primary graph token yet: {dfg_path}: {dfg}")


def assert_component_mapping_status(
    evidence_dir: Path,
    case: str,
    graph: str,
    expected_status: str,
    expected_diagnostic: str | None = None,
) -> None:
    aggregate_path = evidence_dir / f"{case}.mapping.json"
    aggregate = json.loads(aggregate_path.read_text())
    identities = aggregate.get("component_mapping_artifact_identities")
    if not isinstance(identities, list):
        raise AssertionError(f"{case} aggregate mapping lacks component identities: {aggregate_path}: {aggregate}")
    for identity in identities:
        if not isinstance(identity, str) or not identity:
            continue
        component_path = evidence_dir / f"{identity}.json"
        if not component_path.is_file():
            raise AssertionError(f"{case} component mapping identity does not resolve: {component_path}")
        component = json.loads(component_path.read_text())
        if component.get("graph") != graph:
            continue
        if component.get("status") != expected_status:
            raise AssertionError(f"{case} component {graph} should have status {expected_status}: {component}")
        diagnostics = component.get("diagnostics", [])
        if expected_diagnostic is not None and expected_diagnostic not in diagnostics:
            raise AssertionError(
                f"{case} component {graph} should include diagnostic {expected_diagnostic}: {component}"
            )
        return
    raise AssertionError(f"{case} aggregate mapping should include component graph {graph}: {aggregate}")


def assert_dfg_unsupported_operation(evidence_dir: Path, case: str, operation: str) -> None:
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    expected_dfg = f"unsupported op: {operation}"
    if expected_dfg not in dfg.get("diagnostics", []):
        raise AssertionError(f"{case} DFG unsupported diagnostic should be {expected_dfg}: {dfg_path}: {dfg}")


def assert_dfg_unsupported_row(repo: Path, rows: list[dict[str, str]], case: str) -> None:
    row = one_row(rows, case)
    if row["status"] != "blocked":
        raise AssertionError(f"{case} should stay blocked while DFG-sim is unsupported: {row}")
    if row["dfg_status"] != "unsupported":
        raise AssertionError(f"{case} should have dfg_status=unsupported: {row}")
    if row["mapping_status"] != "unsupported":
        raise AssertionError(f"{case} should have mapping_status=unsupported: {row}")
    if row["cgra_status"] != "blocked":
        raise AssertionError(f"{case} should have cgra_status=blocked: {row}")
    if row["comparison_status"] != "blocked":
        raise AssertionError(f"{case} should have comparison_status=blocked: {row}")
    if row["hardware_system"] != "shared_reduction_adg":
        raise AssertionError(f"{case} should use the shared reduction hardware blocker: {row}")
    if row["diagnostic_class"] != "dfg_report_unsupported":
        raise AssertionError(f"{case} should block first on unsupported DFG-sim evidence: {row}")
    if row["blocking_prerequisite"] != "dfg_report":
        raise AssertionError(f"{case} should name dfg_report as the prerequisite: {row}")
    if row["final_outputs_present"] != "false" or row["final_memory_state_present"] != "false":
        raise AssertionError(f"{case} should not treat unsupported DFG state as functional evidence: {row}")
    for artifact_column, fingerprint_column in (
        ("dfg_report", "dfg_report_fingerprint"),
        ("mapping_artifact", "mapping_artifact_fingerprint"),
        ("cgra_report", "cgra_report_fingerprint"),
        ("comparison_report", "comparison_report_fingerprint"),
    ):
        path = repo / row[artifact_column]
        if not path.is_file():
            raise AssertionError(f"{case} row references missing artifact {artifact_column}: {row}")
        actual = artifact_test_common.fingerprint(path)
        if actual != row[fingerprint_column]:
            raise AssertionError(f"{case} row has stale {artifact_column} fingerprint: {row}")


def assert_app_no_dfg_tier_blocked_row(rows: list[dict[str, str]], case: str) -> None:
    row = one_row(rows, case)
    if row["status"] != "blocked":
        raise AssertionError(f"{case} should be structured blocked until a dataflow tier exists: {row}")
    if row["diagnostic_class"] != "app_dataflow_tier_missing":
        raise AssertionError(f"{case} should expose the app no-DFG diagnostic class: {row}")
    if row["owner"] != "compiler_pipeline":
        raise AssertionError(f"{case} should assign the blocker to compiler_pipeline: {row}")
    if row["blocking_prerequisite"] != "dataflow":
        raise AssertionError(f"{case} should block on dataflow: {row}")
    if row["required_slice_count"] != "0":
        raise AssertionError(f"{case} should not claim required DFG slices before dataflow exists: {row}")
    if row["graph_ids"]:
        raise AssertionError(f"{case} should not claim graph ids before dataflow exists: {row}")
    for column in ("dfg_status", "mapping_status", "cgra_status", "comparison_status"):
        if row[column] != "not_run":
            raise AssertionError(f"{case} should keep {column}=not_run before dataflow evidence exists: {row}")
    for column in (
        "dfg_mlir",
        "dfg_mlir_fingerprint",
        "dfg_report",
        "dfg_report_fingerprint",
        "mapping_artifact",
        "mapping_artifact_fingerprint",
        "cgra_report",
        "cgra_report_fingerprint",
        "comparison_report",
        "comparison_report_fingerprint",
    ):
        if row[column]:
            raise AssertionError(f"{case} should not reference fabricated artifact evidence in {column}: {row}")
    if row["final_outputs_present"] != "false" or row["final_memory_state_present"] != "false":
        raise AssertionError(f"{case} should not claim final-state evidence before dataflow exists: {row}")
    if "app manifest has no dfg tier" not in row["diagnostic"]:
        raise AssertionError(f"{case} diagnostic should identify the app manifest no-DFG blocker: {row}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <repo>")
    repo = Path(argv[1]).resolve()
    assert_default_sweep_cases(repo / "test/e2e/run_cgra_sim_evidence_sweep.sh")
    with artifact_test_common.repo_temp_dir(repo, "cgra-sim-evidence-sweep-") as tmp:
        out_dir = Path(tmp)
        evidence_dir = out_dir / "current-sim-cycle"
        assert_default_batch_rejects_non_pass_evidence(repo, out_dir)
        sweep_result = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_sim_evidence_sweep.sh",
                "--output-dir",
                str(evidence_dir),
                "--case",
                "autocorrelation",
                "--case",
                "vecsum",
                "--case",
                "vecsum-while",
                "--case",
                "reduction",
                "--case",
                "dotproduct",
                "--case",
                "dotprod",
                "--case",
                "dot_product_3d",
                "--case",
                "axpy",
                "--case",
                "binary_search",
                "--case",
                "bit_reverse",
                "--case",
                "clz",
                "--case",
                "ctz",
                "--case",
                "downsample",
                "--case",
                "delta_encode",
                "--case",
                "delta_decode",
                "--case",
                "find_first_set",
                "--case",
                "spmv",
                "--case",
                "spmspv",
                "--case",
                "gather",
                "--case",
                "gf_mul",
                "--case",
                "modmul",
                "--case",
                "byte_swap",
                "--case",
                "xor_block",
                "--case",
                "vecmul",
                "--case",
                "vecscale",
                "--case",
                "compare_swap",
                "--case",
                "compact",
                "--case",
                "hash_mix",
                "--case",
                "string_hash",
                "--case",
                "merge",
                "--case",
                "prefix_sum",
                "--case",
                "cumsum",
                "--case",
                "prefix_sum_inclusive",
                "--case",
                "prefix_sum_exclusive",
                "--case",
                "lower_bound",
                "--case",
                "moving_avg",
                "--case",
                "newton_iter",
                "--case",
                "outer",
                "--case",
                "pack_bits",
                "--case",
                "parity",
                "--case",
                "partition",
                "--case",
                "popcount",
                "--case",
                "scatter_add",
                "--case",
                "unpack_bits",
                "--case",
                "mean",
                "--case",
                "vecnorm_l1",
                "--case",
                "vecnorm_l2",
                "--case",
                "gemv",
                "--case",
                "matvec",
                "--case",
                "downsample_avg",
                "--case",
                "vecadd",
                "--case",
                "conv1d",
                "--case",
                "convolve_1d_same",
                "--case",
                "crc32",
                "--case",
                "variance",
                "--case",
                "integrate_trapz",
                "--case",
                "fir_filter",
                "--case",
                "fir_filter_stateful",
                "--case",
                "gemm",
                "--case",
                "matmul",
                "--case",
                "mat3x3_mult",
                "--case",
                "correlation",
                "--case",
                "covariance",
                "--case",
                "convolve_1d",
                "--case",
                "relu",
                "--case",
                "rotate_bits",
                "--case",
                "runge_kutta_step",
                "--case",
                "sbox_lookup",
                "--case",
                "transpose",
                "--case",
                "upper_bound",
                "--case",
                "upsample",
            ],
        )
        sweep_statuses = parse_sweep_statuses(sweep_result.stdout)
        if sweep_statuses.get("delta_encode") != "pass":
            raise AssertionError(f"delta_encode pass status missing from sweep stdout: {sweep_result.stdout}")
        for case in (
            "vecsum",
            "vecsum-while",
            "reduction",
            "dotproduct",
            "dotprod",
            "dot_product_3d",
            "spmv",
            "spmspv",
            "axpy",
            "bit_reverse",
            "byte_swap",
            "downsample",
            "xor_block",
            "vecmul",
            "vecscale",
            "prefix_sum",
            "cumsum",
            "prefix_sum_inclusive",
            "prefix_sum_exclusive",
            "partition",
            "mean",
            "newton_iter",
            "vecnorm_l1",
            "vecnorm_l2",
            "gemv",
            "gemm",
            "matmul",
            "mat3x3_mult",
            "matvec",
            "downsample_avg",
            "vecadd",
            "conv1d",
            "variance",
            "covariance",
            "integrate_trapz",
            "delta_encode",
            "delta_decode",
            "correlation",
            "convolve_1d",
            "convolve_1d_same",
            "fir_filter",
            "fir_filter_stateful",
            "gf_mul",
            "compare_swap",
            "compact",
            "hash_mix",
            "string_hash",
            "modmul",
            "relu",
            "upsample",
            "sbox_lookup",
            "rotate_bits",
            "runge_kutta_step",
        ):
            assert_sweep_artifact(evidence_dir, case, "dfg.report.json")
            assert_sweep_artifact(evidence_dir, case, "mapping.json")
            assert_sweep_artifact(evidence_dir, case, "cgra.report.json")
            assert_comparison_artifact(evidence_dir, case, "pass")
        for case in MAPPING_FAILED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "pass")
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", "fail")
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        for case in MAPPING_BLOCKED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "pass")
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", "blocked")
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        for case in DFG_BLOCKED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "blocked")
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", "pass")
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        assert_dfg_dynamic_work_items(evidence_dir, "gemm", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "matmul", 3)
        assert_dfg_dynamic_work_items(evidence_dir, "mat3x3_mult", 3)
        assert_dfg_dynamic_work_items(evidence_dir, "modmul", 1)
        assert_dfg_dynamic_work_items(evidence_dir, "newton_iter", 1)
        assert_dfg_dynamic_work_items(evidence_dir, "runge_kutta_step", 1)
        assert_dfg_dynamic_work_items(evidence_dir, "upsample", 4)
        assert_dfg_dynamic_work_items(evidence_dir, "sbox_lookup", 64)
        assert_dfg_dynamic_work_items(evidence_dir, "string_hash", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "fir_filter_stateful", 4)
        assert_dfg_dynamic_work_items(evidence_dir, "covariance", 2048)
        assert_prefix_sum_exclusive_evidence(evidence_dir)
        assert_delta_decode_evidence(evidence_dir)
        assert_spmspv_evidence(evidence_dir)
        assert_mat3x3_mult_evidence(evidence_dir)
        assert_fir_filter_stateful_evidence(evidence_dir)
        assert_covariance_evidence(evidence_dir)
        assert_modmul_evidence(evidence_dir)
        assert_newton_iter_evidence(evidence_dir)
        assert_runge_kutta_step_evidence(evidence_dir)
        assert_gf_mul_evidence(evidence_dir)
        assert_compact_evidence(evidence_dir)
        assert_partition_evidence(evidence_dir)
        assert_string_hash_evidence(evidence_dir)
        for case in DFG_UNSUPPORTED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "unsupported")
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", "unsupported")
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        assert_unsupported_operation(
            evidence_dir,
            "autocorrelation",
            "llvm.intr.umax",
            "scf.for",
            rejected_dfg_operations=("scf.if",),
        )
        assert_unsupported_operation(
            evidence_dir,
            "pack_bits",
            "llvm.intr.umin",
            "scf.for",
            rejected_dfg_operations=("scf.if",),
        )
        assert_unsupported_operation(
            evidence_dir,
            "merge",
            "arith.extui",
            "scf.for",
            rejected_dfg_operations=("scf.if",),
        )
        assert_unsupported_operation(evidence_dir, "crc32", "scf.for", "scf.for")
        assert_unsupported_operation(
            evidence_dir, "unpack_bits", "llvm.intr.umin", "scf.for"
        )
        for case, expected_token in PRIMARY_GRAPH_MISSING_SWEEP_CASES:
            assert_primary_graph_missing(evidence_dir, case, expected_token)
        assert_mapping_hardware(evidence_dir, "dotproduct", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "dotprod", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "dot_product_3d", "shared_reduction_adg")
        assert_component_references_resolve(evidence_dir, "dotprod")
        assert_component_references_resolve(evidence_dir, "dot_product_3d")
        assert_component_mapping_status(
            evidence_dir,
            "dot_product_3d",
            "g_t_dot_product_3d_0_0",
            "pass",
        )
        assert_component_mapping_status(evidence_dir, "dot_product_3d", "g_t_main_red_0_0", "pass")
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "dot_product_3d",
            {
                "arith.mulf#0.result0->llvm.intr.fmuladd#0.operand2",
                "dataflow.load#0.result0->llvm.intr.fmuladd#0.operand0",
                "dataflow.load#2.result0->llvm.intr.fmuladd#1.operand0",
                "dataflow.load#3.result0->llvm.intr.fmuladd#0.operand1",
                "dataflow.load#5.result0->llvm.intr.fmuladd#1.operand1",
                "llvm.intr.fmuladd#0.result0->llvm.intr.fmuladd#1.operand2",
                "llvm.intr.fmuladd#1.result0->dataflow.store#0.operand2",
            },
        )
        assert_mapping_hardware(evidence_dir, "vecsum-while", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "axpy", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "binary_search", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "bit_reverse", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "clz", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "ctz", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "downsample", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "delta_encode", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "delta_decode", "shared_reduction_adg")
        assert_mapping_uses_switch_multihop(evidence_dir, "delta_decode")
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "delta_encode",
            {
                "arith.subi#0.result0->dataflow.store#0.operand2",
                "dataflow.load#0.result0->arith.subi#0.operand0",
                "llvm.load#0.result0->arith.subi#0.operand1",
            },
        )
        assert_mapping_hardware(evidence_dir, "find_first_set", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "spmv", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "spmspv", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "byte_swap", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "xor_block", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "vecmul", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "vecscale", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "cumsum", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum_inclusive", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum_exclusive", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "pack_bits", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "parity", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "unpack_bits", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "popcount", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "mean", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "newton_iter", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "runge_kutta_step", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "vecnorm_l1", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "vecnorm_l2", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "gemv", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "matvec", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "downsample_avg", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "vecadd", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "conv1d", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "convolve_1d_same", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "crc32", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "gemm", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "matmul", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "mat3x3_mult", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "modmul", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "variance", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "covariance", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "correlation", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "autocorrelation", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "fir_filter", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "fir_filter_stateful", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "gather", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "lower_bound", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "moving_avg", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "outer", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "compare_swap", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "compact", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "hash_mix", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "string_hash", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "merge", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "partition", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "scatter_add", "shared_reduction_adg")
        assert_component_references_resolve(evidence_dir, "partition")
        assert_component_mapping_status(
            evidence_dir,
            "partition",
            "g_t_partition_red_0_0",
            "pass",
        )
        assert_component_mapping_status(
            evidence_dir,
            "partition",
            "g_t_partition_red_1_0",
            "pass",
        )
        assert_mapping_hardware(evidence_dir, "convolve_1d", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "relu", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "rotate_bits", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "sbox_lookup", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "transpose", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "upper_bound", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "upsample", "shared_reduction_adg")
        for case in MAPPING_FAILED_SWEEP_CASES:
            assert_mapping_hardware(evidence_dir, case, "shared_reduction_adg")
        for case in MAPPING_BLOCKED_SWEEP_CASES:
            assert_mapping_hardware(evidence_dir, case, "shared_reduction_adg")
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "compare_swap",
            {
                "arith.select#1.result0->dataflow.store#1.operand2",
                "dataflow.load#1.result0->arith.cmpf#0.operand1",
                "dataflow.store#1.result0->dataflow.sync#0.operand3",
            },
        )
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "compact",
            {
                "arith.cmpi#0.result0->dataflow.demux#0.operand0",
                "arith.cmpi#0.result0->dataflow.demux#1.operand0",
                "arith.cmpi#0.result0->dataflow.demux#2.operand0",
                "arith.cmpi#0.result0->dataflow.mux#0.operand0",
                "dataflow.demux#2.result1->dataflow.mux#0.operand2",
                "dataflow.store#0.result0->dataflow.mux#0.operand1",
            },
        )
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "partition",
            {
                "arith.cmpf#0.result0->dataflow.demux#0.operand0",
                "arith.cmpf#0.result0->dataflow.mux#0.operand0",
                "dataflow.demux#2.result1->dataflow.mux#0.operand2",
                "dataflow.demux#2.result1->dataflow.store#0.operand3",
                "dataflow.store#0.result0->dataflow.mux#0.operand1",
                "dataflow.store#0.result0->dataflow.mux#0.operand2",
            },
        )
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "hash_mix",
            {
                "dataflow.load#0.result0->arith.addi#0.operand1",
                "dataflow.load#1.result0->arith.addi#0.operand0",
                "arith.addi#0.result0->llvm.intr.fshl#0.operand0",
                "arith.addi#0.result0->llvm.intr.fshl#0.operand1",
                "llvm.intr.fshl#0.result0->arith.xori#0.operand0",
                "dataflow.load#1.result0->arith.xori#0.operand1",
                "arith.xori#0.result0->arith.muli#0.operand0",
                "arith.muli#0.result0->llvm.intr.fshl#1.operand0",
                "arith.muli#0.result0->llvm.intr.fshl#1.operand1",
                "llvm.intr.fshl#1.result0->dataflow.store#0.operand2",
            },
        )
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "string_hash",
            {
                "arith.addi#0.result0->arith.remui#0.operand0",
                "arith.remui#0.result0->dataflow.carry#0.operand2",
                "arith.shli#0.result0->arith.addi#0.operand1",
                "dataflow.load#0.result0->arith.addi#0.operand0",
            },
        )
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "bit_reverse",
            {
                "arith.ori#0.result0->dataflow.carry#0.operand2",
                "arith.shli#0.result0->arith.ori#0.operand0",
                "arith.shrui#0.result0->dataflow.carry#1.operand2",
                "dataflow.carry#1.result0->arith.andi#0.operand0",
                "dataflow.carry#1.result0->arith.shrui#0.operand0",
                "dataflow.invariant#0.result0->arith.shrui#0.operand1",
                "dataflow.stream#0.result1->dataflow.carry#1.operand0",
            },
        )
        assert_mapping_uses_switch_multihop(evidence_dir, "byte_swap")
        assert_mapping_uses_switch_multihop(evidence_dir, "xor_block")
        assert_mapping_uses_switch_multihop(evidence_dir, "vecmul")
        assert_mapping_uses_switch_multihop(evidence_dir, "vecscale")
        assert_mapping_uses_switch_multihop(evidence_dir, "axpy")
        assert_mapping_uses_switch_multihop(evidence_dir, "dotproduct")
        assert_mapping_uses_switch_multihop(evidence_dir, "dotprod")
        assert_mapping_uses_switch_multihop(evidence_dir, "vecsum-while")
        assert_mapping_uses_switch_multihop(evidence_dir, "spmv")
        assert_mapping_uses_switch_multihop(evidence_dir, "gemv")
        assert_mapping_uses_switch_multihop(evidence_dir, "gemm")
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "gemm",
            {
                "arith.shrui#0.result0->dataflow.load#1.operand1",
                "dataflow.stream#0.result0->arith.shli#0.operand0",
            },
        )
        assert_mapping_uses_switch_multihop(evidence_dir, "matmul")
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "mat3x3_mult",
            {
                "arith.muli#0.result0->arith.shrui#0.operand0",
                "arith.shrui#0.result0->dataflow.load#1.operand1",
            },
        )
        assert_mapping_uses_switch_multihop(evidence_dir, "matvec")
        assert_mapping_uses_switch_multihop(evidence_dir, "downsample")
        assert_mapping_uses_switch_multihop(evidence_dir, "downsample_avg")
        assert_mapping_uses_switch_multihop(evidence_dir, "correlation")
        assert_mapping_uses_switch_multihop(evidence_dir, "covariance")
        assert_mapping_uses_switch_multihop(evidence_dir, "upsample")
        assert_mapping_uses_switch_multihop(evidence_dir, "convolve_1d")
        assert_mapping_uses_switch_multihop(evidence_dir, "relu")
        assert_mapping_uses_switch_multihop(evidence_dir, "sbox_lookup")
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "sbox_lookup",
            {
                "arith.andi#0.result0->dataflow.load#1.operand1",
                "dataflow.load#0.result0->arith.andi#0.operand0",
                "dataflow.load#1.result0->dataflow.store#0.operand2",
            },
        )
        assert_mapping_uses_switch_multihop(evidence_dir, "rotate_bits")
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "runge_kutta_step",
            {
                "arith.addf#0.result0->llvm.intr.fmuladd#2.operand1",
                "dataflow.load#0.result0->llvm.intr.fmuladd#0.operand2",
                "dataflow.load#1.result0->llvm.intr.fmuladd#0.operand0",
                "dataflow.load#4.result0->llvm.intr.fmuladd#2.operand2",
                "dataflow.store#0.result0->dataflow.sync#0.operand5",
                "llvm.intr.fmuladd#1.result0->arith.addf#0.operand0",
                "llvm.intr.fmuladd#2.result0->dataflow.store#0.operand2",
            },
        )
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "rotate_bits",
            {
                "arith.andi#0.result0->arith.cmpi#0.operand0",
                "arith.cmpi#0.result0->arith.select#0.operand0",
                "arith.select#0.result0->dataflow.store#0.operand2",
                "dataflow.load#0.result0->llvm.intr.fshl#0.operand2",
                "dataflow.load#1.result0->arith.select#0.operand1",
                "dataflow.load#1.result0->llvm.intr.fshl#0.operand0",
                "dataflow.load#1.result0->llvm.intr.fshl#0.operand1",
                "llvm.intr.fshl#0.result0->arith.select#0.operand2",
            },
        )
        assert_component_references_resolve(evidence_dir, "vecadd")
        assert_component_references_resolve(evidence_dir, "variance")
        assert_component_references_resolve(evidence_dir, "covariance")

        status_csv = out_dir / "cgra-status-summary.csv"
        status_json = out_dir / "cgra-status-summary.json"
        legacy_root = repo / "temp/old_implementation_loom/loom/tests/app"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(status_csv),
                "--json-output",
                str(status_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--sim-evidence-dir",
                str(evidence_dir),
            ],
        )
        rows = read_rows(status_csv)
        for case in (
            "vecsum",
            "vecsum-while",
            "reduction",
            "dotproduct",
            "dotprod",
            "dot_product_3d",
            "spmv",
            "axpy",
            "bit_reverse",
            "byte_swap",
            "downsample",
            "xor_block",
            "vecmul",
            "prefix_sum",
            "cumsum",
            "prefix_sum_inclusive",
            "prefix_sum_exclusive",
            "partition",
            "mean",
            "vecnorm_l1",
            "vecnorm_l2",
            "gemv",
            "gemm",
            "matmul",
            "mat3x3_mult",
            "matvec",
            "downsample_avg",
            "vecadd",
            "vecscale",
            "conv1d",
            "variance",
            "covariance",
            "integrate_trapz",
            "delta_encode",
            "delta_decode",
            "correlation",
            "convolve_1d",
            "convolve_1d_same",
            "fir_filter_stateful",
            "compare_swap",
            "compact",
            "hash_mix",
            "string_hash",
            "modmul",
            "relu",
            "upsample",
            "sbox_lookup",
            "rotate_bits",
            "runge_kutta_step",
        ):
            assert_promoted_row(repo, rows, case)
        for case in MAPPING_FAILED_SWEEP_CASES:
            assert_structured_blocker_row(repo, rows, case, "fail", "fail")
        for case in MAPPING_BLOCKED_SWEEP_CASES:
            assert_structured_blocker_row(repo, rows, case, "blocked", "blocked")
        for case in DFG_BLOCKED_SWEEP_CASES:
            assert_dfg_blocked_row(repo, rows, case)
        for case in DFG_UNSUPPORTED_SWEEP_CASES:
            assert_dfg_unsupported_row(repo, rows, case)
        for case in app_manifest_no_dfg_cases(repo):
            assert_app_no_dfg_tier_blocked_row(rows, case)
        dotproduct_row = one_row(rows, "dotproduct")
        if dotproduct_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"dotproduct should use shared reduction hardware: {dotproduct_row}")
        dotprod_row = one_row(rows, "dotprod")
        if dotprod_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"dotprod should use shared reduction hardware: {dotprod_row}")
        spmv_row = one_row(rows, "spmv")
        if spmv_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"spmv should use shared reduction hardware: {spmv_row}")
        axpy_row = one_row(rows, "axpy")
        if axpy_row["hardware_system"] != "shared_vector_alu_adg":
            raise AssertionError(f"axpy should use shared vector hardware: {axpy_row}")
        byte_swap_row = one_row(rows, "byte_swap")
        if byte_swap_row["hardware_system"] != "shared_vector_alu_adg":
            raise AssertionError(f"byte_swap should use shared vector hardware: {byte_swap_row}")
        downsample_row = one_row(rows, "downsample")
        if downsample_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"downsample should use shared reduction hardware: {downsample_row}")
        xor_block_row = one_row(rows, "xor_block")
        if xor_block_row["hardware_system"] != "shared_vector_alu_adg":
            raise AssertionError(f"xor_block should use shared vector hardware: {xor_block_row}")
        vecmul_row = one_row(rows, "vecmul")
        if vecmul_row["hardware_system"] != "shared_vector_alu_adg":
            raise AssertionError(f"vecmul should use shared vector hardware: {vecmul_row}")
        vecscale_row = one_row(rows, "vecscale")
        if vecscale_row["hardware_system"] != "shared_vector_alu_adg":
            raise AssertionError(f"vecscale should use shared vector hardware: {vecscale_row}")
        prefix_sum_row = one_row(rows, "prefix_sum")
        if prefix_sum_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"prefix_sum should use shared reduction hardware: {prefix_sum_row}")
        cumsum_row = one_row(rows, "cumsum")
        if cumsum_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"cumsum should use shared reduction hardware: {cumsum_row}")
        prefix_sum_inclusive_row = one_row(rows, "prefix_sum_inclusive")
        if prefix_sum_inclusive_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(
                f"prefix_sum_inclusive should use shared reduction hardware: {prefix_sum_inclusive_row}"
            )
        mean_row = one_row(rows, "mean")
        if mean_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"mean should use shared reduction hardware: {mean_row}")
        vecnorm_l1_row = one_row(rows, "vecnorm_l1")
        if vecnorm_l1_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"vecnorm_l1 should use shared reduction hardware: {vecnorm_l1_row}")
        vecnorm_l2_row = one_row(rows, "vecnorm_l2")
        if vecnorm_l2_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"vecnorm_l2 should use shared reduction hardware: {vecnorm_l2_row}")
        gemv_row = one_row(rows, "gemv")
        if gemv_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"gemv should use shared reduction hardware: {gemv_row}")
        gemm_row = one_row(rows, "gemm")
        if gemm_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"gemm should use shared reduction hardware: {gemm_row}")
        matmul_row = one_row(rows, "matmul")
        if matmul_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"matmul should use shared reduction hardware: {matmul_row}")
        newton_iter_row = one_row(rows, "newton_iter")
        if newton_iter_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"newton_iter should use shared reduction hardware: {newton_iter_row}")
        matvec_row = one_row(rows, "matvec")
        if matvec_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"matvec should use shared reduction hardware: {matvec_row}")
        covariance_row = one_row(rows, "covariance")
        if covariance_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"covariance should use shared reduction hardware: {covariance_row}")
        downsample_row = one_row(rows, "downsample_avg")
        if downsample_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"downsample_avg should use shared reduction hardware: {downsample_row}")
        counts = json.loads(status_json.read_text())["counts"]["app"]
        expected_counts = {
            "total": 109,
            "pass": 53,
            "fail": 0,
            "blocked": 56,
            "unsupported": 0,
            "missing_status": 0,
        }
        if counts != expected_counts:
            raise AssertionError(f"app counter shape should reflect promoted app coverage: {counts}")
        sim_cycle = out_dir / "sim-cycle-summary.csv"
        sim_args = [
            "bash",
            "test/app/run_sim_cycle_summary.sh",
            "--output",
            str(sim_cycle),
        ]
        for report in sorted(evidence_dir.glob("*.dfg.report.json")):
            sim_args.extend(["--dfg-report", str(report)])
        for report in sorted(evidence_dir.glob("*.cgra.report.json")):
            sim_args.extend(["--cgra-report", str(report)])
        run(repo, sim_args)
        audit_json = out_dir / "artifact-audit-summary.json"
        audit_args = [
            "python3",
            "test/e2e/audit_intermediate_artifacts.py",
            "--output",
            str(audit_json),
            str(status_csv),
            str(sim_cycle),
        ]
        audit_args.extend(str(path) for path in sorted(evidence_dir.glob("*.json")))
        run(repo, audit_args)
        audit = json.loads(audit_json.read_text())
        if audit.get("verdict") != "pass":
            raise AssertionError(f"sweep evidence should pass artifact audit: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
