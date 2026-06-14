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


DEFAULT_SWEEP_CASES = (
    "autocorrelation",
    "vecsum",
    "vecsum-while",
    "dotproduct",
    "dot_product_3d",
    "axpy",
    "bit_reverse",
    "downsample",
    "downsample_avg",
    "delta_encode",
    "prefix_sum",
    "cumsum",
    "prefix_sum_inclusive",
    "prefix_sum_exclusive",
    "pack_bits",
    "unpack_bits",
    "integrate_trapz",
    "reduction",
    "mean",
    "vecnorm_l1",
    "vecnorm_l2",
    "correlation",
    "compare_swap",
    "hash_mix",
    "spmv",
    "convolve_1d",
    "conv1d",
    "convolve_1d_same",
    "crc32",
    "fir_filter",
    "gemv",
    "gemm",
    "matvec",
    "byte_swap",
    "xor_block",
    "relu",
    "rotate_bits",
    "vecadd",
    "vecmul",
    "vecscale",
    "variance",
)
BLOCKED_SWEEP_CASES = (
    "bit_reverse",
    "compare_swap",
    "dot_product_3d",
    "gemm",
    "gemv",
    "hash_mix",
    "integrate_trapz",
    "relu",
    "rotate_bits",
)
DFG_UNSUPPORTED_SWEEP_CASES = (
    "autocorrelation",
    "convolve_1d_same",
    "crc32",
    "delta_encode",
    "fir_filter",
    "pack_bits",
    "prefix_sum_exclusive",
    "unpack_bits",
)

HEADER = [
    "suite",
    "case",
    "source_row",
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


def one_row(rows: list[dict[str, str]], case: str) -> dict[str, str]:
    matches = [row for row in rows if row["suite"] == "app" and row["case"] == case]
    if len(matches) != 1:
        raise AssertionError(f"expected one app/{case} row, got {matches}")
    return matches[0]


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


def assert_dfg_dynamic_work_items(evidence_dir: Path, case: str, expected_count: int) -> None:
    path = evidence_dir / f"{case}.dfg.report.json"
    data = json.loads(path.read_text())
    if data.get("dynamic_work_items") != expected_count:
        raise AssertionError(f"{case} should have {expected_count} dynamic work items: {path}: {data}")


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


def assert_unsupported_operation(evidence_dir: Path, case: str, operation: str) -> None:
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    mapping_path = evidence_dir / f"{case}.mapping.json"
    dfg = json.loads(dfg_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    expected_dfg = f"unsupported op: {operation}"
    expected_mapping = f"unsupported PnR graph operation: {operation}"
    if expected_dfg not in dfg.get("diagnostics", []):
        raise AssertionError(f"{case} DFG unsupported diagnostic should be {expected_dfg}: {dfg_path}: {dfg}")
    if expected_mapping not in mapping.get("diagnostics", []):
        raise AssertionError(
            f"{case} mapping unsupported diagnostic should be {expected_mapping}: {mapping_path}: {mapping}"
        )


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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <repo>")
    repo = Path(argv[1]).resolve()
    assert_default_sweep_cases(repo / "test/e2e/run_cgra_sim_evidence_sweep.sh")
    with artifact_test_common.repo_temp_dir(repo, "cgra-sim-evidence-sweep-") as tmp:
        out_dir = Path(tmp)
        evidence_dir = out_dir / "current-sim-cycle"
        run(
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
                "dot_product_3d",
                "--case",
                "axpy",
                "--case",
                "bit_reverse",
                "--case",
                "downsample",
                "--case",
                "delta_encode",
                "--case",
                "spmv",
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
                "hash_mix",
                "--case",
                "prefix_sum",
                "--case",
                "cumsum",
                "--case",
                "prefix_sum_inclusive",
                "--case",
                "prefix_sum_exclusive",
                "--case",
                "pack_bits",
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
                "gemm",
                "--case",
                "correlation",
                "--case",
                "convolve_1d",
                "--case",
                "relu",
                "--case",
                "rotate_bits",
            ],
        )
        for case in (
            "vecsum",
            "vecsum-while",
            "reduction",
            "dotproduct",
            "spmv",
            "axpy",
            "byte_swap",
            "downsample",
            "xor_block",
            "vecmul",
            "vecscale",
            "prefix_sum",
            "cumsum",
            "prefix_sum_inclusive",
            "mean",
            "vecnorm_l1",
            "vecnorm_l2",
            "matvec",
            "downsample_avg",
            "vecadd",
            "conv1d",
            "variance",
            "correlation",
            "convolve_1d",
        ):
            assert_sweep_artifact(evidence_dir, case, "dfg.report.json")
            assert_sweep_artifact(evidence_dir, case, "mapping.json")
            assert_sweep_artifact(evidence_dir, case, "cgra.report.json")
            assert_comparison_artifact(evidence_dir, case, "pass")
        for case in BLOCKED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "pass")
            expected_mapping_status = "blocked" if case in {"dot_product_3d", "relu"} else "fail"
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", expected_mapping_status)
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        assert_dfg_dynamic_work_items(evidence_dir, "gemm", 8)
        assert_mapping_unrouted_edges(
            evidence_dir,
            "gemm",
            {
                "arith.shrui#0.result0->dataflow.load#1.operand1",
                "dataflow.invariant#0.result0->arith.shli#0.operand1",
                "dataflow.stream#0.result0->arith.shli#0.operand0",
            },
        )
        for case in DFG_UNSUPPORTED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "unsupported")
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", "unsupported")
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        for case in ("autocorrelation", "convolve_1d_same", "crc32", "fir_filter"):
            assert_unsupported_operation(evidence_dir, case, "scf.for")
        assert_unsupported_operation(evidence_dir, "delta_encode", "llvm.getelementptr")
        assert_dfg_unsupported_operation(evidence_dir, "delta_encode", "llvm.load")
        assert_mapping_hardware(evidence_dir, "dotproduct", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "dot_product_3d", "shared_reduction_adg")
        assert_component_references_resolve(evidence_dir, "dot_product_3d")
        assert_component_mapping_status(
            evidence_dir,
            "dot_product_3d",
            "g_t_dot_product_3d_0_0",
            "fail",
            "missing hardware resource for software op llvm.intr.fmuladd",
        )
        assert_component_mapping_status(evidence_dir, "dot_product_3d", "g_t_main_red_0_0", "pass")
        assert_mapping_hardware(evidence_dir, "vecsum-while", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "axpy", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "bit_reverse", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "downsample", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "delta_encode", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "spmv", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "byte_swap", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "xor_block", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "vecmul", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "vecscale", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "cumsum", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum_inclusive", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum_exclusive", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "pack_bits", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "unpack_bits", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "mean", "shared_reduction_adg")
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
        assert_mapping_hardware(evidence_dir, "variance", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "correlation", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "autocorrelation", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "fir_filter", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "compare_swap", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "hash_mix", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "convolve_1d", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "rotate_bits", "shared_reduction_adg")
        for case in BLOCKED_SWEEP_CASES:
            assert_mapping_hardware(evidence_dir, case, "shared_reduction_adg")
        assert_mapping_uses_switch_multihop(evidence_dir, "byte_swap")
        assert_mapping_uses_switch_multihop(evidence_dir, "xor_block")
        assert_mapping_uses_switch_multihop(evidence_dir, "vecmul")
        assert_mapping_uses_switch_multihop(evidence_dir, "vecscale")
        assert_mapping_uses_switch_multihop(evidence_dir, "axpy")
        assert_mapping_uses_switch_multihop(evidence_dir, "dotproduct")
        assert_mapping_uses_switch_multihop(evidence_dir, "vecsum-while")
        assert_mapping_uses_switch_multihop(evidence_dir, "spmv")
        assert_mapping_uses_switch_multihop(evidence_dir, "matvec")
        assert_mapping_uses_switch_multihop(evidence_dir, "downsample")
        assert_mapping_uses_switch_multihop(evidence_dir, "downsample_avg")
        assert_mapping_uses_switch_multihop(evidence_dir, "correlation")
        assert_mapping_uses_switch_multihop(evidence_dir, "convolve_1d")
        assert_mapping_uses_switch_multihop(evidence_dir, "relu")
        assert_component_references_resolve(evidence_dir, "vecadd")
        assert_component_references_resolve(evidence_dir, "variance")

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
            "spmv",
            "axpy",
            "byte_swap",
            "downsample",
            "xor_block",
            "vecmul",
            "prefix_sum",
            "cumsum",
            "prefix_sum_inclusive",
            "mean",
            "vecnorm_l1",
            "vecnorm_l2",
            "matvec",
            "downsample_avg",
            "vecadd",
            "vecscale",
            "conv1d",
            "variance",
            "correlation",
            "convolve_1d",
        ):
            assert_promoted_row(repo, rows, case)
        for case in BLOCKED_SWEEP_CASES:
            expected_status = "blocked" if case == "relu" else "fail"
            if case == "dot_product_3d":
                expected_status = "blocked"
            expected_mapping_status = "blocked" if case in {"dot_product_3d", "relu"} else "fail"
            assert_structured_blocker_row(repo, rows, case, expected_status, expected_mapping_status)
        for case in DFG_UNSUPPORTED_SWEEP_CASES:
            assert_dfg_unsupported_row(repo, rows, case)
        dotproduct_row = one_row(rows, "dotproduct")
        if dotproduct_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"dotproduct should use shared reduction hardware: {dotproduct_row}")
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
        matvec_row = one_row(rows, "matvec")
        if matvec_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"matvec should use shared reduction hardware: {matvec_row}")
        downsample_row = one_row(rows, "downsample_avg")
        if downsample_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"downsample_avg should use shared reduction hardware: {downsample_row}")
        counts = json.loads(status_json.read_text())["counts"]["app"]
        expected_counts = {
            "total": 109,
            "pass": 24,
            "fail": 7,
            "blocked": 10,
            "unsupported": 0,
            "missing_status": 68,
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
