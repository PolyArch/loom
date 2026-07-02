#!/usr/bin/env python3
"""Regression test for batch CGRA-sim evidence production."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import artifact_test_common


APP_NO_DFG_TIER_COUNT = 0
DEFAULT_SWEEP_CASES = (
    "autocorrelation",
    "vecsum",
    "vecsum-while",
    "dotproduct",
    "dotprod",
    "dot_product_3d",
    "axpy",
    "batchnorm",
    "binary_search",
    "bitonic_stage",
    "bitonic_stage-tweak",
    "bit_reverse",
    "bisection_step",
    "clz",
    "ctz",
    "downsample",
    "downsample_avg",
    "delta_encode",
    "delta_decode",
    "find_first_set",
    "prefix_sum",
    "cumsum",
    "database_join",
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
    "compact_predicate",
    "hash_mix",
    "string_hash",
    "merge",
    "modexp",
    "modmul",
    "spmv",
    "sort_bubble",
    "bitrev",
    "bitrev_complex",
    "convolve_1d",
    "conv1d",
    "conv2d",
    "im2col",
    "convolve_1d_same",
    "crc32",
    "cross_product",
    "quat_mult",
    "fir_filter",
    "fir_filter_stateful",
    "gather",
    "gf_mul",
    "gemv",
    "gemm",
    "matmul",
    "mmtile",
    "mat3x3_mult",
    "spmspv",
    "stream_update",
    "lower_bound",
    "matvec",
    "moving_avg",
    "newton_iter",
    "outer",
    "byte_swap",
    "cdma",
    "pool_avg",
    "pool_max",
    "scatter_add",
    "edge_update",
    "edge_update_batch",
    "bitonic_stage-modified",
    "col2im",
    "hist_bin",
    "histogram",
    "histogram_strided",
    "quantile",
    "sort_insertion",
    "sort_merge",
    "sort_quick",
    "spmspm",
    "string_compare",
    "xor_block",
    "relu",
    "rotate_bits",
    "rle_decode",
    "rle_encode",
    "runge_kutta_step",
    "sbox_lookup",
    "sigmoid",
    "softmax",
    "window_blackman",
    "window_hamming",
    "window_hanning",
    "interpolate_linear",
    "jacobi_stencil_5pt",
    "jacobi_stencil_7pt",
    "distance_point",
    "line_intersect",
    "depthwise_conv",
    "edit_distance_step",
    "normalize",
    "normalize_vec3",
    "transpose",
    "transform_point",
    "upper_bound",
    "upsample",
    "upsample_linear",
    "vecadd",
    "vecmul",
    "vecscale",
    "variance",
)
MAPPING_FAILED_SWEEP_CASES: tuple[str, ...] = ()
MAPPING_BLOCKED_SWEEP_CASES: tuple[str, ...] = ()
MAPPING_UNSUPPORTED_SWEEP_CASES: tuple[str, ...] = ()
DFG_BLOCKED_SWEEP_CASES: tuple[str, ...] = ()
DFG_UNSUPPORTED_SWEEP_CASES = (
    "edge_update",
    "edge_update_batch",
    "col2im",
    "sort_insertion",
    "sort_merge",
    "sort_quick",
    "spmspm",
    "string_compare",
)
EMPTY_DISCOVERED_GRAPH_IDS = "__empty__"
PRIMARY_GRAPH_MISSING_SWEEP_CASES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "col2im",
        "col2im_kernel",
        "primary workload graph absent: col2im_kernel remains a residual call target outside "
        "the discovered dataflow graphs; no discovered graph ids were emitted, so DFG-sim cannot "
        "observe the kernel return value",
        EMPTY_DISCOVERED_GRAPH_IDS,
        "col2im_kernel",
    ),
    (
        "string_compare",
        "string_compare_kernel",
        "primary workload graph absent: string_compare_kernel remains a residual call target outside "
        "the discovered dataflow graphs; discovered graph ids include g_t_main_0_0,g_t_main_1_0,"
        "g_t_main_2_0, so DFG-sim cannot observe the kernel return value",
        "g_t_main_0_0",
        "string_compare_kernel",
    ),
)
GRAPH_PRESENT_UNWIRED_SWEEP_CASES = {
}
GRAPH_PRESENT_UNWIRED_DIAGNOSTIC = (
    "primary workload graph is present but app simulator fixture is not wired for search-style control flow"
)
PARTIAL_LOWERING_SWEEP_CASES = {
    "edge_update": (
        "primary workload graph is partial: edge_update lowering covers the input-to-output copy loop "
        "while the CSR lookup and update loop remains outside dataflow",
        "edge_update_kernel",
    ),
    "edge_update_batch": (
        "primary workload graph is partial: edge_update_batch lowering covers the input-to-output copy loop "
        "while the batched CSR lookup and update loops remain outside dataflow",
        "edge_update_batch_kernel",
    ),
    "sort_insertion": (
        "primary workload graph is partial: sort_insertion lowering covers the copy loop "
        "while the insertion-sort compare-and-shift loop remains outside dataflow",
        "sort_insertion_kernel",
    ),
    "sort_merge": (
        "primary workload graph is partial: sort_merge lowering covers copy and remainder-copy slices "
        "while the merge compare loop remains outside dataflow",
        "sort_merge_kernel",
    ),
    "sort_quick": (
        "primary workload graph is partial: sort_quick lowering covers copy and partition slices "
        "while iterative stack control remains outside dataflow",
        "sort_quick_kernel",
    ),
    "spmspm": (
        "primary workload graph is partial: spmspm lowering covers final nonzero compression "
        "while sparse multiply-accumulate loops remain outside dataflow",
        "spmspm_kernel",
    ),
}
MAPPING_FAILED_SWEEP_EVIDENCE: dict[str, dict[str, object]] = {}

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
                "cases": [{"case": "byte_swap", "hardware": "shared_vector_mesh_adg"}],
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
        (evidence / f"byte_swap.{suffix}").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": kind,
                    "workload": "byte_swap",
                    "hardware": "shared_vector_mesh_adg",
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
    if "sort_insertion" in no_dfg_cases:
        raise AssertionError(f"sort_insertion should keep its DFG tier: {no_dfg_cases}")
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


def assert_cgra_hardware(evidence_dir: Path, case: str, expected_hardware: str) -> None:
    path = evidence_dir / f"{case}.cgra.report.json"
    data = json.loads(path.read_text())
    if data.get("hardware") != expected_hardware:
        raise AssertionError(f"{case} CGRA evidence should use {expected_hardware}: {path}: {data}")


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
            if len(set(identities)) != len(identities):
                raise AssertionError(f"aggregate has duplicate component identities: {aggregate_path}: {field}: {identities}")
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


def raw_component_identity(case: str, identity: str) -> str:
    prefix = f"{case}."
    if identity.startswith(prefix):
        return identity[len(prefix):]
    return identity


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


def assert_cdma_evidence(evidence_dir: Path) -> None:
    expected_memory = {"arg2": [f"i32:{index * 3 + 7}" for index in range(32)]}
    expected_counts = {
        "dataflow.load": 32,
        "dataflow.store": 32,
        "dataflow.sync": 32,
    }

    dfg = json.loads((evidence_dir / "cdma.dfg.report.json").read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_cdma_candidate_0_0"
        or dfg.get("dynamic_work_items") != 32
        or dfg.get("optimistic_cycles") != 291
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state", {}).get("arg2") != expected_memory["arg2"]
        or len(set(dfg.get("final_memory_state", {}).get("arg2", []))) != 32
        or dfg.get("operation_fire_counts") != expected_counts
    ):
        raise AssertionError(f"cdma should preserve true DFG copy evidence: {dfg}")

    mapping = json.loads((evidence_dir / "cdma.mapping.json").read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("graph") != "g_t_cdma_candidate_0_0"
        or mapping.get("placed_records") != 3
        or mapping.get("routed_edges") != 3
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"cdma should route on shared reduction hardware: {mapping}")

    cgra = json.loads((evidence_dir / "cdma.cgra.report.json").read_text())
    comparison = json.loads((evidence_dir / "cdma.sim-comparison-report.json").read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_reduction_adg"
        or cgra.get("dfg_cycles") != 291
        or cgra.get("hardware_aware_cycles") != 314
        or cgra.get("routed_edges") != 3
        or cgra.get("route_segments") != 9
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state", {}).get("arg2") != expected_memory["arg2"]
        or comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
    ):
        raise AssertionError(f"cdma should preserve CGRA/comparison copy evidence: {cgra} {comparison}")


def assert_compact_predicate_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg4": ["i32:1", "i32:0", "i32:1", "i32:0", "i32:1", "i32:1", "i32:0", "i32:1"],
        "arg6": ["i32:10", "i32:20", "i32:30", "i32:40", "i32:50", "i32:60", "i32:70", "i32:80"],
        "arg7": ["i32:10", "i32:30", "i32:50", "i32:60", "i32:80", "i32:0", "i32:0", "i32:0"],
    }
    expected_counts = {
        "arith.addi": 5,
        "arith.cmpi": 8,
        "arith.index_cast": 18,
        "dataflow.load": 13,
        "dataflow.store": 5,
        "scf.if": 8,
    }

    dfg = json.loads((evidence_dir / "compact_predicate.dfg.report.json").read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_compact_predicate_candidate_red_0_0"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 140
        or dfg.get("final_outputs") != ["none", "i32:5"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("operation_fire_counts") != expected_counts
    ):
        raise AssertionError(f"compact_predicate should preserve true DFG predicate-compaction evidence: {dfg}")

    mapping = json.loads((evidence_dir / "compact_predicate.mapping.json").read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("graph") != "g_t_compact_predicate_candidate_red_0_0"
        or mapping.get("placed_records") != 7
        or mapping.get("routed_edges") != 4
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"compact_predicate should route on shared reduction hardware: {mapping}")
    assert_mapping_uses_switch_multihop(evidence_dir, "compact_predicate")

    cgra = json.loads((evidence_dir / "compact_predicate.cgra.report.json").read_text())
    comparison = json.loads((evidence_dir / "compact_predicate.sim-comparison-report.json").read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_reduction_adg"
        or cgra.get("dfg_cycles") != 140
        or cgra.get("hardware_aware_cycles") != 182
        or cgra.get("routed_edges") != 4
        or cgra.get("route_segments") != 20
        or cgra.get("final_outputs") != ["none", "i32:5"]
        or cgra.get("final_memory_state") != expected_memory
        or comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
    ):
        raise AssertionError(f"compact_predicate should preserve CGRA/comparison predicate evidence: {cgra} {comparison}")


def assert_scatter_add_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": [
            "i32:1",
            "i32:2",
            "i32:3",
            "i32:4",
            "i32:5",
            "i32:1",
            "i32:2",
            "i32:3",
            "i32:4",
            "i32:5",
            "i32:1",
            "i32:2",
            "i32:3",
            "i32:4",
            "i32:5",
            "i32:1",
        ],
        "arg2": [
            "i32:0",
            "i32:3",
            "i32:1",
            "i32:3",
            "i32:7",
            "i32:8",
            "i32:1",
            "i32:4",
            "i32:7",
            "i32:2",
            "i32:5",
            "i32:3",
            "i32:12",
            "i32:6",
            "i32:0",
            "i32:7",
        ],
        "arg3": ["i32:6", "i32:6", "i32:7", "i32:11", "i32:7", "i32:6", "i32:10", "i32:17"],
    }
    dfg = json.loads((evidence_dir / "scatter_add.dfg.report.json").read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_scatter_add_0"
        or dfg.get("dynamic_work_items") != 16
        or dfg.get("operation_fire_counts", {}).get("dataflow.load") != 44
        or dfg.get("operation_fire_counts", {}).get("dataflow.store") != 14
        or dfg.get("operation_fire_counts", {}).get("arith.cmpi") != 16
        or dfg.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"scatter_add should preserve true DFG memory evidence: {dfg}")

    mapping = json.loads((evidence_dir / "scatter_add.mapping.json").read_text())
    constant_placements = [
        placement
        for placement in mapping.get("placements", [])
        if isinstance(placement, dict) and placement.get("operation") == "dataflow.constant"
    ]
    constant_configs = [
        entry
        for entry in mapping.get("config_bitstream", [])
        if isinstance(entry, dict)
        and entry.get("register") == "sw_configs.const_hex_value"
    ]
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 9
        or mapping.get("routed_edges") != 9
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("resource_pressure", []) != []
        or len(constant_placements) != 1
    ):
        raise AssertionError(f"scatter_add should route through one real shared constant: {mapping}")
    constant_placement = constant_placements[0]
    expected_constant_config = {
        "source": f"placement:{constant_placement.get('software')}",
        "target": constant_placement.get("hardware"),
        "register": "sw_configs.const_hex_value",
        "value": "0x00000008",
    }
    if constant_configs != [expected_constant_config]:
        raise AssertionError(
            f"scatter_add constant config should follow the mapped placement: {mapping}"
        )

    cgra = json.loads((evidence_dir / "scatter_add.cgra.report.json").read_text())
    comparison = json.loads((evidence_dir / "scatter_add.sim-comparison-report.json").read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("final_memory_state") != expected_memory
        or comparison.get("status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
    ):
        raise AssertionError(f"scatter_add should preserve CGRA/comparison memory evidence: {cgra} {comparison}")


def assert_bitonic_stage_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": [
            "f32:1",
            "f32:3",
            "f32:2",
            "f32:4",
            "f32:8",
            "f32:6",
            "f32:7",
            "f32:5",
        ]
    }
    expected_counts = {
        "arith.cmpf": 8,
        "arith.cmpi": 4,
        "arith.index_cast": 16,
        "arith.select": 12,
        "arith.shli": 12,
        "arith.shrui": 8,
        "dataflow.constant": 21,
        "dataflow.load": 8,
        "dataflow.store": 8,
        "llvm.getelementptr": 8,
    }

    dfg = json.loads((evidence_dir / "bitonic_stage.dfg.report.json").read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_bitonic_stage_0"
        or dfg.get("dynamic_work_items") != 4
        or dfg.get("optimistic_cycles") != 195
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"bitonic_stage should preserve true DFG compare-swap evidence: {dfg}")
    assert_operation_fire_counts("bitonic_stage", dfg, expected_counts)

    mapping = json.loads((evidence_dir / "bitonic_stage.mapping.json").read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 23
        or mapping.get("routed_edges") != 29
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 734
    ):
        raise AssertionError(f"bitonic_stage should route on shared memory reduction hardware: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_edges = {
        "arith.cmpi#0.result0->arith.select#0.operand0",
        "dataflow.constant#1.result0->arith.index_cast#0.operand0",
        "dataflow.constant#1.result0->arith.index_cast#1.operand0",
        "arith.index_cast#0.result0->arith.shli#1.operand1",
        "arith.index_cast#1.result0->arith.shli#2.operand1",
        "arith.shrui#0.result0->dataflow.load#0.operand1",
        "arith.shrui#1.result0->dataflow.store#0.operand1",
        "arith.select#1.result0->dataflow.store#0.operand2",
        "arith.select#2.result0->dataflow.store#1.operand2",
    }
    if not expected_edges.issubset(route_edges):
        raise AssertionError(f"bitonic_stage mapping should expose predicate and address routes: {mapping}")
    placeholder_segments = [
        segment
        for route in mapping.get("routes", [])
        if isinstance(route, dict)
        for segment in route.get("segments", [])
        if isinstance(segment, dict)
        and any(
            isinstance(endpoint, str)
            and (endpoint.endswith(".out") or endpoint.endswith(".in"))
            for endpoint in (segment.get("from"), segment.get("to"))
        )
    ]
    if placeholder_segments:
        raise AssertionError(f"bitonic_stage routes should use real fabric endpoints: {mapping}")
    index_cast_sites = [
        placement.get("hardware")
        for placement in mapping.get("placements", [])
        if isinstance(placement, dict) and placement.get("operation") == "arith.index_cast"
    ]
    if len(index_cast_sites) != 2 or not all(
        isinstance(site, str) and site.startswith("shared_memory_reduction_adg::fabric.op#")
        for site in index_cast_sites
    ):
        raise AssertionError(
            f"bitonic_stage should place both shifted-address index casts: {mapping}"
        )

    cgra = json.loads((evidence_dir / "bitonic_stage.cgra.report.json").read_text())
    comparison = json.loads((evidence_dir / "bitonic_stage.sim-comparison-report.json").read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("dfg_cycles") != 195
        or cgra.get("hardware_aware_cycles") != 372
        or cgra.get("routed_edges") != 29
        or cgra.get("route_segments") != 135
        or cgra.get("config_records") != 734
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra.get("fidelity_level") != "mapping_constraint_estimate"
        or comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
    ):
        raise AssertionError(f"bitonic_stage should preserve CGRA/comparison evidence: {cgra} {comparison}")


def assert_bitonic_stage_modified_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": [
            "f32:1",
            "f32:2",
            "f32:2",
            "f32:3",
            "f32:128",
            "f32:94",
            "f32:112",
            "f32:79",
        ]
    }
    expected_counts = {
        "arith.addf": 4,
        "arith.addi": 11,
        "arith.andi": 17,
        "arith.cmpf": 8,
        "arith.cmpi": 21,
        "arith.index_cast": 44,
        "arith.mulf": 16,
        "arith.select": 4,
        "arith.shli": 9,
        "arith.shrui": 4,
        "arith.xori": 1,
        "dataflow.constant": 7,
        "dataflow.load": 28,
        "dataflow.store": 24,
        "llvm.trunc": 20,
        "llvm.zext": 1,
        "scf.forall": 4,
        "scf.if": 17,
    }

    dfg = json.loads((evidence_dir / "bitonic_stage-modified.dfg.report.json").read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_bitonic_stage_modified_kernel_0"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 486
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(
            f"bitonic_stage-modified should preserve true DFG memory evidence: {dfg}"
        )
    assert_operation_fire_counts("bitonic_stage-modified", dfg, expected_counts)

    mapping = json.loads((evidence_dir / "bitonic_stage-modified.mapping.json").read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 44
        or mapping.get("routed_edges") != 50
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 1360
    ):
        raise AssertionError(
            f"bitonic_stage-modified should route on shared memory reduction hardware: {mapping}"
        )
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_edges = {
        "arith.addf#0.result0->dataflow.store#3.operand2",
        "arith.mulf#0.result0->dataflow.store#2.operand2",
        "arith.index_cast#0.result0->dataflow.load#0.operand1",
        "arith.index_cast#1.result0->dataflow.store#0.operand1",
        "arith.index_cast#2.result0->dataflow.load#3.operand1",
        "arith.index_cast#3.result0->dataflow.store#3.operand1",
        "arith.cmpf#0.result0->arith.select#0.operand1",
        "arith.cmpf#1.result0->arith.select#0.operand2",
        "arith.cmpi#1.result0->arith.select#0.operand0",
    }
    if not expected_edges.issubset(route_edges):
        raise AssertionError(
            f"bitonic_stage-modified mapping should expose compare, multiply, and memory routes: {mapping}"
        )
    assert_mapping_edges_use_switch_multihop(
        evidence_dir,
        "bitonic_stage-modified",
        {
            "arith.index_cast#2.result0->dataflow.load#3.operand1",
            "arith.index_cast#3.result0->dataflow.store#3.operand1",
        },
    )

    cgra = json.loads((evidence_dir / "bitonic_stage-modified.cgra.report.json").read_text())
    comparison = json.loads(
        (evidence_dir / "bitonic_stage-modified.sim-comparison-report.json").read_text()
    )
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("dfg_cycles") != 486
        or cgra.get("hardware_aware_cycles") != 822
        or cgra.get("routed_edges") != 50
        or cgra.get("route_segments") != 252
        or cgra.get("config_records") != 1360
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra.get("fidelity_level") != "mapping_constraint_estimate"
        or comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("cgra_sim_cycles") != 822
        or comparison.get("dfg_sim_cycles") != 486
    ):
        raise AssertionError(
            f"bitonic_stage-modified should preserve CGRA/comparison evidence: {cgra} {comparison}"
        )


def assert_bitonic_stage_tweak_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": [
            "f32:1",
            "f32:2",
            "f32:2",
            "f32:3",
            "f32:8",
            "f32:5",
            "f32:7",
            "f32:4",
        ]
    }
    expected_counts = {
        "arith.addf": 12,
        "arith.addi": 11,
        "arith.andi": 17,
        "arith.cmpf": 8,
        "arith.cmpi": 21,
        "arith.index_cast": 48,
        "arith.select": 4,
        "arith.shli": 9,
        "arith.xori": 1,
        "dataflow.constant": 7,
        "dataflow.load": 20,
        "dataflow.store": 16,
        "llvm.trunc": 20,
        "llvm.zext": 1,
        "scf.if": 17,
    }

    dfg = json.loads((evidence_dir / "bitonic_stage-tweak.dfg.report.json").read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_bitonic_stage_tweak_kernel_0"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 407
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"bitonic_stage-tweak should preserve true DFG compare-swap evidence: {dfg}")
    assert_operation_fire_counts("bitonic_stage-tweak", dfg, expected_counts)

    mapping = json.loads((evidence_dir / "bitonic_stage-tweak.mapping.json").read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 45
        or mapping.get("routed_edges") != 51
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 1403
    ):
        raise AssertionError(f"bitonic_stage-tweak should route on shared memory reduction hardware: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_edges = {
        "arith.index_cast#4.result0->dataflow.load#3.operand1",
        "arith.index_cast#5.result0->dataflow.store#3.operand1",
        "arith.addf#0.result0->dataflow.store#2.operand2",
        "arith.addf#1.result0->dataflow.store#3.operand2",
        "dataflow.load#0.result0->arith.cmpf#0.operand0",
        "dataflow.load#1.result0->arith.cmpf#0.operand1",
    }
    if not expected_edges.issubset(route_edges):
        raise AssertionError(f"bitonic_stage-tweak mapping should expose address and data routes: {mapping}")
    assert_mapping_edges_use_switch_multihop(
        evidence_dir,
        "bitonic_stage-tweak",
        {
            "arith.index_cast#4.result0->dataflow.load#3.operand1",
            "arith.index_cast#5.result0->dataflow.store#3.operand1",
        },
    )

    cgra = json.loads((evidence_dir / "bitonic_stage-tweak.cgra.report.json").read_text())
    comparison = json.loads((evidence_dir / "bitonic_stage-tweak.sim-comparison-report.json").read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("dfg_cycles") != 407
        or cgra.get("hardware_aware_cycles") != 752
        or cgra.get("routed_edges") != 51
        or cgra.get("route_segments") != 261
        or cgra.get("config_records") != 1403
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra.get("fidelity_level") != "mapping_constraint_estimate"
        or comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("cgra_sim_cycles") != 752
        or comparison.get("dfg_sim_cycles") != 407
    ):
        raise AssertionError(f"bitonic_stage-tweak should preserve CGRA/comparison evidence: {cgra} {comparison}")


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
    if expected_mapping_status == "unsupported":
        expected_diagnostic_class = "mapping_artifact_unsupported"
    elif expected_status == "blocked":
        expected_diagnostic_class = "mapping_artifact_blocked"
    else:
        expected_diagnostic_class = "mapping_artifact_failed"
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


def assert_quantile_evidence(evidence_dir: Path) -> None:
    dfg = json.loads((evidence_dir / "quantile.dfg.report.json").read_text())
    mapping = json.loads((evidence_dir / "quantile.mapping.json").read_text())
    cgra = json.loads((evidence_dir / "quantile.cgra.report.json").read_text())
    comparison = json.loads((evidence_dir / "quantile.sim-comparison-report.json").read_text())
    expected_outputs = ["none", "f32:511.500000"]
    expected_counts = {
        "arith.addi": 3,
        "arith.cmpi": 1,
        "arith.index_cast": 3,
        "arith.mulf": 2,
        "arith.subf": 2,
        "dataflow.constant": 3,
        "dataflow.load": 2,
        "llvm.fptoui": 1,
        "llvm.intr.fmuladd": 1,
        "llvm.uitofp": 2,
        "scf.if": 1,
    }
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_quantile_kernel_0"
        or dfg.get("dynamic_work_items") != 1
        or dfg.get("final_outputs") != expected_outputs
    ):
        raise AssertionError(f"quantile should preserve scalar-return DFG evidence: {dfg}")
    assert_operation_fire_counts("quantile", dfg, expected_counts)
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_signal_window_adg"
        or mapping.get("placed_records") != 19
        or mapping.get("routed_edges") != 23
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"quantile should map to shared signal-window hardware: {mapping}")
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_signal_window_adg"
        or cgra.get("final_outputs") != expected_outputs
        or cgra.get("dfg_cycles") != dfg.get("optimistic_cycles")
        or cgra.get("hardware_aware_cycles", 0) < cgra.get("dfg_cycles", 0)
        or comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("cgra_sim_cycles", 0) < comparison.get("dfg_sim_cycles", 0)
    ):
        raise AssertionError(f"quantile should preserve CGRA/comparison evidence: {cgra} {comparison}")


def assert_operation_fire_counts(case: str, dfg: dict, expected_counts: dict[str, int]) -> None:
    actual_counts = dfg.get("operation_fire_counts", {})
    for op_name, expected in expected_counts.items():
        actual = actual_counts.get(op_name)
        if actual != expected:
            raise AssertionError(f"{case} {op_name} fire count should be {expected}, got {actual}: {dfg}")


def assert_mapping_failed_evidence(evidence_dir: Path, case: str) -> None:
    expected = MAPPING_FAILED_SWEEP_EVIDENCE[case]
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    mapping_path = evidence_dir / f"{case}.mapping.json"
    dfg = json.loads(dfg_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    if not any(expected["diagnostic"] in diagnostic for diagnostic in mapping.get("diagnostics", [])):
        raise AssertionError(f"{case} should expose the expected mapping failure: {mapping_path}: {mapping}")
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != expected["graph"]
        or dfg.get("dynamic_work_items") != expected["dynamic_work_items"]
        or dfg.get("final_outputs") != expected["final_outputs"]
    ):
        raise AssertionError(f"{case} should preserve real DFG evidence before mapping failure: {dfg_path}: {dfg}")
    for argument, values in expected["final_memory_state"].items():
        actual = dfg.get("final_memory_state", {}).get(argument)
        if actual != values:
            raise AssertionError(f"{case} final memory {argument} should be {values}, got {actual}: {dfg}")
    assert_operation_fire_counts(case, dfg, expected["operation_fire_counts"])


def assert_pack_bits_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg11": ["i32:-749385939"],
        "arg8": [
            "i32:1",
            "i32:0",
            "i32:1",
            "i32:1",
            "i32:0",
            "i32:1",
            "i32:0",
            "i32:0",
            "i32:1",
            "i32:1",
            "i32:1",
            "i32:0",
            "i32:0",
            "i32:0",
            "i32:1",
            "i32:0",
            "i32:1",
            "i32:0",
            "i32:1",
            "i32:0",
            "i32:1",
            "i32:0",
            "i32:1",
            "i32:0",
            "i32:1",
            "i32:1",
            "i32:0",
            "i32:0",
            "i32:1",
            "i32:0",
            "i32:1",
            "i32:1",
        ],
    }
    expected_outputs = ["none", "i64:32"]
    expected_counts = {
        "arith.addi": 2,
        "arith.andi": 32,
        "arith.cmpi": 33,
        "arith.index_cast": 33,
        "dataflow.load": 32,
        "dataflow.store": 1,
        "arith.ori": 32,
        "arith.select": 32,
        "arith.shli": 33,
        "arith.subi": 32,
        "llvm.intr.umin": 1,
        "llvm.trunc": 33,
        "llvm.zext": 1,
        "scf.if": 1,
    }
    dfg = json.loads((evidence_dir / "pack_bits.dfg.report.json").read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 32
        or dfg.get("event_count") != 298
        or dfg.get("final_outputs") != expected_outputs
        or dfg.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"pack_bits should preserve real packed-bit DFG evidence: {dfg}")
    assert_operation_fire_counts("pack_bits", dfg, expected_counts)

    mapping = json.loads((evidence_dir / "pack_bits.mapping.json").read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("placed_records") != 18
        or mapping.get("routed_edges") != 16
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 471
    ):
        raise AssertionError(f"pack_bits should map computed address casts on shared ADG: {mapping}")
    index_cast_sites = {
        str(placement.get("hardware"))
        for placement in mapping.get("placements", [])
        if isinstance(placement, dict) and placement.get("operation") == "arith.index_cast"
    }
    if index_cast_sites != {
        "shared_reduction_adg::fabric.op#97",
        "shared_reduction_adg::fabric.op#98",
    }:
        raise AssertionError(f"pack_bits should place both computed address casts on shared ADG: {mapping}")
    route_edges = {
        str(route.get("edge_ref"))
        for route in mapping.get("routes", [])
        if isinstance(route, dict)
    }
    expected_route_edges = {
        "arith.addi#0.result0->llvm.intr.umin#0.operand0",
        "arith.andi#0.result0->arith.cmpi#1.operand0",
        "arith.cmpi#1.result0->arith.select#0.operand0",
        "arith.index_cast#0.result0->dataflow.load#0.operand1",
        "arith.index_cast#1.result0->dataflow.store#0.operand1",
        "arith.ori#0.result0->dataflow.store#0.operand2",
        "arith.select#0.result0->arith.ori#0.operand0",
        "arith.shli#0.result0->arith.cmpi#0.operand0",
        "arith.shli#0.result0->arith.subi#0.operand1",
        "arith.shli#0.result0->llvm.trunc#0.operand0",
        "arith.shli#1.result0->arith.select#0.operand2",
        "arith.subi#0.result0->llvm.trunc#1.operand0",
        "dataflow.load#0.result0->arith.andi#0.operand0",
        "llvm.intr.umin#0.result0->llvm.zext#0.operand0",
        "llvm.trunc#0.result0->arith.addi#0.operand0",
        "llvm.trunc#1.result0->arith.shli#1.operand1",
    }
    if route_edges != expected_route_edges:
        raise AssertionError(f"pack_bits should expose all packed-bit route edges: {mapping}")

    cgra = json.loads((evidence_dir / "pack_bits.cgra.report.json").read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 445
        or cgra.get("hardware_aware_cycles") != 564
        or cgra.get("placed_records") != 18
        or cgra.get("routed_edges") != 16
        or cgra.get("route_segments") != 86
        or cgra.get("config_records") != 471
        or cgra.get("width_adapter_latency_cycles") != 3
        or cgra.get("final_outputs") != expected_outputs
        or cgra.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"pack_bits should preserve real CGRA evidence: {cgra}")

    comparison = json.loads((evidence_dir / "pack_bits.sim-comparison-report.json").read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 445
        or comparison.get("cgra_sim_cycles") != 564
    ):
        raise AssertionError(f"pack_bits comparison should pass with real final-state checks: {comparison}")


def signed_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return value - 0x100000000
    return value


def assert_unpack_bits_evidence(evidence_dir: Path) -> None:
    packed_words = [0xAAAAAAAA, 0x13579BDF, 0x80000001, 0x0000000F]
    expected_bits = [
        f"i32:{(packed_words[bit // 32] >> (bit % 32)) & 1}"
        for bit in range(100)
    ]
    expected_memory = {
        "arg4": [f"i32:{signed_i32(word)}" for word in packed_words],
        "arg10": expected_bits,
    }
    expected_outputs = ["none", "i64:128"]
    expected_counts = {
        "dataflow.load": 4,
        "dataflow.store": 100,
        "arith.shrui": 100,
        "arith.andi": 100,
        "llvm.intr.umin": 4,
        "scf.forall": 4,
    }
    dfg = json.loads((evidence_dir / "unpack_bits.dfg.report.json").read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("event_count", 0) <= 0
        or dfg.get("final_outputs") != expected_outputs
        or dfg.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unpack_bits should preserve real unpacked-bit DFG evidence: {dfg}")
    assert_operation_fire_counts("unpack_bits", dfg, expected_counts)
    cgra = json.loads((evidence_dir / "unpack_bits.cgra.report.json").read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware_aware_cycles", 0) < dfg.get("optimistic_cycles", 0)
        or cgra.get("final_outputs") != expected_outputs
        or cgra.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unpack_bits should preserve real CGRA evidence: {cgra}")
    comparison = json.loads((evidence_dir / "unpack_bits.sim-comparison-report.json").read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
    ):
        raise AssertionError(f"unpack_bits comparison should pass on real final state: {comparison}")


def bit_scan_input_values(case: str) -> list[int]:
    values = []
    for i in range(32):
        if case == "clz":
            seeds = [0, 0x80000000, 0x40000000, 0x20000000, 1, 0xFFFFFFFF, 0x00FF00FF, 0x01000000]
            raw = seeds[i] if i < len(seeds) else i * 0x0012345
        elif case == "ctz":
            seeds = [0, 1, 2, 0x80000000, 0xFFFFFFFF, 0x00010000, 0x01000000, 8]
            raw = seeds[i] if i < len(seeds) else i * 0x00005678
        elif case == "find_first_set":
            seeds = [0, 1, 2, 4, 0x80000000, 0xFFFFFFFF, 0xFFFFFFF0, 0x00000100]
            raw = seeds[i] if i < len(seeds) else i * 0x00008765
        elif case == "parity":
            if i == 0:
                raw = 0
            elif i == 1:
                raw = 1
            elif i == 2:
                raw = 3
            elif i == 3:
                raw = 7
            else:
                raw = 0x9ABCDEF0 * i
        else:
            raise AssertionError(f"unknown bit-scan case: {case}")
        values.append(signed_i32(raw))
    return values


def clz32(value: int) -> int:
    raw = value & 0xFFFFFFFF
    if raw == 0:
        return 32
    return 32 - raw.bit_length()


def ctz32(value: int) -> int:
    raw = value & 0xFFFFFFFF
    if raw == 0:
        return 32
    return (raw & -raw).bit_length() - 1


def bit_scan_expected_output(case: str, values: list[int]) -> list[int]:
    if case == "clz":
        return [clz32(value) for value in values]
    if case == "ctz":
        return [ctz32(value) for value in values]
    if case == "find_first_set":
        return [0 if (value & 0xFFFFFFFF) == 0 else ctz32(value) + 1 for value in values]
    if case == "parity":
        return [int.bit_count(value & 0xFFFFFFFF) & 1 for value in values]
    raise AssertionError(f"unknown bit-scan case: {case}")


def assert_bit_scan_evidence(
    evidence_dir: Path,
    case: str,
    *,
    graph: str,
    output_arg: str,
    event_count: int,
    dfg_cycles: int,
    cgra_cycles: int,
    placed_records: int,
    routed_edges: int,
    config_records: int,
    route_segments: int,
    operation_fire_counts: dict[str, int],
    expected_route_edges: set[str],
) -> None:
    values = bit_scan_input_values(case)
    expected_memory = {
        "arg1": [f"i32:{value}" for value in values],
        output_arg: [f"i32:{value}" for value in bit_scan_expected_output(case, values)],
    }
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != graph
        or dfg.get("dynamic_work_items") != 32
        or dfg.get("event_count") != event_count
        or dfg.get("optimistic_cycles") != dfg_cycles
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG should execute the real bit-scan loop: {dfg_path}: {dfg}")
    assert_operation_fire_counts(case, dfg, operation_fire_counts)

    mapping_path = evidence_dir / f"{case}.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != placed_records
        or mapping.get("routed_edges") != routed_edges
        or mapping.get("unrouted_edges") != 0
        or mapping.get("config_records") != config_records
        or route_edges != expected_route_edges
    ):
        raise AssertionError(f"{case} should route real bit-scan dataflow on a shared ADG: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / f"{case}.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("dfg_cycles") != dfg_cycles
        or cgra.get("hardware_aware_cycles") != cgra_cycles
        or cgra.get("route_segments") != route_segments
        or cgra.get("placed_records") != placed_records
        or cgra.get("routed_edges") != routed_edges
        or cgra.get("config_records") != config_records
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"{case} CGRA evidence should preserve real final memory: {cgra_path}: {cgra}")


BOUND_SEARCH_FIRE_COUNTS = {
    "arith.addi": 84,
    "arith.cmpf": 28,
    "arith.cmpi": 28,
    "arith.index_cast": 84,
    "arith.select": 56,
    "arith.shrui": 56,
    "arith.subi": 56,
    "dataflow.load": 36,
    "dataflow.store": 8,
    "dataflow.sync": 8,
    "scf.while": 8,
}


BOUND_SEARCH_COMMON_ROUTE_EDGES = {
    "arith.addi#0.result0->arith.addi#2.operand0",
    "arith.addi#1.result0->dataflow.load#1.operand1",
    "arith.cmpf#0.result0->arith.select#0.operand0",
    "arith.cmpf#0.result0->arith.select#1.operand0",
    "arith.select#0.result0->arith.cmpi#0.operand0",
    "arith.select#0.result0->dataflow.store#0.operand2",
    "arith.select#1.result0->arith.cmpi#0.operand1",
    "arith.shrui#0.result0->arith.addi#0.operand0",
    "arith.shrui#1.result0->arith.addi#1.operand0",
    "arith.subi#0.result0->arith.shrui#0.operand0",
    "arith.subi#1.result0->arith.shrui#1.operand0",
    "dataflow.load#0.result0->arith.cmpf#0.operand1",
    "dataflow.load#0.result1->dataflow.sync#0.operand0",
    "dataflow.load#1.result0->arith.cmpf#0.operand0",
    "dataflow.store#0.result0->dataflow.sync#0.operand1",
}


def assert_bound_search_evidence(
    evidence_dir: Path,
    case: str,
    *,
    graph: str,
    expected_output: list[str],
    case_route_edges: set[str],
) -> None:
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != graph
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("event_count") != 452
        or dfg.get("optimistic_cycles") != 651
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state", {}).get("arg6") != expected_output
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG should execute real bound-search inputs: {dfg_path}: {dfg}")
    assert_operation_fire_counts(case, dfg, BOUND_SEARCH_FIRE_COUNTS)

    mapping_path = evidence_dir / f"{case}.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    routes = mapping.get("routes", [])
    route_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    route_segments = sum(len(route.get("segments", [])) for route in routes if isinstance(route, dict))
    expected_edges = BOUND_SEARCH_COMMON_ROUTE_EDGES | case_route_edges
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 15
        or mapping.get("routed_edges") != 17
        or mapping.get("unrouted_edges") != 0
        or mapping.get("config_records") != 430
        or route_segments != 77
        or route_edges != expected_edges
    ):
        raise AssertionError(f"{case} should route real bound-search dataflow on shared memory ADG: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / f"{case}.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("dfg_cycles") != 651
        or cgra.get("hardware_aware_cycles") != 756
        or cgra.get("route_segments") != 77
        or cgra.get("placed_records") != 15
        or cgra.get("routed_edges") != 17
        or cgra.get("config_records") != 430
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state", {}).get("arg6") != expected_output
    ):
        raise AssertionError(f"{case} CGRA evidence should preserve real bound-search state: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / f"{case}.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 651
        or comparison.get("cgra_sim_cycles") != 756
    ):
        raise AssertionError(f"{case} comparison should pass with real final-state checks: {comparison_path}: {comparison}")


def assert_binary_search_evidence(evidence_dir: Path) -> None:
    case = "binary_search"
    graph = "g_t__ZN12_GLOBAL__N_123binary_search_candidateEPKfS1_Pjjj_0_0"
    expected_output = [
        "i32:3",
        "i32:-1",
        "i32:7",
        "i32:-1",
        "i32:0",
    ]

    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != graph
        or dfg.get("dynamic_work_items") != 5
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state", {}).get("arg8") != expected_output
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG should execute real binary-search inputs: {dfg_path}: {dfg}")
    assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addi": 48,
            "arith.cmpf": 32,
            "arith.cmpi": 18,
            "arith.extui": 16,
            "arith.index_cast": 16,
            "arith.select": 48,
            "arith.shrui": 16,
            "arith.subi": 16,
            "arith.trunci": 18,
            "dataflow.load": 21,
            "dataflow.mux": 32,
            "dataflow.store": 5,
            "dataflow.sync": 5,
            "llvm.intr.smax": 5,
            "llvm.sext": 16,
            "scf.if": 18,
            "scf.while": 5,
        },
    )

    mapping_path = evidence_dir / f"{case}.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 22
        or mapping.get("routed_edges") != 27
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 679
    ):
        raise AssertionError(f"{case} should route on shared memory ADG: {mapping_path}: {mapping}")
    index_cast_sites = [
        str(placement.get("hardware"))
        for placement in mapping.get("placements", [])
        if isinstance(placement, dict) and placement.get("operation") == "arith.index_cast"
    ]
    if (
        len(index_cast_sites) != 1
        or not index_cast_sites[0].startswith("shared_memory_reduction_adg::fabric.op#")
    ):
        raise AssertionError(f"{case} should place its computed load address cast on shared memory ADG: {mapping_path}: {mapping}")
    route_edges = {
        str(route.get("edge_ref"))
        for route in mapping.get("routes", [])
        if isinstance(route, dict)
    }
    expected_route_edges = {
        "arith.addi#0.result0->arith.addi#1.operand0",
        "arith.addi#0.result0->arith.addi#2.operand0",
        "arith.addi#0.result0->arith.select#0.operand2",
        "arith.addi#0.result0->llvm.sext#0.operand0",
        "arith.addi#1.result0->arith.select#1.operand1",
        "arith.addi#2.result0->arith.select#2.operand2",
        "arith.cmpf#0.result0->arith.extui#0.operand0",
        "arith.cmpf#0.result0->arith.select#0.operand0",
        "arith.cmpf#0.result0->dataflow.mux#0.operand0",
        "arith.cmpf#0.result0->dataflow.mux#1.operand0",
        "arith.cmpf#1.result0->arith.select#1.operand0",
        "arith.cmpf#1.result0->arith.select#2.operand0",
        "arith.extui#0.result0->arith.trunci#0.operand0",
        "arith.index_cast#0.result0->dataflow.load#1.operand1",
        "arith.select#0.result0->llvm.intr.smax#0.operand0",
        "arith.select#1.result0->dataflow.mux#0.operand2",
        "arith.select#2.result0->dataflow.mux#1.operand2",
        "arith.shrui#0.result0->arith.addi#0.operand0",
        "arith.subi#0.result0->arith.shrui#0.operand0",
        "dataflow.load#0.result0->arith.cmpf#0.operand1",
        "dataflow.load#0.result0->arith.cmpf#1.operand1",
        "dataflow.load#0.result1->dataflow.sync#0.operand0",
        "dataflow.load#1.result0->arith.cmpf#0.operand0",
        "dataflow.load#1.result0->arith.cmpf#1.operand0",
        "dataflow.store#0.result0->dataflow.sync#0.operand1",
        "llvm.intr.smax#0.result0->dataflow.store#0.operand2",
        "llvm.sext#0.result0->arith.index_cast#0.operand0",
    }
    if route_edges != expected_route_edges:
        raise AssertionError(f"{case} should expose all binary-search route edges: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / f"{case}.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("dfg_cycles") != 510
        or cgra.get("hardware_aware_cycles") != 671
        or cgra.get("placed_records") != 22
        or cgra.get("routed_edges") != 27
        or cgra.get("route_segments") != 125
        or cgra.get("config_records") != 679
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state", {}).get("arg8") != expected_output
    ):
        raise AssertionError(f"{case} CGRA evidence should preserve real binary-search state: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / f"{case}.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 510
        or comparison.get("cgra_sim_cycles") != 671
    ):
        raise AssertionError(f"{case} comparison should pass with real final-state checks: {comparison_path}: {comparison}")


def popcount_input_values() -> list[int]:
    seeds = [0, 1, 2, 3, 7, 15, 0xFFFFFFFF, 0x80000000]
    values = []
    for i in range(32):
        raw = seeds[i] if i < len(seeds) else (i * 0x12345678 + (i << 16))
        values.append(signed_i32(raw))
    return values


def assert_popcount_evidence(evidence_dir: Path) -> None:
    expected_output = [f"i32:{int.bit_count(value & 0xFFFFFFFF)}" for value in popcount_input_values()]
    expected_memory = {
        "arg1": [f"i32:{value}" for value in popcount_input_values()],
        "arg4": expected_output,
    }
    expected_counts = {
        "arith.addi": 826,
        "arith.andi": 826,
        "arith.cmpi": 858,
        "arith.shrui": 826,
        "dataflow.load": 32,
        "dataflow.store": 32,
        "dataflow.sync": 32,
        "scf.if": 32,
    }
    dfg_path = evidence_dir / "popcount.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t__ZN12_GLOBAL__N_118popcount_candidateEPKjPjj_0_0"
        or dfg.get("dynamic_work_items") != 32
        or dfg.get("event_count") != 3464
        or dfg.get("optimistic_cycles") != 3664
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"popcount DFG should execute the real candidate loop: {dfg_path}: {dfg}")
    assert_operation_fire_counts("popcount", dfg, expected_counts)

    mapping_path = evidence_dir / "popcount.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    expected_edges = {
        "arith.addi#0.result0->dataflow.store#0.operand2",
        "arith.andi#0.result0->arith.addi#0.operand0",
        "arith.shrui#0.result0->arith.cmpi#1.operand0",
        "dataflow.load#0.result0->arith.cmpi#0.operand0",
        "dataflow.load#0.result1->dataflow.sync#0.operand0",
        "dataflow.store#0.result0->dataflow.sync#0.operand1",
    }
    actual_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 8
        or mapping.get("routed_edges") != 6
        or mapping.get("unrouted_edges") != 0
        or mapping.get("config_records") != 153
        or actual_edges != expected_edges
    ):
        raise AssertionError(f"popcount should route real bitcount dataflow on a shared ADG: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "popcount.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("dfg_cycles") != 3664
        or cgra.get("hardware_aware_cycles") != 3707
        or cgra.get("route_segments") != 24
        or cgra.get("placed_records") != 8
        or cgra.get("routed_edges") != 6
        or cgra.get("config_records") != 153
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"popcount CGRA evidence should preserve real final memory: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / "popcount.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 3664
        or comparison.get("cgra_sim_cycles") != 3707
    ):
        raise AssertionError(f"popcount comparison should pass with real final-state checks: {comparison_path}: {comparison}")


def f32_token(value: float) -> str:
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        return "f32:-0"
    if math.floor(value) == value:
        return f"f32:{int(value)}"
    return f"f32:{value:.6f}"


def dot_product_3d_values() -> list[str]:
    values = []
    for i in range(16):
        lhs0 = i + 1
        lhs1 = (i % 5) - 2
        lhs2 = (i % 3) + 1
        values.append(f32_token(2 * lhs0 - 3 * lhs1 + 4 * lhs2))
    return values


def assert_dot_product_3d_evidence(evidence_dir: Path) -> None:
    expected_products = dot_product_3d_values()
    expected_outputs = ["none", "none", "f32:402"]

    dfg_path = evidence_dir / "dot_product_3d.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    memory = dfg.get("final_memory_state", {})
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 32
        or dfg.get("final_outputs") != expected_outputs
        or memory.get("g_t_dot_product_3d_0_0:arg6") != expected_products
        or memory.get("g_t_main_red_0_0:arg4") != expected_products
    ):
        raise AssertionError(f"dot_product_3d DFG evidence should carry all real product lanes: {dfg_path}: {dfg}")
    assert_operation_fire_counts(
        "dot_product_3d",
        dfg,
        {
            "dataflow.load": 112,
            "dataflow.store": 16,
            "llvm.intr.fmuladd": 32,
            "arith.mulf": 16,
            "dataflow.sync": 32,
        },
    )
    component_identities = dfg.get("component_dfg_sim_report_identities", [])
    components = [json.loads((evidence_dir / f"{identity}.json").read_text()) for identity in component_identities]
    by_graph = {component.get("graph"): component for component in components}
    core = by_graph.get("g_t_dot_product_3d_0_0")
    reduction = by_graph.get("g_t_main_red_0_0")
    if (
        not isinstance(core, dict)
        or not isinstance(reduction, dict)
        or core.get("diagnostics") != []
        or reduction.get("diagnostics") != []
        or core.get("final_memory_state", {}).get("arg6") != expected_products
        or reduction.get("final_memory_state", {}).get("arg4") != expected_products
    ):
        raise AssertionError(f"dot_product_3d component reports should carry real product lanes: {dfg_path}: {dfg}")

    cgra_path = evidence_dir / "dot_product_3d.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("final_outputs") != expected_outputs
        or cgra.get("final_memory_state", {}).get("g_t_dot_product_3d_0_0:arg6") != expected_products
        or cgra.get("final_memory_state", {}).get("g_t_main_red_0_0:arg4") != expected_products
        or cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
    ):
        raise AssertionError(f"dot_product_3d CGRA evidence should carry all real product lanes: {cgra_path}: {cgra}")


def cross_product_lhs_values() -> list[str]:
    values = []
    for i in range(64):
        value = 1.0 + i * 0.1
        values.extend((f32_token(value), "f32:0", "f32:0"))
    return values


def cross_product_rhs_values() -> list[str]:
    values = []
    for i in range(64):
        value = 1.0 + i * 0.1
        values.extend(("f32:0", f32_token(value), "f32:0"))
    return values


def cross_product_output_values() -> list[str]:
    values = []
    for i in range(64):
        value = 1.0 + i * 0.1
        values.extend(("f32:0", "f32:0", f32_token(value * value)))
    return values


def quat_mult_lhs_values() -> list[str]:
    values = []
    for i in range(16):
        values.extend(
            (
                f32_token(1.0 + i * 0.01),
                f32_token(0.1 + i * 0.03),
                f32_token(-0.2 + i * 0.02),
                f32_token(0.05 + i * 0.025),
            )
        )
    return values


def quat_mult_rhs_values() -> list[str]:
    values = []
    for i in range(16):
        values.extend(
            (
                f32_token(0.8 - i * 0.005),
                f32_token(-0.1 + i * 0.01),
                f32_token(0.2 + i * 0.015),
                f32_token(-0.3 + i * 0.02),
            )
        )
    return values


def quat_mult_output_values() -> list[str]:
    values = []
    for i in range(16):
        w1 = 1.0 + i * 0.01
        x1 = 0.1 + i * 0.03
        y1 = -0.2 + i * 0.02
        z1 = 0.05 + i * 0.025
        w2 = 0.8 - i * 0.005
        x2 = -0.1 + i * 0.01
        y2 = 0.2 + i * 0.015
        z2 = -0.3 + i * 0.02
        values.extend(
            (
                f32_token(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2),
                f32_token(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2),
                f32_token(w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2),
                f32_token(w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2),
            )
        )
    return values


def assert_cross_product_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg2": cross_product_lhs_values(),
        "arg5": cross_product_rhs_values(),
        "arg6": cross_product_output_values(),
    }
    dfg_path = evidence_dir / "cross_product.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 64
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"cross_product DFG evidence should write all real output lanes: {dfg_path}: {dfg}")
    assert_operation_fire_counts(
        "cross_product",
        dfg,
        {
            "dataflow.load": 384,
            "dataflow.store": 192,
            "llvm.fneg": 192,
            "arith.mulf": 192,
            "llvm.intr.fmuladd": 192,
            "dataflow.sync": 64,
        },
    )

    mapping_path = evidence_dir / "cross_product.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_vector_math_adg"
        or mapping.get("placed_records") != 25
        or mapping.get("routed_edges") != 44
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"cross_product should map onto the shared vector math ADG: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "cross_product.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_vector_math_adg"
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"cross_product CGRA evidence should preserve the DFG final state: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / "cross_product.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
    ):
        raise AssertionError(f"cross_product comparison should pass with real final state: {comparison_path}: {comparison}")


def assert_quat_mult_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": quat_mult_lhs_values(),
        "arg2": quat_mult_rhs_values(),
        "arg3": quat_mult_output_values(),
    }
    dfg_path = evidence_dir / "quat_mult.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 16
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"quat_mult DFG evidence should write all real output lanes: {dfg_path}: {dfg}")
    assert_operation_fire_counts(
        "quat_mult",
        dfg,
        {
            "dataflow.load": 128,
            "dataflow.store": 64,
            "llvm.fneg": 64,
            "arith.mulf": 64,
            "llvm.intr.fmuladd": 192,
            "arith.shli": 64,
            "arith.ori": 48,
        },
    )

    mapping_path = evidence_dir / "quat_mult.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_vector_math_adg"
        or mapping.get("routed_edges", 0) <= 0
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"quat_mult should map onto the shared vector math ADG: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "quat_mult.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_vector_math_adg"
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"quat_mult CGRA evidence should preserve the DFG final state: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / "quat_mult.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
    ):
        raise AssertionError(f"quat_mult comparison should pass with real final state: {comparison_path}: {comparison}")


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
        or dfg.get("optimistic_cycles") != 82
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
        or cgra.get("dfg_cycles") != 82
        or cgra.get("hardware_aware_cycles") != 149
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
        or dfg.get("optimistic_cycles") != 110
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
        or cgra.get("dfg_cycles") != 110
        or cgra.get("hardware_aware_cycles") != 186
        or cgra.get("routed_edges") != 14
        or cgra.get("route_segments") != 54
        or cgra.get("final_outputs") != ["none", "f32:0.835938"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"mat3x3_mult CGRA evidence should carry the first real matrix dot state: {cgra_path}: {cgra}")


def softmax_input_values() -> list[float]:
    return [float(index % 20) - 10.0 for index in range(128)]


def parse_float_token(value: str) -> float:
    prefix, raw = value.split(":", 1)
    if prefix != "f32":
        raise AssertionError(f"expected f32 token, got {value!r}")
    return float(raw)


def assert_float_tokens_close(values: list[str], expected: list[float], *, label: str) -> None:
    if len(values) != len(expected):
        raise AssertionError(f"{label} length mismatch: expected {len(expected)}, got {len(values)}")
    for index, (actual_token, expected_value) in enumerate(zip(values, expected)):
        actual = parse_float_token(actual_token)
        if not math.isclose(actual, expected_value, rel_tol=1.0e-5, abs_tol=1.0e-6):
            raise AssertionError(
                f"{label}[{index}] should be close to {expected_value}, got {actual_token}"
            )


def assert_im2col_evidence(evidence_dir: Path) -> None:
    expected = [
        1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0,
        2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0,
        5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0,
        6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0,
    ]
    dfg = json.loads((evidence_dir / "im2col.dfg.report.json").read_text())
    mapping = json.loads((evidence_dir / "im2col.mapping.json").read_text())
    cgra = json.loads((evidence_dir / "im2col.cgra.report.json").read_text())
    comparison = json.loads((evidence_dir / "im2col.sim-comparison-report.json").read_text())
    if dfg.get("graph") != "g_t_im2col_kernel_0_0" or dfg.get("status") != "pass":
        raise AssertionError(f"im2col DFG report should pass the kernel graph: {dfg}")
    if dfg.get("dynamic_work_items") != 3:
        raise AssertionError(f"im2col should expose three dynamic outer work items: {dfg}")
    assert_operation_fire_counts(
        "im2col",
        dfg,
        {
            "dataflow.load": 36,
            "dataflow.store": 36,
        },
    )
    assert_float_tokens_close(
        dfg.get("final_memory_state", {}).get("arg11", []),
        expected,
        label="im2col DFG output",
    )
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("routed_edges") != 19
        or mapping.get("unrouted_edges") != 0
    ):
        raise AssertionError(f"im2col mapping should be fully routed on shared memory hardware: {mapping}")
    if cgra.get("status") != "pass" or cgra.get("hardware") != "shared_memory_reduction_adg":
        raise AssertionError(f"im2col CGRA report should pass on shared memory hardware: {cgra}")
    assert_float_tokens_close(
        cgra.get("final_memory_state", {}).get("arg11", []),
        expected,
        label="im2col CGRA output",
    )
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("cgra_sim_cycles", 0) < comparison.get("dfg_sim_cycles", 0)
    ):
        raise AssertionError(f"im2col comparison should pass final-state checks: {comparison}")


def assert_softmax_evidence(evidence_dir: Path) -> None:
    expected_graphs = [
        "g_t_softmax_kernel_red_0_0",
        "g_t_softmax_kernel_red_1_0",
        "g_t_softmax_kernel_0_0",
    ]
    source = softmax_input_values()
    max_value = max(source)
    exp_values = [math.exp(value - max_value) for value in source]
    exp_sum = sum(exp_values)
    normalized = [value / exp_sum for value in exp_values]

    dfg_path = evidence_dir / "softmax.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("component_graphs") != expected_graphs
        or dfg.get("dynamic_work_items") != 383
        or dfg.get("diagnostics")
        != ["derived workload graph-set DFG report from component DFG simulator reports"]
    ):
        raise AssertionError(f"softmax DFG aggregate should carry three source-derived components: {dfg_path}: {dfg}")
    assert_operation_fire_counts(
        "softmax",
        dfg,
        {
            "arith.divf": 128,
            "dataflow.store": 256,
            "math.exp": 128,
        },
    )
    memory = dfg.get("final_memory_state", {})
    assert_float_tokens_close(
        memory.get("g_t_softmax_kernel_red_1_0:arg6", []),
        exp_values,
        label="softmax exp buffer",
    )
    normalized_tokens = memory.get("g_t_softmax_kernel_0_0:arg1", [])
    assert_float_tokens_close(normalized_tokens, normalized, label="softmax normalized output")
    if not math.isclose(
        sum(parse_float_token(value) for value in normalized_tokens),
        1.0,
        rel_tol=1.0e-5,
        abs_tol=1.0e-5,
    ):
        raise AssertionError(f"softmax normalized output should sum to one: {normalized_tokens}")

    mapping_path = evidence_dir / "softmax.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_signal_window_adg"
        or mapping.get("component_graphs") != expected_graphs
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("placed_records", 0) < 19
        or mapping.get("routed_edges", 0) < 26
    ):
        raise AssertionError(f"softmax should aggregate three passing shared-signal mappings: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "softmax.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_signal_window_adg"
        or cgra.get("component_graphs") != expected_graphs
        or cgra.get("dfg_cycles", 0) < dfg.get("optimistic_cycles", 0)
        or cgra.get("hardware_aware_cycles", 0) < cgra.get("dfg_cycles", 0)
        or cgra.get("functional_state_source")
        != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
    ):
        raise AssertionError(f"softmax CGRA aggregate should carry source-derived normalized state: {cgra_path}: {cgra}")
    assert_float_tokens_close(
        cgra.get("final_memory_state", {}).get("g_t_softmax_kernel_0_0:arg1", []),
        normalized,
        label="softmax CGRA normalized output",
    )


def sigmoid_input_values() -> list[float]:
    return [(float(index) / 1024.0 - 0.5) * 10.0 for index in range(1024)]


def assert_sigmoid_evidence(evidence_dir: Path) -> None:
    source = sigmoid_input_values()
    expected = [1.0 / (1.0 + math.exp(-value)) for value in source]

    dfg_path = evidence_dir / "sigmoid.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_sigmoid_kernel_0_0"
        or dfg.get("dynamic_work_items") != 1024
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"sigmoid DFG evidence should cover the real kernel loop: {dfg_path}: {dfg}")
    assert_operation_fire_counts(
        "sigmoid",
        dfg,
        {
            "dataflow.load": 1024,
            "math.exp": 1024,
            "arith.addf": 1024,
            "arith.divf": 1024,
            "dataflow.store": 1024,
        },
    )
    memory = dfg.get("final_memory_state", {})
    assert_float_tokens_close(memory.get("arg1", []), source, label="sigmoid input")
    output_tokens = memory.get("arg3", [])
    assert_float_tokens_close(output_tokens, expected, label="sigmoid output")
    if not (
        parse_float_token(output_tokens[0]) < 0.01
        and 0.49 < parse_float_token(output_tokens[512]) < 0.51
        and parse_float_token(output_tokens[-1]) > 0.99
    ):
        raise AssertionError(f"sigmoid output should expose nontrivial activation shape: {output_tokens[:3]} ...")

    mapping_path = evidence_dir / "sigmoid.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_signal_window_adg"
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("placed_records", 0) < 6
        or mapping.get("routed_edges", 0) < 5
    ):
        raise AssertionError(f"sigmoid should map to the shared signal-window ADG: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "sigmoid.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_signal_window_adg"
        or cgra.get("dfg_cycles", 0) < dfg.get("optimistic_cycles", 0)
        or cgra.get("hardware_aware_cycles", 0) < cgra.get("dfg_cycles", 0)
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"sigmoid CGRA evidence should carry DFG final state: {cgra_path}: {cgra}")
    assert_float_tokens_close(
        cgra.get("final_memory_state", {}).get("arg3", []),
        expected,
        label="sigmoid CGRA output",
    )


def assert_mmtile_evidence(evidence_dir: Path) -> None:
    expected_output = [
        "i32:3",
        "i32:8",
        "i32:5",
        "i32:11",
        "i32:5",
        "i32:7",
        "i32:20",
        "i32:11",
        "i32:3",
        "i32:12",
        "i32:7",
        "i32:7",
    ]
    expected_counts = {
        "arith.addi": 232,
        "arith.cmpi": 22,
        "arith.index_cast": 432,
        "arith.muli": 192,
        "dataflow.load": 120,
        "dataflow.store": 24,
        "llvm.intr.umin": 16,
        "llvm.trunc": 144,
        "llvm.zext": 24,
        "scf.if": 54,
    }

    dfg_path = evidence_dir / "mmtile.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_mmtile_kernel_red_0_0"
        or dfg.get("dynamic_work_items") != 2
        or dfg.get("final_outputs") != ["none", "i32:6"]
        or dfg.get("final_memory_state", {}).get("arg11") != expected_output
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"mmtile DFG evidence should match the real tiled multiply output: {dfg_path}: {dfg}")
    assert_operation_fire_counts("mmtile", dfg, expected_counts)

    mapping_path = evidence_dir / "mmtile.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"mmtile mapping should route on the shared memory reduction ADG: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "mmtile.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("final_outputs") != ["none", "i32:6"]
        or cgra.get("final_memory_state", {}).get("arg11") != expected_output
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"mmtile CGRA evidence should carry real final memory state: {cgra_path}: {cgra}")


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
        "dataflow.invariant": 22,
        "dataflow.load": 8,
        "dataflow.stream": 9,
        "dataflow.sync": 8,
    }

    dfg_path = evidence_dir / "string_hash.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 187
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
        or cgra.get("dfg_cycles") != 187
        or cgra.get("hardware_aware_cycles") != 250
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
        "arith.index_cast": 14,
        "arith.subi": 5,
        "dataflow.carry": 5,
        "dataflow.invariant": 7,
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
        or dfg.get("optimistic_cycles") != 126
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
        or cgra.get("dfg_cycles") != 126
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
        or dfg.get("optimistic_cycles") != 48155
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
        or cgra.get("dfg_cycles") != 48155
        or cgra.get("hardware_aware_cycles") != 48295
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
        or dfg.get("optimistic_cycles") != 34
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
        or cgra.get("dfg_cycles") != 34
        or cgra.get("hardware_aware_cycles") != 104
        or cgra.get("width_adapter_latency_cycles") != 3
        or cgra.get("routed_edges") != 10
        or cgra.get("route_segments") != 42
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"modmul CGRA-sim should carry the first real modular product state: {cgra_path}: {cgra}")


def assert_modexp_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": [
            "i32:3",
            "i32:4",
            "i32:2",
            "i32:7",
            "i32:11",
            "i32:5",
            "i32:13",
            "i32:17",
        ],
        "arg4": [
            "i32:2",
            "i32:3",
            "i32:5",
            "i32:123",
            "i32:65535",
            "i32:1000000006",
            "i32:314159",
            "i32:271828",
        ],
        "arg9": [
            "i32:8",
            "i32:81",
            "i32:25",
            "i32:593996258",
            "i32:586778098",
            "i32:1000000006",
            "i32:154996558",
            "i32:89848317",
        ],
    }
    expected_counts = {
        "arith.andi": 26,
        "arith.cmpi": 60,
        "arith.muli": 52,
        "arith.remui": 60,
        "arith.shrui": 26,
        "dataflow.load": 16,
        "dataflow.mux": 26,
        "dataflow.store": 8,
        "dataflow.sync": 8,
        "llvm.trunc": 8,
        "llvm.zext": 8,
        "scf.if": 8,
    }

    dfg_path = evidence_dir / "modexp.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 940
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"modexp DFG evidence should match real modular exponent rows: {dfg_path}: {dfg}")
    assert_operation_fire_counts("modexp", dfg, expected_counts)

    mapping_path = evidence_dir / "modexp.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 17
        or mapping.get("routed_edges") != 13
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 370
        or mapping.get("diagnostics") != ["mapped software graph to fabric resources"]
    ):
        raise AssertionError(f"modexp mapping should route wide modular exponentiation on the shared ADG: {mapping_path}: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_edges = {
        "arith.cmpi#1.result0->dataflow.mux#0.operand0",
        "arith.muli#0.result0->arith.remui#1.operand0",
        "arith.muli#1.result0->arith.remui#2.operand0",
        "dataflow.mux#0.result0->llvm.trunc#0.operand0",
        "llvm.trunc#0.result0->dataflow.store#0.operand2",
    }
    if not expected_edges.issubset(route_edges):
        raise AssertionError(f"modexp mapping should expose compare/mul/rem/mux/store route edges: {mapping}")

    cgra_path = evidence_dir / "modexp.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 940
        or cgra.get("hardware_aware_cycles") != 1038
        or cgra.get("width_adapter_latency_cycles") != 2
        or cgra.get("routed_edges") != 13
        or cgra.get("route_segments") != 63
        or cgra.get("config_records") != 370
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"modexp CGRA-sim should carry real modular exponentiation state: {cgra_path}: {cgra}")


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
        or dfg.get("optimistic_cycles") != 36
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
        or cgra.get("dfg_cycles") != 36
        or cgra.get("hardware_aware_cycles") != 93
        or cgra.get("routed_edges") != 9
        or cgra.get("route_segments") != 31
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"newton_iter CGRA evidence should carry x - f/df final state: {cgra_path}: {cgra}")


def assert_moving_avg_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": [
            "f32:0",
            "f32:1",
            "f32:2",
            "f32:3",
            "f32:4",
            "f32:5",
            "f32:6",
            "f32:7",
            "f32:8",
            "f32:9",
            "f32:0",
            "f32:1",
            "f32:2",
            "f32:3",
            "f32:4",
            "f32:5",
        ],
        "arg2": [
            "f32:0",
            "f32:0.500000",
            "f32:1",
            "f32:1.500000",
            "f32:2",
            "f32:3",
            "f32:4",
            "f32:5",
            "f32:6",
            "f32:7",
            "f32:6",
            "f32:5",
            "f32:4",
            "f32:3",
            "f32:2",
            "f32:3",
        ],
    }
    expected_counts = {
        "arith.divf": 16,
        "dataflow.constant": 5,
        "dataflow.load": 70,
        "dataflow.store": 16,
        "llvm.intr.umin": 16,
        "llvm.intr.usub.sat": 16,
        "llvm.uitofp": 16,
    }

    dfg_path = evidence_dir / "moving_avg.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_moving_avg_kernel_0"
        or dfg.get("dynamic_work_items") != 16
        or dfg.get("optimistic_cycles") != 1249
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"moving_avg DFG evidence should match the real window-average fixture: {dfg_path}: {dfg}")
    assert_operation_fire_counts("moving_avg", dfg, expected_counts)

    mapping_path = evidence_dir / "moving_avg.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_signal_window_adg"
        or mapping.get("placed_records") != 25
        or mapping.get("routed_edges") != 21
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 637
    ):
        raise AssertionError(f"moving_avg should route on shared signal-window hardware: {mapping_path}: {mapping}")
    placements = {
        (placement.get("software"), placement.get("hardware"))
        for placement in mapping.get("placements", [])
        if isinstance(placement, dict)
    }
    if ("dataflow.constant#4", "shared_signal_window_adg::fabric.op#126") not in placements:
        raise AssertionError(f"moving_avg should place the i64 bound constant on a wide constant PE: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_route_edges = {
        "llvm.intr.usub.sat#0.result0->arith.subi#1.operand1",
        "llvm.intr.umin#0.result0->arith.subi#0.operand1",
        "arith.divf#0.result0->dataflow.store#0.operand2",
    }
    if not expected_route_edges.issubset(route_edges):
        raise AssertionError(f"moving_avg mapping should expose usub/min/div/store route edges: {mapping}")

    cgra_path = evidence_dir / "moving_avg.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_signal_window_adg"
        or cgra.get("dfg_cycles") != 1249
        or cgra.get("hardware_aware_cycles") != 1402
        or cgra.get("placed_records") != 25
        or cgra.get("routed_edges") != 21
        or cgra.get("route_segments") != 115
        or cgra.get("config_records") != 637
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"moving_avg CGRA evidence should carry the real window-average state: {cgra_path}: {cgra}")


def assert_pool_avg_evidence(evidence_dir: Path) -> None:
    expected_outputs = [
        "none",
        "f32:3.500000",
        "none",
        "f32:5.500000",
        "none",
        "f32:11.500000",
        "none",
        "f32:13.500000",
    ]
    expected_input = [
        "f32:1",
        "f32:2",
        "f32:3",
        "f32:4",
        "f32:5",
        "f32:6",
        "f32:7",
        "f32:8",
        "f32:9",
        "f32:10",
        "f32:11",
        "f32:12",
        "f32:13",
        "f32:14",
        "f32:15",
        "f32:16",
    ]
    expected_counts = {
        "arith.addf": 16,
        "arith.addi": 48,
        "arith.divf": 16,
        "arith.index_cast": 80,
        "arith.muli": 16,
        "dataflow.load": 16,
        "llvm.trunc": 16,
        "scf.if": 8,
    }
    expected_graphs = ["g_t_pool_avg_kernel_0_0"] * 4

    dfg_path = evidence_dir / "pool_avg.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("aggregation_kind") != "workload_graph_set"
        or dfg.get("component_graphs") != expected_graphs
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 536
        or dfg.get("final_outputs") != expected_outputs
    ):
        raise AssertionError(f"pool_avg DFG evidence should cover all four real pooling outputs: {dfg_path}: {dfg}")
    memory = dfg.get("final_memory_state", {})
    if not isinstance(memory, dict) or len(memory) != 4:
        raise AssertionError(f"pool_avg DFG aggregate should retain four component input memories: {dfg}")
    for key, values in memory.items():
        if not key.startswith("pool_avg-dfg-sim-") or values != expected_input:
            raise AssertionError(f"pool_avg DFG memory provenance changed: {key}: {values}")
    assert_operation_fire_counts("pool_avg", dfg, expected_counts)

    mapping_path = evidence_dir / "pool_avg.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_signal_window_adg"
        or mapping.get("aggregation_kind") != "workload_graph_set"
        or mapping.get("component_graphs") != expected_graphs
        or mapping.get("placed_records") != 32
        or mapping.get("routed_edges") != 28
        or mapping.get("route_segments") != 140
        or mapping.get("config_records") != 788
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"pool_avg should route four pooling components on shared signal-window hardware: {mapping_path}: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_route_edges = {
        "dataflow.load#0.result0->arith.divf#0.operand0",
        "arith.divf#0.result0->arith.addf#0.operand1",
        "arith.addi#2.result0->dataflow.load#0.operand1",
    }
    if not expected_route_edges.issubset(route_edges):
        raise AssertionError(f"pool_avg mapping should expose load/div/add/address route edges: {mapping}")

    cgra_path = evidence_dir / "pool_avg.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_signal_window_adg"
        or cgra.get("aggregation_kind") != "workload_graph_set"
        or cgra.get("component_graphs") != expected_graphs
        or cgra.get("dfg_cycles") != 536
        or cgra.get("hardware_aware_cycles") != 736
        or cgra.get("placed_records") != 32
        or cgra.get("routed_edges") != 28
        or cgra.get("route_segments") != 140
        or cgra.get("config_records") != 788
        or cgra.get("final_outputs") != expected_outputs
        or cgra.get("final_memory_state") != memory
        or cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
    ):
        raise AssertionError(f"pool_avg CGRA evidence should carry real pooling outputs: {cgra_path}: {cgra}")

    comparison = json.loads((evidence_dir / "pool_avg.sim-comparison-report.json").read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("cgra_sim_cycles", 0) < comparison.get("dfg_sim_cycles", 0)
    ):
        raise AssertionError(f"pool_avg comparison should pass with CGRA cycles no lower than DFG: {comparison}")


def assert_pool_max_evidence(evidence_dir: Path) -> None:
    expected_outputs = [
        "none",
        "f32:6",
        "none",
        "f32:8",
        "none",
        "f32:14",
        "none",
        "f32:16",
    ]
    expected_input = [
        "f32:1",
        "f32:2",
        "f32:3",
        "f32:4",
        "f32:5",
        "f32:6",
        "f32:7",
        "f32:8",
        "f32:9",
        "f32:10",
        "f32:11",
        "f32:12",
        "f32:13",
        "f32:14",
        "f32:15",
        "f32:16",
    ]
    expected_counts = {
        "arith.addi": 48,
        "arith.cmpf": 16,
        "arith.index_cast": 80,
        "arith.muli": 16,
        "arith.select": 16,
        "dataflow.load": 16,
        "llvm.trunc": 16,
        "scf.if": 8,
    }
    expected_graphs = ["g_t_pool_max_kernel_0_0"] * 4

    dfg_path = evidence_dir / "pool_max.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("aggregation_kind") != "workload_graph_set"
        or dfg.get("component_graphs") != expected_graphs
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 360
        or dfg.get("final_outputs") != expected_outputs
    ):
        raise AssertionError(f"pool_max DFG evidence should cover all four real pooling outputs: {dfg_path}: {dfg}")
    memory = dfg.get("final_memory_state", {})
    if not isinstance(memory, dict) or len(memory) != 4:
        raise AssertionError(f"pool_max DFG aggregate should retain four component input memories: {dfg}")
    for key, values in memory.items():
        if not key.startswith("pool_max-dfg-sim-") or values != expected_input:
            raise AssertionError(f"pool_max DFG memory provenance changed: {key}: {values}")
    assert_operation_fire_counts("pool_max", dfg, expected_counts)

    mapping_path = evidence_dir / "pool_max.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_signal_window_adg"
        or mapping.get("aggregation_kind") != "workload_graph_set"
        or mapping.get("component_graphs") != expected_graphs
        or mapping.get("placed_records") != 32
        or mapping.get("routed_edges") != 32
        or mapping.get("route_segments") != 152
        or mapping.get("config_records") != 848
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"pool_max should route four pooling components on shared signal-window hardware: {mapping_path}: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_route_edges = {
        "arith.cmpf#0.result0->arith.select#0.operand0",
        "dataflow.load#0.result0->arith.cmpf#0.operand0",
        "dataflow.load#0.result0->arith.select#0.operand1",
        "arith.addi#2.result0->dataflow.load#0.operand1",
    }
    if not expected_route_edges.issubset(route_edges):
        raise AssertionError(f"pool_max mapping should expose load/cmp/select/address route edges: {mapping}")

    cgra_path = evidence_dir / "pool_max.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_signal_window_adg"
        or cgra.get("aggregation_kind") != "workload_graph_set"
        or cgra.get("component_graphs") != expected_graphs
        or cgra.get("dfg_cycles") != 360
        or cgra.get("hardware_aware_cycles") != 568
        or cgra.get("placed_records") != 32
        or cgra.get("routed_edges") != 32
        or cgra.get("route_segments") != 152
        or cgra.get("config_records") != 848
        or cgra.get("final_outputs") != expected_outputs
        or cgra.get("final_memory_state") != memory
        or cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
    ):
        raise AssertionError(f"pool_max CGRA evidence should carry real pooling outputs: {cgra_path}: {cgra}")

    comparison = json.loads((evidence_dir / "pool_max.sim-comparison-report.json").read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("cgra_sim_cycles", 0) < comparison.get("dfg_sim_cycles", 0)
    ):
        raise AssertionError(f"pool_max comparison should pass with CGRA cycles no lower than DFG: {comparison}")


def assert_upsample_linear_evidence(evidence_dir: Path) -> None:
    expected_input = [
        "f32:0",
        "f32:0.382683",
        "f32:0.707106",
        "f32:0.923879",
    ]
    expected_output = [
        "f32:0",
        "f32:0.095671",
        "f32:0.191342",
        "f32:0.287012",
        "f32:0.382683",
        "f32:0.463789",
        "f32:0.544895",
        "f32:0.626000",
        "f32:0.707106",
        "f32:0.761300",
        "f32:0.815493",
        "f32:0.869686",
        "f32:0.923879",
        "f32:0.923879",
        "f32:0.923879",
        "f32:0.923879",
    ]
    expected_memory = {
        "arg5": expected_input,
        "arg8": ["f32:0.923879"],
        "arg9": expected_output,
    }
    expected_counts = {
        "arith.andi": 16,
        "arith.cmpi": 28,
        "arith.index_cast": 40,
        "arith.mulf": 18,
        "arith.shrui": 28,
        "arith.subf": 9,
        "dataflow.constant": 13,
        "dataflow.load": 25,
        "dataflow.store": 16,
        "dataflow.sync": 16,
        "llvm.getelementptr": 18,
        "llvm.intr.fmuladd": 9,
        "llvm.trunc": 32,
        "llvm.uitofp": 9,
        "llvm.zext": 9,
        "scf.if": 28,
    }

    dfg_path = evidence_dir / "upsample_linear.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 16
        or dfg.get("optimistic_cycles") != 604
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"upsample_linear DFG evidence should carry the real interpolated output: {dfg_path}: {dfg}")
    assert_operation_fire_counts("upsample_linear", dfg, expected_counts)

    mapping_path = evidence_dir / "upsample_linear.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_signal_window_adg"
        or mapping.get("placed_records") != 22
        or mapping.get("routed_edges") != 23
        or mapping.get("config_records") != 588
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"upsample_linear should route on shared signal-window hardware: {mapping_path}: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_route_edges = {
        "arith.andi#0.result0->llvm.uitofp#0.operand0",
        "arith.shrui#1.result0->dataflow.load#0.operand1",
        "arith.shrui#2.result0->dataflow.load#1.operand1",
        "dataflow.load#1.result0->llvm.intr.fmuladd#0.operand1",
        "llvm.intr.fmuladd#0.result0->dataflow.store#0.operand2",
    }
    if not expected_route_edges.issubset(route_edges):
        raise AssertionError(f"upsample_linear mapping should expose interpolation route edges: {mapping}")

    cgra_path = evidence_dir / "upsample_linear.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_signal_window_adg"
        or cgra.get("dfg_cycles") != 604
        or cgra.get("hardware_aware_cycles") != 759
        or cgra.get("placed_records") != 22
        or cgra.get("routed_edges") != 23
        or cgra.get("route_segments") != 105
        or cgra.get("config_records") != 588
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"upsample_linear CGRA evidence should carry the real interpolated output: {cgra_path}: {cgra}")

    comparison = json.loads((evidence_dir / "upsample_linear.sim-comparison-report.json").read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("cgra_sim_cycles", 0) < comparison.get("dfg_sim_cycles", 0)
    ):
        raise AssertionError(f"upsample_linear comparison should pass with CGRA cycles no lower than DFG: {comparison}")


def assert_bisection_step_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg1": ["f32:0", "f32:1", "f32:2"],
        "arg2": ["f32:2", "f32:5", "f32:6"],
        "arg4": ["f32:-1", "f32:-2", "f32:4"],
        "arg5": ["f32:0.250000", "f32:-0.500000", "f32:5"],
        "arg7": ["f32:0", "f32:3", "f32:0"],
        "arg8": ["f32:0", "f32:5", "f32:0"],
    }
    expected_counts = {
        "arith.addf": 1,
        "arith.cmpf": 1,
        "arith.mulf": 2,
        "arith.select": 2,
        "dataflow.load": 4,
        "dataflow.store": 2,
        "dataflow.sync": 1,
    }

    dfg_path = evidence_dir / "bisection_step.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 1
        or dfg.get("optimistic_cycles") != 44
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"bisection_step DFG evidence should update the real interval row: {dfg_path}: {dfg}")
    assert_operation_fire_counts("bisection_step", dfg, expected_counts)

    mapping_path = evidence_dir / "bisection_step.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("routed_edges") != 20
    ):
        raise AssertionError(f"bisection_step should route on shared memory reduction hardware: {mapping_path}: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_edges = {
        "arith.addf#0.result0->arith.mulf#0.operand0",
        "arith.mulf#1.result0->arith.cmpf#0.operand0",
        "arith.cmpf#0.result0->arith.select#0.operand0",
        "arith.select#0.result0->dataflow.store#0.operand2",
        "dataflow.store#1.result0->dataflow.sync#0.operand5",
    }
    if not expected_edges.issubset(route_edges):
        raise AssertionError(f"bisection_step mapping should expose add/mul/cmp/select/store routes: {mapping}")

    cgra_path = evidence_dir / "bisection_step.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 44
        or cgra.get("hardware_aware_cycles") != 162
        or cgra.get("routed_edges") != 20
        or cgra.get("route_segments") != 76
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"bisection_step CGRA evidence should carry the real final interval state: {cgra_path}: {cgra}")


def assert_edit_distance_step_evidence(evidence_dir: Path) -> None:
    expected_output = [f"i32:{value}" for value in range(1, 65)]
    expected_counts = {
        "arith.addi": 192,
        "arith.cmpi": 64,
        "dataflow.load": 320,
        "dataflow.store": 64,
        "dataflow.sync": 64,
        "llvm.intr.umin": 128,
        "llvm.zext": 64,
    }

    dfg_path = evidence_dir / "edit_distance_step.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_edit_distance_step_kernel_0_0"
        or dfg.get("dynamic_work_items") != 64
        or dfg.get("event_count") != 896
        or dfg.get("optimistic_cycles") != 2055
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"edit_distance_step DFG evidence should execute the full legacy input: {dfg_path}: {dfg}")
    assert_operation_fire_counts("edit_distance_step", dfg, expected_counts)
    memory = dfg.get("final_memory_state", {})
    if not isinstance(memory, dict) or memory.get("arg7") != expected_output:
        raise AssertionError(f"edit_distance_step DFG should write the real DP output row: {dfg_path}: {dfg}")

    mapping_path = evidence_dir / "edit_distance_step.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 14
        or mapping.get("routed_edges") != 18
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 388
    ):
        raise AssertionError(f"edit_distance_step should route on shared memory reduction hardware: {mapping_path}: {mapping}")
    route_edges = {
        str(route.get("edge_ref"))
        for route in mapping.get("routes", [])
        if isinstance(route, dict)
    }
    required_edges = {
        "dataflow.load#0.result0->arith.cmpi#0.operand0",
        "arith.addi#0.result0->llvm.intr.umin#0.operand0",
        "llvm.intr.umin#0.result0->llvm.intr.umin#1.operand0",
        "llvm.intr.umin#1.result0->dataflow.store#0.operand2",
        "dataflow.store#0.result0->dataflow.sync#0.operand5",
    }
    if not required_edges <= route_edges:
        raise AssertionError(f"edit_distance_step mapping missed required DP routes: {mapping_path}: {mapping}")
    assert_mapping_edges_use_switch_multihop(evidence_dir, "edit_distance_step", required_edges)

    cgra_path = evidence_dir / "edit_distance_step.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("dfg_cycles") != 2055
        or cgra.get("hardware_aware_cycles") != 2161
        or cgra.get("routed_edges") != 18
        or cgra.get("route_segments") != 68
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state", {}).get("arg7") != expected_output
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"edit_distance_step CGRA evidence should carry the full DP output row: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / "edit_distance_step.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 2055
        or comparison.get("cgra_sim_cycles") != 2161
    ):
        raise AssertionError(f"edit_distance_step comparison should pass with real final-state checks: {comparison_path}: {comparison}")


def assert_transform_point_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg2": [
            "f32:1",
            "f32:2",
            "f32:3",
            "f32:1.100000",
            "f32:2.200000",
            "f32:3.300000",
            "f32:1.200000",
            "f32:2.400000",
            "f32:3.600000",
            "f32:1.300000",
            "f32:2.600000",
            "f32:3.900000",
        ],
        "arg9": [
            "f32:0",
            "f32:0",
            "f32:0",
            "f32:0",
            "f32:0",
            "f32:0",
            "f32:3.400000",
            "f32:6.800000",
            "f32:10.200000",
            "f32:0",
            "f32:0",
            "f32:0",
        ],
    }
    expected_counts = {
        "arith.addf": 3,
        "arith.addi": 2,
        "arith.index_cast": 9,
        "arith.mulf": 3,
        "arith.muli": 3,
        "dataflow.load": 3,
        "dataflow.store": 3,
        "dataflow.sync": 1,
        "llvm.intr.fmuladd": 6,
        "llvm.trunc": 1,
    }

    dfg_path = evidence_dir / "transform_point.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 1
        or dfg.get("optimistic_cycles") != 128
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"transform_point DFG evidence should update the real output point: {dfg_path}: {dfg}")
    assert_operation_fire_counts("transform_point", dfg, expected_counts)

    mapping_path = evidence_dir / "transform_point.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 25
        or mapping.get("routed_edges") != 38
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
    ):
        raise AssertionError(f"transform_point should route on shared memory reduction hardware: {mapping_path}: {mapping}")
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    expected_edges = {
        "arith.addf#0.result0->dataflow.store#0.operand2",
        "arith.addf#1.result0->dataflow.store#1.operand2",
        "arith.addf#2.result0->dataflow.store#2.operand2",
        "llvm.intr.fmuladd#0.result0->llvm.intr.fmuladd#1.operand2",
        "llvm.intr.fmuladd#5.result0->arith.addf#2.operand1",
        "dataflow.store#2.result0->dataflow.sync#0.operand5",
    }
    if not expected_edges.issubset(route_edges):
        raise AssertionError(f"transform_point mapping should expose fma/add/store routes: {mapping}")

    cgra_path = evidence_dir / "transform_point.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 128
        or cgra.get("hardware_aware_cycles") != 358
        or cgra.get("routed_edges") != 38
        or cgra.get("route_segments") != 160
        or cgra.get("final_outputs") != ["none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra.get("fidelity_level") != "mapping_constraint_estimate"
    ):
        raise AssertionError(f"transform_point CGRA evidence should carry the real affine output state: {cgra_path}: {cgra}")


def assert_rle_decode_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg4": ["i32:1", "i32:2", "i32:3", "i32:4", "i32:5", "i32:6", "i32:7"],
        "arg5": ["i32:3", "i32:2", "i32:4", "i32:5", "i32:1", "i32:3", "i32:2"],
        "arg7": [
            "i32:1",
            "i32:1",
            "i32:1",
            "i32:2",
            "i32:2",
            "i32:3",
            "i32:3",
            "i32:3",
            "i32:3",
            "i32:4",
            "i32:4",
            "i32:4",
            "i32:4",
            "i32:4",
            "i32:5",
            "i32:6",
            "i32:6",
            "i32:6",
            "i32:7",
            "i32:7",
        ],
    }

    dfg_path = evidence_dir / "rle_decode.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 7
        or dfg.get("optimistic_cycles") != 227
        or dfg.get("final_outputs") != ["none", "i32:20"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"rle_decode DFG evidence should decode the real run-length stream: {dfg_path}: {dfg}")
    assert_operation_fire_counts(
        "rle_decode",
        dfg,
        {
            "arith.addi": 14,
            "arith.cmpi": 7,
            "arith.index_cast": 35,
            "dataflow.load": 14,
            "dataflow.store": 20,
            "scf.forall": 7,
            "scf.if": 7,
        },
    )

    mapping_path = evidence_dir / "rle_decode.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    route_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    index_cast_placements = [
        placement
        for placement in mapping.get("placements", [])
        if placement.get("operation") == "arith.index_cast"
    ]
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 8
        or mapping.get("routed_edges") != 6
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 158
        or len(index_cast_placements) != 2
        or route_edges
        != {
            "arith.index_cast#0.result0->dataflow.load#0.operand1",
            "arith.index_cast#1.result0->dataflow.load#1.operand1",
            "dataflow.load#0.result0->dataflow.store#0.operand2",
            "dataflow.load#1.result0->arith.addi#0.operand1",
            "dataflow.load#1.result0->arith.addi#1.operand1",
            "dataflow.load#1.result0->arith.cmpi#0.operand0",
        }
    ):
        raise AssertionError(f"rle_decode should route on shared memory reduction hardware: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "rle_decode.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_memory_reduction_adg"
        or cgra.get("dfg_cycles") != 227
        or cgra.get("hardware_aware_cycles") != 276
        or cgra.get("placed_records") != 8
        or cgra.get("routed_edges") != 6
        or cgra.get("route_segments") != 26
        or cgra.get("config_records") != 158
        or cgra.get("final_outputs") != ["none", "i32:20"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
        or cgra.get("hardware_aware_cycles", 0) < cgra.get("dfg_cycles", 0)
    ):
        raise AssertionError(f"rle_decode CGRA evidence should carry the decoded output state: {cgra_path}: {cgra}")


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
        or dfg.get("optimistic_cycles") != 56
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
        or cgra.get("dfg_cycles") != 56
        or cgra.get("hardware_aware_cycles") != 144
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
        or dfg.get("optimistic_cycles") != 197
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
        or cgra.get("dfg_cycles") != 197
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
        "dataflow.invariant": 30,
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
        or cgra.get("dfg_cycles") != 341
        or cgra.get("hardware_aware_cycles") != 482
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
        "dataflow.invariant": 52,
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
        or dfg.get("optimistic_cycles") != 586
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
        or cgra.get("dfg_cycles") != 586
        or cgra.get("hardware_aware_cycles") != 868
        or cgra.get("routed_edges") != 50
        or cgra.get("route_segments") != 226
        or cgra.get("final_outputs") != ["none", "i32:5", "none", "i32:10"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
        or cgra.get("diagnostics") != ["derived workload graph-set CGRA report from component CGRA simulator reports"]
    ):
        raise AssertionError(f"partition CGRA-sim should carry the two-sided real partition state: {cgra_path}: {cgra}")


def assert_outer_evidence(evidence_dir: Path) -> None:
    graph = "g_t_outer_kernel_0_0"
    expected_graphs = [graph, graph, graph]
    expected_lhs = ["i32:1", "i32:2", "i32:3"]
    expected_rhs = ["i32:1", "i32:3", "i32:5", "i32:7"]
    expected_output = [
        "i32:1",
        "i32:3",
        "i32:5",
        "i32:7",
        "i32:2",
        "i32:6",
        "i32:10",
        "i32:14",
        "i32:3",
        "i32:9",
        "i32:15",
        "i32:21",
    ]
    expected_output_rows = [
        [
            "i32:1",
            "i32:3",
            "i32:5",
            "i32:7",
            "i32:0",
            "i32:0",
            "i32:0",
            "i32:0",
            "i32:0",
            "i32:0",
            "i32:0",
            "i32:0",
        ],
        [
            "i32:1",
            "i32:3",
            "i32:5",
            "i32:7",
            "i32:2",
            "i32:6",
            "i32:10",
            "i32:14",
            "i32:0",
            "i32:0",
            "i32:0",
            "i32:0",
        ],
        expected_output,
    ]
    expected_counts = {
        "arith.index_cast": 3,
        "arith.muli": 12,
        "arith.shli": 3,
        "dataflow.load": 24,
        "dataflow.store": 12,
        "llvm.getelementptr": 3,
        "scf.forall": 3,
    }

    dfg_path = evidence_dir / "outer.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    component_identities = dfg.get("component_dfg_sim_report_identities", [])
    if not isinstance(component_identities, list) or len(component_identities) != 3:
        raise AssertionError(f"outer should cite three row component reports: {dfg}")
    component_labels = [raw_component_identity("outer", str(identity)) for identity in component_identities]
    expected_memory = {}
    for label, row_output in zip(component_labels, expected_output_rows):
        expected_memory[f"{label}:arg1"] = expected_lhs
        expected_memory[f"{label}:arg3"] = row_output
        expected_memory[f"{label}:arg4"] = expected_rhs
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 12
        or dfg.get("optimistic_cycles") != 213
        or dfg.get("component_graphs") != expected_graphs
        or dfg.get("final_outputs") != ["none", "none", "none"]
        or dfg.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"outer DFG aggregate should carry full 3x4 outer product state: {dfg_path}: {dfg}")
    assert_operation_fire_counts("outer", dfg, expected_counts)

    components = [json.loads((evidence_dir / f"{identity}.json").read_text()) for identity in component_identities]
    if len(components) != 3 or any(component.get("graph") != graph for component in components):
        raise AssertionError(f"outer should cite three row component reports: {dfg}")
    if components[-1].get("final_memory_state", {}).get("arg3") != expected_output:
        raise AssertionError(f"outer final row component should contain complete product output: {components[-1]}")

    mapping_path = evidence_dir / "outer.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("component_graphs") != expected_graphs
        or mapping.get("placed_records") != 15
        or mapping.get("routed_edges") != 9
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("route_segments") != 33
    ):
        raise AssertionError(f"outer should aggregate three passing row mappings: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "outer.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 213
        or cgra.get("hardware_aware_cycles") != 306
        or cgra.get("component_graphs") != expected_graphs
        or cgra.get("routed_edges") != 9
        or cgra.get("route_segments") != 33
        or cgra.get("final_outputs") != ["none", "none", "none"]
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
    ):
        raise AssertionError(f"outer CGRA aggregate should carry full 3x4 outer product state: {cgra_path}: {cgra}")


def assert_transpose_evidence(evidence_dir: Path) -> None:
    graph = "g_t_transpose_0_0"
    expected_graphs = [graph, graph, graph]
    expected_input = [
        "i32:1",
        "i32:3",
        "i32:5",
        "i32:7",
        "i32:9",
        "i32:11",
        "i32:13",
        "i32:15",
        "i32:17",
        "i32:19",
        "i32:21",
        "i32:23",
        "i32:25",
        "i32:27",
        "i32:29",
    ]
    expected_output = [
        "i32:1",
        "i32:11",
        "i32:21",
        "i32:3",
        "i32:13",
        "i32:23",
        "i32:5",
        "i32:15",
        "i32:25",
        "i32:7",
        "i32:17",
        "i32:27",
        "i32:9",
        "i32:19",
        "i32:29",
    ]
    expected_output_rows = [
        [
            "i32:1",
            "i32:0",
            "i32:0",
            "i32:3",
            "i32:0",
            "i32:0",
            "i32:5",
            "i32:0",
            "i32:0",
            "i32:7",
            "i32:0",
            "i32:0",
            "i32:9",
            "i32:0",
            "i32:0",
        ],
        [
            "i32:1",
            "i32:11",
            "i32:0",
            "i32:3",
            "i32:13",
            "i32:0",
            "i32:5",
            "i32:15",
            "i32:0",
            "i32:7",
            "i32:17",
            "i32:0",
            "i32:9",
            "i32:19",
            "i32:0",
        ],
        expected_output,
    ]
    expected_counts = {
        "arith.index_cast": 18,
        "arith.muli": 18,
        "arith.shrui": 15,
        "dataflow.constant": 15,
        "dataflow.load": 15,
        "dataflow.store": 15,
        "llvm.getelementptr": 6,
        "scf.forall": 3,
    }

    dfg_path = evidence_dir / "transpose.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    component_identities = dfg.get("component_dfg_sim_report_identities", [])
    if not isinstance(component_identities, list) or len(component_identities) != 3:
        raise AssertionError(f"transpose should cite three row component reports: {dfg}")
    component_labels = [raw_component_identity("transpose", str(identity)) for identity in component_identities]
    expected_memory = {}
    for label, row_output in zip(component_labels, expected_output_rows):
        expected_memory[f"{label}:arg2"] = expected_input
        expected_memory[f"{label}:arg3"] = row_output
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 15
        or dfg.get("optimistic_cycles") != 285
        or dfg.get("component_graphs") != expected_graphs
        or dfg.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"transpose DFG aggregate should carry full 3x5 transpose state: {dfg_path}: {dfg}")
    assert_operation_fire_counts("transpose", dfg, expected_counts)

    components = [json.loads((evidence_dir / f"{identity}.json").read_text()) for identity in component_identities]
    if len(components) != 3 or any(component.get("graph") != graph for component in components):
        raise AssertionError(f"transpose should cite three row component reports: {dfg}")
    if components[-1].get("final_memory_state", {}).get("arg3") != expected_output:
        raise AssertionError(f"transpose final row component should contain complete transposed output: {components[-1]}")

    mapping_path = evidence_dir / "transpose.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("component_graphs") != expected_graphs
        or mapping.get("placed_records") != 18
        or mapping.get("routed_edges") != 12
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("route_segments") != 54
    ):
        raise AssertionError(f"transpose should aggregate three passing row mappings: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "transpose.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 285
        or cgra.get("hardware_aware_cycles") != 399
        or cgra.get("component_graphs") != expected_graphs
        or cgra.get("routed_edges") != 12
        or cgra.get("route_segments") != 54
        or cgra.get("final_memory_state") != expected_memory
        or cgra.get("functional_state_source") != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
    ):
        raise AssertionError(f"transpose CGRA aggregate should carry full 3x5 transpose state: {cgra_path}: {cgra}")


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


def assert_merge_dfg_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg10": [
            "f32:1",
            "f32:2",
            "f32:3",
            "f32:4",
            "f32:9",
            "f32:10",
            "f32:13",
            "f32:14",
            "f32:20",
            "f32:21",
            "f32:22",
        ],
        "arg7": ["f32:1", "f32:4", "f32:9", "f32:13", "f32:21"],
        "arg8": ["f32:2", "f32:3", "f32:10", "f32:14", "f32:20", "f32:22"],
    }
    expected_counts = {
        "arith.addi": 11,
        "arith.cmpf": 10,
        "arith.cmpi": 22,
        "arith.extui": 10,
        "arith.index_cast": 42,
        "arith.index_castui": 22,
        "arith.xori": 10,
        "dataflow.load": 31,
        "dataflow.store": 11,
        "scf.if": 22,
        "scf.index_switch": 22,
    }
    dfg_path = evidence_dir / "merge.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 11
        or dfg.get("optimistic_cycles") != 413
        or dfg.get("final_outputs") != ["none", "i32:5", "i32:6"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"merge DFG evidence should preserve the real merge output before PnR blocking: {dfg_path}: {dfg}")
    assert_operation_fire_counts("merge", dfg, expected_counts)

    mapping_path = evidence_dir / "merge.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    expected_edges = {
        "arith.cmpf#0.result0->arith.xori#0.operand0",
        "arith.index_cast#0.result0->dataflow.store#0.operand1",
        "arith.index_cast#1.result0->dataflow.store#1.operand1",
        "arith.xori#0.result0->arith.extui#0.operand0",
        "dataflow.load#0.result0->arith.cmpf#0.operand0",
        "dataflow.load#1.result0->arith.cmpf#0.operand1",
        "dataflow.load#2.result0->dataflow.store#0.operand2",
        "dataflow.load#3.result0->dataflow.store#1.operand2",
    }
    actual_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("placed_records") != 15
        or mapping.get("routed_edges") != 8
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 234
        or actual_edges != expected_edges
    ):
        raise AssertionError(f"merge mapping should route compare, xor, and load-store edges: {mapping_path}: {mapping}")
    index_cast_sites = {
        str(placement.get("hardware"))
        for placement in mapping.get("placements", [])
        if isinstance(placement, dict) and placement.get("operation") == "arith.index_cast"
    }
    if index_cast_sites != {
        "shared_reduction_adg::fabric.op#97",
        "shared_reduction_adg::fabric.op#98",
    }:
        raise AssertionError(f"merge should place both store address casts on shared ADG: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "merge.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != "shared_reduction_adg"
        or cgra.get("dfg_cycles") != 413
        or cgra.get("hardware_aware_cycles") != 492
        or cgra.get("placed_records") != 15
        or cgra.get("routed_edges") != 8
        or cgra.get("route_segments") != 36
        or cgra.get("config_records") != 234
        or cgra.get("final_outputs") != ["none", "i32:5", "i32:6"]
        or cgra.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"merge CGRA-sim should preserve the real merge output: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / "merge.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 413
        or comparison.get("cgra_sim_cycles") != 492
    ):
        raise AssertionError(f"merge comparison should pass with real final-state checks: {comparison_path}: {comparison}")


def assert_autocorrelation_dfg_evidence(evidence_dir: Path) -> None:
    expected_memory = {
        "arg7": [
            "f32:1",
            "f32:2",
            "f32:3",
            "f32:4",
            "f32:5",
            "f32:6",
            "f32:7",
            "f32:8",
        ],
        "arg9": [
            "f32:0",
            "f32:36",
            "f32:36",
            "f32:36",
            "f32:36",
            "f32:36",
            "f32:36",
            "f32:36",
        ],
    }
    expected_counts = {
        "arith.addi": 64,
        "arith.andi": 56,
        "arith.cmpi": 8,
        "arith.index_cast": 232,
        "dataflow.load": 112,
        "dataflow.store": 8,
        "llvm.intr.fmuladd": 56,
        "llvm.intr.umax": 7,
        "llvm.zext": 7,
        "scf.if": 8,
    }
    dfg_path = evidence_dir / "autocorrelation.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("event_count") != 558
        or dfg.get("final_outputs") != ["none", "i32:0"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"autocorrelation DFG should execute nested structured loops: {dfg_path}: {dfg}")
    assert_operation_fire_counts("autocorrelation", dfg, expected_counts)
    mapping_path = evidence_dir / "autocorrelation.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    index_cast_placements = [
        placement
        for placement in mapping.get("placements", [])
        if placement.get("operation") == "arith.index_cast"
    ]
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_reduction_adg"
        or mapping.get("placed_records") != 12
        or mapping.get("routed_edges") != 8
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 235
        or len(index_cast_placements) != 2
    ):
        raise AssertionError(f"autocorrelation mapping should route on shared reduction ADG: {mapping_path}: {mapping}")
    expected_edges = {
        "arith.addi#0.result0->arith.andi#0.operand0",
        "arith.andi#0.result0->dataflow.load#1.operand1",
        "arith.index_cast#0.result0->dataflow.load#0.operand1",
        "arith.index_cast#1.result0->dataflow.store#0.operand1",
        "dataflow.load#0.result0->llvm.intr.fmuladd#0.operand0",
        "dataflow.load#1.result0->llvm.intr.fmuladd#0.operand1",
        "llvm.intr.fmuladd#0.result0->dataflow.store#0.operand2",
        "llvm.intr.umax#0.result0->llvm.zext#0.operand0",
    }
    actual_edges = {route.get("edge_ref") for route in mapping.get("routes", [])}
    if actual_edges != expected_edges:
        raise AssertionError(f"autocorrelation mapping should expose routed DFG edges: {mapping_path}: {mapping}")
    cgra_path = evidence_dir / "autocorrelation.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("placed_records") != 12
        or cgra.get("routed_edges") != 8
        or cgra.get("route_segments") != 40
        or cgra.get("config_records") != 235
        or cgra.get("hardware_aware_cycles") != 1519
        or cgra.get("width_adapter_latency_cycles") != 1
        or cgra.get("dfg_cycles") != 1448
        or cgra.get("final_outputs") != ["none", "i32:0"]
        or cgra.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"autocorrelation CGRA-sim should preserve final state: {cgra_path}: {cgra}")
    comparison_path = evidence_dir / "autocorrelation.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 1448
        or comparison.get("cgra_sim_cycles") != 1519
    ):
        raise AssertionError(f"autocorrelation comparison should pass with real final-state checks: {comparison_path}: {comparison}")


def assert_crc32_evidence(evidence_dir: Path) -> None:
    expected_counts = {
        "arith.andi": 64,
        "arith.index_cast": 144,
        "arith.shli": 64,
        "arith.shrui": 128,
        "arith.xori": 128,
        "dataflow.load": 80,
    }
    dfg_path = evidence_dir / "crc32.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("dynamic_work_items") != 16
        or dfg.get("event_count") != 608
        or dfg.get("final_outputs") != ["none", "i32:-1307787247"]
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"crc32 DFG should execute the structured CRC loop: {dfg_path}: {dfg}")
    assert_operation_fire_counts("crc32", dfg, expected_counts)
    memory = dfg.get("final_memory_state", {})
    if (
        not isinstance(memory, dict)
        or len(memory.get("arg4", [])) != 16
        or len(memory.get("arg8", [])) != 256
        or memory.get("arg4", [None])[1] != "i32:305419896"
        or memory.get("arg8", [None])[-1] != "i32:755167117"
    ):
        raise AssertionError(f"crc32 DFG should carry real input/table memory state: {dfg_path}: {dfg}")

    mapping_path = evidence_dir / "crc32.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("placed_records") != 9
        or mapping.get("routed_edges") != 8
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 214
    ):
        raise AssertionError(f"crc32 should map the two-shift CRC slice on shared ADG: {mapping_path}: {mapping}")
    index_cast_sites = {
        str(placement.get("hardware"))
        for placement in mapping.get("placements", [])
        if isinstance(placement, dict) and placement.get("operation") == "arith.index_cast"
    }
    if index_cast_sites != {"shared_reduction_adg::fabric.op#97"}:
        raise AssertionError(f"crc32 should place its computed address cast on shared ADG: {mapping_path}: {mapping}")
    route_edges = {
        str(route.get("edge_ref"))
        for route in mapping.get("routes", [])
        if isinstance(route, dict)
    }
    expected_route_edges = {
        "arith.andi#0.result0->dataflow.load#1.operand1",
        "arith.index_cast#0.result0->dataflow.load#0.operand1",
        "arith.shli#0.result0->arith.shrui#1.operand1",
        "arith.shrui#0.result0->arith.xori#1.operand0",
        "arith.shrui#1.result0->arith.xori#0.operand0",
        "arith.xori#0.result0->arith.andi#0.operand0",
        "dataflow.load#0.result0->arith.shrui#1.operand0",
        "dataflow.load#1.result0->arith.xori#1.operand1",
    }
    if route_edges != expected_route_edges:
        raise AssertionError(f"crc32 should expose all CRC mix route edges: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "crc32.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 934
        or cgra.get("hardware_aware_cycles") != 989
        or cgra.get("placed_records") != 9
        or cgra.get("routed_edges") != 8
        or cgra.get("route_segments") != 38
        or cgra.get("config_records") != 214
        or cgra.get("final_outputs") != ["none", "i32:-1307787247"]
        or cgra.get("final_memory_state") != memory
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"crc32 CGRA evidence should carry the real CRC result state: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / "crc32.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 934
        or comparison.get("cgra_sim_cycles") != 989
    ):
        raise AssertionError(f"crc32 comparison should pass with real final-state checks: {comparison_path}: {comparison}")


def assert_gather_evidence(evidence_dir: Path) -> None:
    expected_dst = [
        "i32:1",
        "i32:10",
        "i32:28",
        "i32:0",
        "i32:7",
        "i32:22",
        "i32:0",
        "i32:4",
        "i32:16",
        "i32:25",
        "i32:19",
        "i32:13",
        "i32:0",
        "i32:1",
        "i32:28",
        "i32:0",
    ]
    expected_counts = {
        "arith.cmpi": 16,
        "arith.index_cast": 12,
        "dataflow.load": 28,
        "dataflow.store": 16,
        "dataflow.sync": 16,
        "scf.if": 16,
    }
    dfg_path = evidence_dir / "gather.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_gather_0_0"
        or dfg.get("dynamic_work_items") != 16
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"gather DFG should execute the indirect-load graph: {dfg_path}: {dfg}")
    assert_operation_fire_counts("gather", dfg, expected_counts)
    memory = dfg.get("final_memory_state", {})
    if not isinstance(memory, dict) or memory.get("arg5") != expected_dst:
        raise AssertionError(f"gather DFG should preserve real dst memory state: {dfg_path}: {dfg}")

    mapping_path = evidence_dir / "gather.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("placed_records") != 5
        or mapping.get("routed_edges") != 5
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 98
    ):
        raise AssertionError(f"gather should map the bounded indirect load/store graph: {mapping_path}: {mapping}")
    route_edges = {
        str(route.get("edge_ref"))
        for route in mapping.get("routes", [])
        if isinstance(route, dict)
    }
    required_edges = {
        "dataflow.load#0.result0->arith.cmpi#0.operand0",
        "dataflow.load#0.result0->dataflow.load#1.operand1",
        "dataflow.load#1.result0->dataflow.store#0.operand2",
        "dataflow.store#0.result0->dataflow.sync#0.operand1",
    }
    if not required_edges <= route_edges:
        raise AssertionError(f"gather mapping missed required routed edges: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "gather.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 254
        or cgra.get("hardware_aware_cycles") != 289
        or cgra.get("performance_delta_cycles") != 35
        or cgra.get("final_memory_state", {}).get("arg5") != expected_dst
    ):
        raise AssertionError(f"gather CGRA report should carry real final state: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / "gather.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 254
        or comparison.get("cgra_sim_cycles") != 289
    ):
        raise AssertionError(f"gather comparison should pass with real final-state checks: {comparison_path}: {comparison}")


def assert_histogram_evidence(evidence_dir: Path) -> None:
    expected_hist = [f"i32:{value}" for value in range(1, 17)]
    expected_counts = {
        "arith.addi": 136,
        "arith.cmpi": 138,
        "arith.index_cast": 409,
        "dataflow.constant": 22,
        "dataflow.load": 272,
        "dataflow.store": 152,
        "llvm.zext": 1,
        "scf.if": 138,
    }
    dfg_path = evidence_dir / "histogram.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_histogram_kernel_0"
        or dfg.get("dynamic_work_items") != 136
        or dfg.get("event_count") != 1268
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("diagnostics") != []
        or dfg.get("final_memory_state", {}).get("arg2") != expected_hist
    ):
        raise AssertionError(f"histogram DFG should execute the full clear-and-update kernel: {dfg_path}: {dfg}")
    assert_operation_fire_counts("histogram", dfg, expected_counts)

    mapping_path = evidence_dir / "histogram.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 13
        or mapping.get("routed_edges") != 10
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 268
    ):
        raise AssertionError(f"histogram should map on shared memory-reduction hardware: {mapping_path}: {mapping}")
    route_edges = {
        str(route.get("edge_ref"))
        for route in mapping.get("routes", [])
        if isinstance(route, dict)
    }
    required_edges = {
        "arith.addi#0.result0->dataflow.store#1.operand2",
        "dataflow.constant#2.result0->dataflow.store#0.operand2",
        "dataflow.load#0.result0->arith.cmpi#2.operand0",
        "dataflow.load#0.result0->dataflow.load#1.operand1",
        "dataflow.load#0.result0->dataflow.store#1.operand1",
        "dataflow.load#1.result0->arith.addi#0.operand0",
    }
    if not required_edges <= route_edges:
        raise AssertionError(f"histogram mapping missed required routed edges: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "histogram.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 3092
        or cgra.get("hardware_aware_cycles") != 3172
        or cgra.get("performance_delta_cycles") != 80
        or cgra.get("placed_records") != 13
        or cgra.get("routed_edges") != 10
        or cgra.get("route_segments") != 44
        or cgra.get("config_records") != 268
        or cgra.get("final_memory_state", {}).get("arg2") != expected_hist
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"histogram CGRA report should carry the real final histogram: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / "histogram.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 3092
        or comparison.get("cgra_sim_cycles") != 3172
    ):
        raise AssertionError(f"histogram comparison should pass with real memory-state checks: {comparison_path}: {comparison}")


def assert_histogram_strided_evidence(evidence_dir: Path) -> None:
    expected_hist = [f"i32:{value}" for value in range(2, 18)]
    expected_counts = {
        "arith.addi": 152,
        "arith.cmpi": 154,
        "arith.divui": 152,
        "arith.index_cast": 457,
        "dataflow.constant": 22,
        "dataflow.load": 304,
        "dataflow.store": 168,
        "llvm.zext": 1,
        "scf.if": 154,
    }
    dfg_path = evidence_dir / "histogram_strided.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_histogram_strided_kernel_0"
        or dfg.get("dynamic_work_items") != 152
        or dfg.get("event_count") != 1564
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("diagnostics") != []
        or dfg.get("final_memory_state", {}).get("arg2") != expected_hist
    ):
        raise AssertionError(
            f"histogram_strided DFG should execute the full clear-and-update kernel: {dfg_path}: {dfg}"
        )
    assert_operation_fire_counts("histogram_strided", dfg, expected_counts)

    mapping_path = evidence_dir / "histogram_strided.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_memory_reduction_adg"
        or mapping.get("placed_records") != 14
        or mapping.get("routed_edges") != 11
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 312
    ):
        raise AssertionError(
            f"histogram_strided should map on shared memory-reduction hardware: {mapping_path}: {mapping}"
        )
    route_edges = {
        str(route.get("edge_ref"))
        for route in mapping.get("routes", [])
        if isinstance(route, dict)
    }
    required_edges = {
        "arith.divui#0.result0->arith.cmpi#2.operand0",
        "arith.divui#0.result0->dataflow.load#1.operand1",
        "arith.divui#0.result0->dataflow.store#1.operand1",
        "dataflow.load#0.result0->arith.divui#0.operand0",
        "dataflow.load#1.result0->arith.addi#0.operand0",
    }
    if not required_edges <= route_edges:
        raise AssertionError(
            f"histogram_strided mapping missed required routed edges: {mapping_path}: {mapping}"
        )

    cgra_path = evidence_dir / "histogram_strided.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 4661
        or cgra.get("hardware_aware_cycles") != 4752
        or cgra.get("performance_delta_cycles") != 91
        or cgra.get("placed_records") != 14
        or cgra.get("routed_edges") != 11
        or cgra.get("route_segments") != 53
        or cgra.get("config_records") != 312
        or cgra.get("final_memory_state", {}).get("arg2") != expected_hist
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(
            f"histogram_strided CGRA report should carry the real final histogram: {cgra_path}: {cgra}"
        )

    comparison_path = evidence_dir / "histogram_strided.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 4661
        or comparison.get("cgra_sim_cycles") != 4752
    ):
        raise AssertionError(
            f"histogram_strided comparison should pass with real memory-state checks: {comparison_path}: {comparison}"
        )


def assert_hist_bin_evidence(evidence_dir: Path) -> None:
    expected_hist = [f"i32:{value}" for value in range(1, 11)]
    expected_counts = {
        "arith.addi": 56,
        "arith.cmpf": 110,
        "arith.cmpi": 57,
        "arith.divf": 56,
        "dataflow.load": 110,
        "dataflow.store": 65,
        "llvm.fptoui": 55,
        "llvm.uitofp": 1,
        "scf.if": 57,
    }
    dfg_path = evidence_dir / "hist_bin.dfg.report.json"
    dfg = json.loads(dfg_path.read_text())
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_hist_bin_kernel_0"
        or dfg.get("dynamic_work_items") != 55
        or dfg.get("event_count") != 917
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("diagnostics") != []
        or dfg.get("final_memory_state", {}).get("arg2") != expected_hist
    ):
        raise AssertionError(f"hist_bin DFG should execute the full clear-and-bin kernel: {dfg_path}: {dfg}")
    assert_operation_fire_counts("hist_bin", dfg, expected_counts)

    mapping_path = evidence_dir / "hist_bin.mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != "shared_signal_window_adg"
        or mapping.get("placed_records") != 25
        or mapping.get("routed_edges") != 24
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != 644
    ):
        raise AssertionError(f"hist_bin should map on shared signal-window hardware: {mapping_path}: {mapping}")
    route_edges = {
        str(route.get("edge_ref"))
        for route in mapping.get("routes", [])
        if isinstance(route, dict)
    }
    required_edges = {
        "arith.divf#0.result0->arith.divf#1.operand1",
        "arith.divf#1.result0->llvm.fptoui#0.operand0",
        "arith.select#0.result0->dataflow.load#1.operand1",
        "arith.select#0.result0->dataflow.store#1.operand1",
        "dataflow.load#1.result0->arith.addi#1.operand0",
        "llvm.fptoui#0.result0->arith.select#0.operand1",
        "llvm.uitofp#0.result0->arith.divf#0.operand1",
    }
    if not required_edges <= route_edges:
        raise AssertionError(f"hist_bin mapping missed required routed edges: {mapping_path}: {mapping}")

    cgra_path = evidence_dir / "hist_bin.cgra.report.json"
    cgra = json.loads(cgra_path.read_text())
    if (
        cgra.get("status") != "pass"
        or cgra.get("dfg_cycles") != 2571
        or cgra.get("hardware_aware_cycles") != 2733
        or cgra.get("performance_delta_cycles") != 162
        or cgra.get("placed_records") != 25
        or cgra.get("routed_edges") != 24
        or cgra.get("route_segments") != 114
        or cgra.get("config_records") != 644
        or cgra.get("final_memory_state", {}).get("arg2") != expected_hist
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
    ):
        raise AssertionError(f"hist_bin CGRA report should carry the real final histogram: {cgra_path}: {cgra}")

    comparison_path = evidence_dir / "hist_bin.sim-comparison-report.json"
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != 2571
        or comparison.get("cgra_sim_cycles") != 2733
    ):
        raise AssertionError(f"hist_bin comparison should pass with real memory-state checks: {comparison_path}: {comparison}")


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


def assert_primary_graph_missing(
    evidence_dir: Path,
    case: str,
    expected_token: str,
    expected_diagnostic: str,
    expected_discovered_graph: str,
    expected_residual_call: str,
) -> None:
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    mapping_path = evidence_dir / f"{case}.mapping.json"
    dfg = json.loads(dfg_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    for artifact_path, artifact in ((dfg_path, dfg), (mapping_path, mapping)):
        diagnostics = artifact.get("diagnostics")
        if not isinstance(diagnostics, list) or expected_diagnostic not in diagnostics:
            raise AssertionError(f"{case} should report primary graph absence: {artifact_path}: {artifact}")
        stale_diagnostic = f"primary workload graph absent: expected token {expected_token}"
        if stale_diagnostic != expected_diagnostic and stale_diagnostic in diagnostics:
            raise AssertionError(
                f"{case} should not keep stale generic primary graph diagnostic: {artifact_path}: {artifact}"
            )
    graph_ids = dfg.get("discovered_graph_ids")
    if not isinstance(graph_ids, list) or any(expected_token in str(graph_id) for graph_id in graph_ids):
        raise AssertionError(f"{case} should not expose its primary graph token yet: {dfg_path}: {dfg}")
    if expected_discovered_graph == EMPTY_DISCOVERED_GRAPH_IDS and graph_ids:
        raise AssertionError(f"{case} should not expose any discovered graph ids yet: {dfg_path}: {dfg}")
    if (
        expected_discovered_graph
        and expected_discovered_graph != EMPTY_DISCOVERED_GRAPH_IDS
        and expected_discovered_graph not in graph_ids
    ):
        raise AssertionError(f"{case} should prove supporting graph {expected_discovered_graph}: {dfg_path}: {dfg}")
    residual_calls = dfg.get("residual_call_targets")
    if expected_residual_call and (
        not isinstance(residual_calls, list) or expected_residual_call not in residual_calls
    ):
        raise AssertionError(f"{case} should prove residual call {expected_residual_call}: {dfg_path}: {dfg}")


def assert_partial_lowering_blocker(
    evidence_dir: Path,
    case: str,
    expected_diagnostic: str,
    expected_graph_token: str,
) -> None:
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    mapping_path = evidence_dir / f"{case}.mapping.json"
    dfg = json.loads(dfg_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    for artifact_path, artifact in ((dfg_path, dfg), (mapping_path, mapping)):
        diagnostics = artifact.get("diagnostics")
        if not isinstance(diagnostics, list) or expected_diagnostic not in diagnostics:
            raise AssertionError(f"{case} should report partial lowering: {artifact_path}: {artifact}")
    graph_ids = dfg.get("discovered_graph_ids")
    if not isinstance(graph_ids, list) or not any(expected_graph_token in str(graph_id) for graph_id in graph_ids):
        raise AssertionError(f"{case} should preserve the partial graph identity: {dfg_path}: {dfg}")


def assert_graph_present_unwired_blocker(evidence_dir: Path, case: str, expected_token: str) -> None:
    dfg_path = evidence_dir / f"{case}.dfg.report.json"
    mapping_path = evidence_dir / f"{case}.mapping.json"
    cgra_path = evidence_dir / f"{case}.cgra.report.json"
    dfg = json.loads(dfg_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    cgra = json.loads(cgra_path.read_text())
    graph_ids = dfg.get("discovered_graph_ids")
    if (
        dfg.get("status") != "unsupported"
        or not isinstance(graph_ids, list)
        or not any(expected_token in str(graph_id) for graph_id in graph_ids)
        or GRAPH_PRESENT_UNWIRED_DIAGNOSTIC not in dfg.get("diagnostics", [])
    ):
        raise AssertionError(f"{case} should expose a present-but-unwired primary graph: {dfg_path}: {dfg}")
    if (
        mapping.get("status") != "unsupported"
        or GRAPH_PRESENT_UNWIRED_DIAGNOSTIC not in mapping.get("diagnostics", [])
    ):
        raise AssertionError(f"{case} mapping should preserve the unwired graph blocker: {mapping_path}: {mapping}")
    if cgra.get("status") != "blocked":
        raise AssertionError(f"{case} CGRA report should remain blocked behind the DFG fixture: {cgra_path}: {cgra}")


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
    if row["status"] != "unsupported":
        raise AssertionError(f"{case} should surface DFG-sim unsupported evidence at row level: {row}")
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


def assert_shared_vector_mesh_evidence(repo: Path, out_dir: Path) -> None:
    evidence_dir = out_dir / "vector-mesh-current-sim-cycle"
    sweep_result = run(
        repo,
        [
            "bash",
            "test/e2e/run_cgra_sim_evidence_sweep.sh",
            "--output-dir",
            str(evidence_dir),
            "--case",
            "byte_swap",
            "--case",
            "xor_block",
            "--hardware-source",
            "shared-vector-mesh",
            "--jobs",
            "2",
        ],
    )
    statuses = parse_sweep_statuses(sweep_result.stdout)
    if statuses != {"byte_swap": "pass", "xor_block": "pass"}:
        raise AssertionError(f"vector mesh focused sweep should pass both rows: {statuses}")

    for case in ("byte_swap", "xor_block"):
        assert_sweep_artifact(evidence_dir, case, "dfg.report.json")
        assert_sweep_artifact(evidence_dir, case, "mapping.json")
        assert_sweep_artifact(evidence_dir, case, "cgra.report.json")
        assert_comparison_artifact(evidence_dir, case, "pass")
        assert_mapping_hardware(evidence_dir, case, "shared_vector_mesh_adg")
        assert_cgra_hardware(evidence_dir, case, "shared_vector_mesh_adg")
        assert_mapping_uses_switch_multihop(evidence_dir, case)

    status_csv = out_dir / "vector-mesh-status.csv"
    status_json = out_dir / "vector-mesh-status.json"
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cgra_status_summary.sh",
            "--output",
            str(status_csv),
            "--json-output",
            str(status_json),
            "--sim-evidence-dir",
            str(evidence_dir),
            "--no-legacy-loombench",
            "--no-cmsis-dfg-auto",
        ],
    )
    rows = read_rows(status_csv)
    for case in ("byte_swap", "xor_block"):
        row = one_row(rows, case)
        if row["status"] != "pass" or row["hardware_system"] != "shared_vector_mesh_adg":
            raise AssertionError(f"{case} should consume shared vector mesh evidence: {row}")


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
                "batchnorm",
                "--case",
                "binary_search",
                "--case",
                "bit_reverse",
                "--case",
                "bisection_step",
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
                "sort_bubble",
                "--case",
                "bitrev",
                "--case",
                "bitrev_complex",
                "--case",
                "spmspv",
                "--case",
                "stream_update",
                "--case",
                "gather",
                "--case",
                "gf_mul",
                "--case",
                "modmul",
                "--case",
                "modexp",
                "--case",
                "byte_swap",
                "--case",
                "cdma",
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
                "compact_predicate",
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
                "database_join",
                "--case",
                "prefix_sum_inclusive",
                "--case",
                "prefix_sum_exclusive",
                "--case",
                "lower_bound",
                "--case",
                "moving_avg",
                "--case",
                "pool_avg",
                "--case",
                "pool_max",
                "--case",
                "upsample_linear",
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
                "edge_update",
                "--case",
                "edge_update_batch",
                "--case",
                "bitonic_stage-modified",
                "--case",
                "col2im",
                "--case",
                "hist_bin",
                "--case",
                "histogram",
                "--case",
                "histogram_strided",
                "--case",
                "quantile",
                "--case",
                "sort_insertion",
                "--case",
                "sort_merge",
                "--case",
                "sort_quick",
                "--case",
                "spmspm",
                "--case",
                "string_compare",
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
                "conv2d",
                "--case",
                "im2col",
                "--case",
                "convolve_1d_same",
                "--case",
                "crc32",
                "--case",
                "cross_product",
                "--case",
                "quat_mult",
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
                "bitonic_stage",
                "--case",
                "bitonic_stage-tweak",
                "--case",
                "mmtile",
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
                "rle_decode",
                "--case",
                "rle_encode",
                "--case",
                "runge_kutta_step",
                "--case",
                "sbox_lookup",
                "--case",
                "sigmoid",
                "--case",
                "softmax",
                "--case",
                "window_blackman",
                "--case",
                "window_hamming",
                "--case",
                "window_hanning",
                "--case",
                "interpolate_linear",
                "--case",
                "jacobi_stencil_5pt",
                "--case",
                "jacobi_stencil_7pt",
                "--case",
                "distance_point",
                "--case",
                "line_intersect",
                "--case",
                "depthwise_conv",
                "--case",
                "edit_distance_step",
                "--case",
                "normalize",
                "--case",
                "normalize_vec3",
                "--case",
                "transpose",
                "--case",
                "transform_point",
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
            "sort_bubble",
            "bitrev",
            "bitrev_complex",
            "spmspv",
            "stream_update",
            "axpy",
            "batchnorm",
            "bit_reverse",
            "bisection_step",
            "byte_swap",
            "cdma",
            "bitonic_stage",
            "bitonic_stage-modified",
            "bitonic_stage-tweak",
            "clz",
            "ctz",
            "downsample",
            "xor_block",
            "vecmul",
            "vecscale",
            "prefix_sum",
            "cumsum",
            "database_join",
            "prefix_sum_inclusive",
            "prefix_sum_exclusive",
            "pack_bits",
            "unpack_bits",
            "partition",
            "popcount",
            "mean",
            "newton_iter",
            "outer",
            "vecnorm_l1",
            "vecnorm_l2",
            "gemv",
            "gemm",
            "matmul",
            "mmtile",
            "mat3x3_mult",
            "matvec",
            "downsample_avg",
            "vecadd",
            "conv1d",
            "conv2d",
            "im2col",
            "variance",
            "covariance",
            "cross_product",
            "quat_mult",
            "integrate_trapz",
            "delta_encode",
            "delta_decode",
            "correlation",
            "convolve_1d",
            "convolve_1d_same",
            "crc32",
            "fir_filter",
            "fir_filter_stateful",
            "find_first_set",
            "gf_mul",
            "compare_swap",
            "compact",
            "hash_mix",
            "string_hash",
            "merge",
            "modmul",
            "modexp",
            "moving_avg",
            "pool_avg",
            "pool_max",
            "quantile",
            "relu",
            "upsample",
            "upsample_linear",
            "sbox_lookup",
            "sigmoid",
            "softmax",
            "window_blackman",
            "window_hamming",
            "window_hanning",
            "interpolate_linear",
            "jacobi_stencil_5pt",
            "jacobi_stencil_7pt",
            "distance_point",
            "line_intersect",
            "depthwise_conv",
            "edit_distance_step",
            "normalize",
            "normalize_vec3",
            "rotate_bits",
            "rle_decode",
            "rle_encode",
            "runge_kutta_step",
            "transpose",
            "transform_point",
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
            assert_mapping_failed_evidence(evidence_dir, case)
        for case in MAPPING_BLOCKED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "pass")
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", "blocked")
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        for case in MAPPING_UNSUPPORTED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "pass")
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", "unsupported")
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        for case in DFG_BLOCKED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "blocked")
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", "pass")
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        assert_dfg_dynamic_work_items(evidence_dir, "gemm", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "batchnorm", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "matmul", 3)
        assert_dfg_dynamic_work_items(evidence_dir, "mat3x3_mult", 3)
        assert_dfg_dynamic_work_items(evidence_dir, "bitonic_stage", 4)
        assert_dfg_dynamic_work_items(evidence_dir, "bitonic_stage-modified", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "bitonic_stage-tweak", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "bisection_step", 1)
        assert_dfg_dynamic_work_items(evidence_dir, "modmul", 1)
        assert_dfg_dynamic_work_items(evidence_dir, "modexp", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "moving_avg", 16)
        assert_dfg_dynamic_work_items(evidence_dir, "pool_avg", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "pool_max", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "quantile", 1)
        assert_dfg_dynamic_work_items(evidence_dir, "newton_iter", 1)
        assert_dfg_dynamic_work_items(evidence_dir, "runge_kutta_step", 1)
        assert_dfg_dynamic_work_items(evidence_dir, "interpolate_linear", 63)
        assert_dfg_dynamic_work_items(evidence_dir, "jacobi_stencil_7pt", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "distance_point", 16)
        assert_dfg_dynamic_work_items(evidence_dir, "line_intersect", 64)
        assert_dfg_dynamic_work_items(evidence_dir, "database_join", 3)
        assert_dfg_dynamic_work_items(evidence_dir, "depthwise_conv", 432)
        assert_dfg_dynamic_work_items(evidence_dir, "edit_distance_step", 64)
        assert_dfg_dynamic_work_items(evidence_dir, "normalize", 23)
        assert_dfg_dynamic_work_items(evidence_dir, "normalize_vec3", 64)
        assert_dfg_dynamic_work_items(evidence_dir, "transform_point", 1)
        assert_dfg_dynamic_work_items(evidence_dir, "upsample", 4)
        assert_dfg_dynamic_work_items(evidence_dir, "upsample_linear", 16)
        assert_dfg_dynamic_work_items(evidence_dir, "sbox_lookup", 64)
        assert_dfg_dynamic_work_items(evidence_dir, "string_hash", 8)
        assert_dfg_dynamic_work_items(evidence_dir, "fir_filter_stateful", 4)
        assert_dfg_dynamic_work_items(evidence_dir, "covariance", 2048)
        assert_dfg_dynamic_work_items(evidence_dir, "popcount", 32)
        for case in ("clz", "ctz", "find_first_set", "parity"):
            assert_dfg_dynamic_work_items(evidence_dir, case, 32)
        assert_prefix_sum_exclusive_evidence(evidence_dir)
        assert_delta_decode_evidence(evidence_dir)
        assert_dot_product_3d_evidence(evidence_dir)
        assert_cross_product_evidence(evidence_dir)
        assert_quat_mult_evidence(evidence_dir)
        assert_spmspv_evidence(evidence_dir)
        run(repo, ["python3", "test/artifacts/assert_batchnorm_cgra_evidence.py", str(evidence_dir)])
        assert_mat3x3_mult_evidence(evidence_dir)
        assert_sigmoid_evidence(evidence_dir)
        assert_softmax_evidence(evidence_dir)
        for case in ("window_blackman", "window_hamming", "window_hanning"):
            run(repo, ["python3", "test/artifacts/assert_signal_window_cgra_evidence.py", "--case", case, str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_jacobi_stencil_5pt_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_jacobi_stencil_7pt_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_interpolate_linear_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_distance_point_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_line_intersect_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_database_join_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_depthwise_conv_cgra_evidence.py", str(evidence_dir)])
        assert_edit_distance_step_evidence(evidence_dir)
        run(repo, ["python3", "test/artifacts/assert_normalize_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_normalize_vec3_cgra_evidence.py", str(evidence_dir)])
        assert_mmtile_evidence(evidence_dir)
        assert_fir_filter_stateful_evidence(evidence_dir)
        assert_covariance_evidence(evidence_dir)
        assert_modmul_evidence(evidence_dir)
        assert_modexp_evidence(evidence_dir)
        assert_moving_avg_evidence(evidence_dir)
        assert_pool_avg_evidence(evidence_dir)
        assert_pool_max_evidence(evidence_dir)
        assert_upsample_linear_evidence(evidence_dir)
        assert_newton_iter_evidence(evidence_dir)
        assert_bisection_step_evidence(evidence_dir)
        assert_quantile_evidence(evidence_dir)
        assert_transform_point_evidence(evidence_dir)
        assert_rle_decode_evidence(evidence_dir)
        run(repo, ["python3", "test/artifacts/assert_rle_encode_cgra_evidence.py", str(evidence_dir)])
        assert_runge_kutta_step_evidence(evidence_dir)
        assert_gf_mul_evidence(evidence_dir)
        assert_compact_evidence(evidence_dir)
        assert_partition_evidence(evidence_dir)
        assert_outer_evidence(evidence_dir)
        assert_transpose_evidence(evidence_dir)
        assert_string_hash_evidence(evidence_dir)
        assert_autocorrelation_dfg_evidence(evidence_dir)
        assert_crc32_evidence(evidence_dir)
        assert_gather_evidence(evidence_dir)
        assert_histogram_evidence(evidence_dir)
        assert_hist_bin_evidence(evidence_dir)
        assert_histogram_strided_evidence(evidence_dir)
        assert_pack_bits_evidence(evidence_dir)
        assert_unpack_bits_evidence(evidence_dir)
        assert_bit_scan_evidence(
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
        assert_bit_scan_evidence(
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
        assert_bit_scan_evidence(
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
        assert_bit_scan_evidence(
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
        assert_popcount_evidence(evidence_dir)
        assert_binary_search_evidence(evidence_dir)
        assert_bitonic_stage_evidence(evidence_dir)
        assert_bitonic_stage_modified_evidence(evidence_dir)
        assert_bitonic_stage_tweak_evidence(evidence_dir)
        assert_bound_search_evidence(
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
        assert_bound_search_evidence(
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
        for case in DFG_UNSUPPORTED_SWEEP_CASES:
            assert_sweep_artifact_status(evidence_dir, case, "dfg.report.json", "unsupported")
            assert_sweep_artifact_status(evidence_dir, case, "mapping.json", "unsupported")
            assert_sweep_artifact_status(evidence_dir, case, "cgra.report.json", "blocked")
            assert_comparison_artifact(evidence_dir, case, "blocked")
        assert_merge_dfg_evidence(evidence_dir)
        for (
            case,
            expected_token,
            expected_diagnostic,
            expected_discovered_graph,
            expected_residual_call,
        ) in PRIMARY_GRAPH_MISSING_SWEEP_CASES:
            assert_primary_graph_missing(
                evidence_dir,
                case,
                expected_token,
                expected_diagnostic,
                expected_discovered_graph,
                expected_residual_call,
            )
        for case, expected_token in GRAPH_PRESENT_UNWIRED_SWEEP_CASES.items():
            assert_graph_present_unwired_blocker(evidence_dir, case, expected_token)
        for case, (expected_diagnostic, expected_graph_token) in PARTIAL_LOWERING_SWEEP_CASES.items():
            assert_partial_lowering_blocker(evidence_dir, case, expected_diagnostic, expected_graph_token)
        assert_mapping_hardware(evidence_dir, "dotproduct", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "dotprod", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "dot_product_3d", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "bisection_step", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "histogram", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "hist_bin", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "histogram_strided", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "mmtile", "shared_memory_reduction_adg")
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
        assert_mapping_hardware(evidence_dir, "binary_search", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "bitonic_stage", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "bitonic_stage-modified", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "bitonic_stage-tweak", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "bit_reverse", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "clz", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "ctz", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "downsample", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "delta_encode", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "delta_decode", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "cross_product", "shared_vector_math_adg")
        assert_mapping_hardware(evidence_dir, "quat_mult", "shared_vector_math_adg")
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
        assert_mapping_hardware(evidence_dir, "find_first_set", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "spmv", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "spmspv", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "stream_update", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "byte_swap", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "cdma", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "xor_block", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "vecmul", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "vecscale", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "cumsum", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "database_join", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum_inclusive", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum_exclusive", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "pack_bits", "shared_reduction_adg")
        assert_mapping_edges_use_switch_multihop(
            evidence_dir,
            "pack_bits",
            {
                "arith.ori#0.result0->dataflow.store#0.operand2",
                "arith.select#0.result0->arith.ori#0.operand0",
                "arith.shli#0.result0->arith.cmpi#0.operand0",
                "llvm.trunc#0.result0->arith.addi#0.operand0",
                "llvm.trunc#1.result0->arith.shli#1.operand1",
            },
        )
        assert_mapping_hardware(evidence_dir, "parity", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "unpack_bits", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "popcount", "shared_memory_reduction_adg")
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
        assert_mapping_hardware(evidence_dir, "conv2d", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "im2col", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "convolve_1d_same", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "crc32", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "gemm", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "matmul", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "mat3x3_mult", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "modmul", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "modexp", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "variance", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "covariance", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "correlation", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "autocorrelation", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "fir_filter", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "fir_filter_stateful", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "gather", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "lower_bound", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "moving_avg", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "jacobi_stencil_5pt", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "jacobi_stencil_7pt", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "pool_avg", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "pool_max", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "upsample_linear", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "batchnorm", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "edit_distance_step", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "outer", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "compare_swap", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "compact", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "compact_predicate", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "hash_mix", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "string_hash", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "merge", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "partition", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "sort_bubble", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "bitrev", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "bitrev_complex", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "scatter_add", "shared_memory_reduction_adg")
        assert_scatter_add_evidence(evidence_dir)
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
        assert_mapping_hardware(evidence_dir, "rle_decode", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "rle_encode", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "sbox_lookup", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "sigmoid", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "softmax", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "window_blackman", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "window_hamming", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "window_hanning", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "interpolate_linear", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "distance_point", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "line_intersect", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "depthwise_conv", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "normalize", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "normalize_vec3", "shared_signal_window_adg")
        assert_mapping_hardware(evidence_dir, "transpose", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "transform_point", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "upper_bound", "shared_memory_reduction_adg")
        assert_mapping_hardware(evidence_dir, "upsample", "shared_reduction_adg")
        for case in MAPPING_FAILED_SWEEP_CASES:
            assert_mapping_hardware(evidence_dir, case, "shared_reduction_adg")
        for case in MAPPING_BLOCKED_SWEEP_CASES:
            assert_mapping_hardware(evidence_dir, case, "shared_reduction_adg")
        for case in MAPPING_UNSUPPORTED_SWEEP_CASES:
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
            "crc32",
            {
                "arith.andi#0.result0->dataflow.load#1.operand1",
                "arith.shli#0.result0->arith.shrui#1.operand1",
                "arith.shrui#0.result0->arith.xori#1.operand0",
                "arith.shrui#1.result0->arith.xori#0.operand0",
                "arith.xori#0.result0->arith.andi#0.operand0",
                "dataflow.load#0.result0->arith.shrui#1.operand0",
                "dataflow.load#1.result0->arith.xori#1.operand1",
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
        assert_mapping_uses_switch_multihop(evidence_dir, "cdma")
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
        assert_mapping_uses_switch_multihop(evidence_dir, "upsample_linear")
        assert_mapping_uses_switch_multihop(evidence_dir, "convolve_1d")
        assert_mapping_uses_switch_multihop(evidence_dir, "relu")
        assert_mapping_uses_switch_multihop(evidence_dir, "sbox_lookup")
        assert_mapping_uses_switch_multihop(evidence_dir, "sigmoid")
        assert_mapping_uses_switch_multihop(evidence_dir, "window_blackman")
        assert_mapping_uses_switch_multihop(evidence_dir, "window_hamming")
        assert_mapping_uses_switch_multihop(evidence_dir, "window_hanning")
        assert_mapping_uses_switch_multihop(evidence_dir, "interpolate_linear")
        assert_mapping_uses_switch_multihop(evidence_dir, "jacobi_stencil_5pt")
        assert_mapping_uses_switch_multihop(evidence_dir, "jacobi_stencil_7pt")
        assert_mapping_uses_switch_multihop(evidence_dir, "moving_avg")
        assert_mapping_uses_switch_multihop(evidence_dir, "pool_avg")
        assert_mapping_uses_switch_multihop(evidence_dir, "pool_max")
        assert_mapping_uses_switch_multihop(evidence_dir, "outer")
        assert_mapping_uses_switch_multihop(evidence_dir, "sort_bubble")
        assert_mapping_uses_switch_multihop(evidence_dir, "bitrev")
        assert_mapping_uses_switch_multihop(evidence_dir, "bitrev_complex")
        assert_mapping_uses_switch_multihop(evidence_dir, "transpose")
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
        assert_mapping_uses_switch_multihop(evidence_dir, "softmax")
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
        assert_component_references_resolve(evidence_dir, "conv2d")
        assert_component_references_resolve(evidence_dir, "sort_bubble")
        run(repo, ["python3", "test/artifacts/assert_sort_bubble_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_bitrev_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_bitrev_complex_cgra_evidence.py", str(evidence_dir)])
        run(repo, ["python3", "test/artifacts/assert_conv2d_cgra_evidence.py", str(evidence_dir)])
        assert_cdma_evidence(evidence_dir)
        assert_compact_predicate_evidence(evidence_dir)
        assert_im2col_evidence(evidence_dir)
        run(repo, ["python3", "test/artifacts/assert_rle_encode_cgra_evidence.py", str(evidence_dir)])
        assert_component_references_resolve(evidence_dir, "variance")
        assert_component_references_resolve(evidence_dir, "covariance")
        assert_component_references_resolve(evidence_dir, "outer")
        assert_component_references_resolve(evidence_dir, "transpose")

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
            "sort_bubble",
            "bitrev",
            "bitrev_complex",
            "stream_update",
            "axpy",
            "bit_reverse",
            "byte_swap",
            "cdma",
            "downsample",
            "xor_block",
            "vecmul",
            "prefix_sum",
            "cumsum",
            "database_join",
            "prefix_sum_inclusive",
            "prefix_sum_exclusive",
            "pack_bits",
            "parity",
            "unpack_bits",
            "partition",
            "mean",
            "vecnorm_l1",
            "vecnorm_l2",
            "gemv",
            "gemm",
            "matmul",
            "mat3x3_mult",
            "mmtile",
            "matvec",
            "downsample_avg",
            "vecadd",
            "vecscale",
            "conv1d",
            "conv2d",
            "variance",
            "covariance",
            "cross_product",
            "quat_mult",
            "integrate_trapz",
            "delta_encode",
            "delta_decode",
            "correlation",
            "convolve_1d",
            "convolve_1d_same",
            "crc32",
            "binary_search",
            "bitonic_stage",
            "bitonic_stage-tweak",
            "fir_filter_stateful",
            "compare_swap",
            "compact",
            "gather",
            "histogram",
            "hash_mix",
            "string_hash",
            "merge",
            "lower_bound",
            "modmul",
            "modexp",
            "quantile",
            "relu",
            "upsample",
            "sbox_lookup",
            "sigmoid",
            "softmax",
            "window_blackman",
            "window_hamming",
            "window_hanning",
            "interpolate_linear",
            "jacobi_stencil_5pt",
            "jacobi_stencil_7pt",
            "distance_point",
            "line_intersect",
            "edit_distance_step",
            "normalize",
            "normalize_vec3",
            "rotate_bits",
            "rle_decode",
            "rle_encode",
            "runge_kutta_step",
            "autocorrelation",
            "upper_bound",
            "scatter_add",
            "quat_mult",
        ):
            assert_promoted_row(repo, rows, case)
        for case in MAPPING_FAILED_SWEEP_CASES:
            assert_structured_blocker_row(repo, rows, case, "fail", "fail")
        for case in MAPPING_BLOCKED_SWEEP_CASES:
            assert_structured_blocker_row(repo, rows, case, "blocked", "blocked")
        for case in MAPPING_UNSUPPORTED_SWEEP_CASES:
            assert_structured_blocker_row(repo, rows, case, "blocked", "unsupported")
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
        database_join_row = one_row(rows, "database_join")
        if database_join_row["hardware_system"] != "shared_memory_reduction_adg":
            raise AssertionError(f"database_join should use shared memory-reduction hardware: {database_join_row}")
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
        cross_product_row = one_row(rows, "cross_product")
        if cross_product_row["hardware_system"] != "shared_vector_math_adg":
            raise AssertionError(f"cross_product should use shared vector math hardware: {cross_product_row}")
        quat_mult_row = one_row(rows, "quat_mult")
        if quat_mult_row["hardware_system"] != "shared_vector_math_adg":
            raise AssertionError(f"quat_mult should use shared vector math hardware: {quat_mult_row}")
        interpolate_row = one_row(rows, "interpolate_linear")
        if interpolate_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"interpolate_linear should use shared signal-window hardware: {interpolate_row}")
        distance_point_row = one_row(rows, "distance_point")
        if distance_point_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"distance_point should use shared signal-window hardware: {distance_point_row}")
        line_intersect_row = one_row(rows, "line_intersect")
        if line_intersect_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"line_intersect should use shared signal-window hardware: {line_intersect_row}")
        depthwise_conv_row = one_row(rows, "depthwise_conv")
        if depthwise_conv_row["hardware_system"] != "shared_memory_reduction_adg":
            raise AssertionError(f"depthwise_conv should use shared memory-reduction hardware: {depthwise_conv_row}")
        normalize_row = one_row(rows, "normalize")
        if normalize_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"normalize should use shared signal-window hardware: {normalize_row}")
        normalize_vec3_row = one_row(rows, "normalize_vec3")
        if normalize_vec3_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"normalize_vec3 should use shared signal-window hardware: {normalize_vec3_row}")
        moving_avg_row = one_row(rows, "moving_avg")
        if moving_avg_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"moving_avg should use shared signal-window hardware: {moving_avg_row}")
        pool_avg_row = one_row(rows, "pool_avg")
        if pool_avg_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"pool_avg should use shared signal-window hardware: {pool_avg_row}")
        pool_max_row = one_row(rows, "pool_max")
        if pool_max_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"pool_max should use shared signal-window hardware: {pool_max_row}")
        upsample_linear_row = one_row(rows, "upsample_linear")
        if upsample_linear_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"upsample_linear should use shared signal-window hardware: {upsample_linear_row}")
        jacobi_row = one_row(rows, "jacobi_stencil_5pt")
        if jacobi_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"jacobi_stencil_5pt should use shared signal-window hardware: {jacobi_row}")
        jacobi7_row = one_row(rows, "jacobi_stencil_7pt")
        if jacobi7_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"jacobi_stencil_7pt should use shared signal-window hardware: {jacobi7_row}")
        quantile_row = one_row(rows, "quantile")
        if quantile_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"quantile should use shared signal-window hardware: {quantile_row}")
        hist_bin_row = one_row(rows, "hist_bin")
        if hist_bin_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"hist_bin should use shared signal-window hardware: {hist_bin_row}")
        histogram_row = one_row(rows, "histogram")
        if histogram_row["hardware_system"] != "shared_memory_reduction_adg":
            raise AssertionError(f"histogram should use shared memory-reduction hardware: {histogram_row}")
        histogram_strided_row = one_row(rows, "histogram_strided")
        if histogram_strided_row["hardware_system"] != "shared_memory_reduction_adg":
            raise AssertionError(
                f"histogram_strided should use shared memory-reduction hardware: {histogram_strided_row}"
            )
        batchnorm_row = one_row(rows, "batchnorm")
        if batchnorm_row["hardware_system"] != "shared_signal_window_adg":
            raise AssertionError(f"batchnorm should use shared signal-window hardware: {batchnorm_row}")
        downsample_row = one_row(rows, "downsample_avg")
        if downsample_row["hardware_system"] != "shared_reduction_adg":
            raise AssertionError(f"downsample_avg should use shared reduction hardware: {downsample_row}")
        counts = json.loads(status_json.read_text())["counts"]["app"]
        expected_counts = {
            "total": 121,
            "pass": 113,
            "fail": 0,
            "blocked": 0,
            "unsupported": 8,
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
        assert_shared_vector_mesh_evidence(repo, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
