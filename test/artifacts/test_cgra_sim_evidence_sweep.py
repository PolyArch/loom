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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <repo>")
    repo = Path(argv[1]).resolve()
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
                "vecsum",
                "--case",
                "reduction",
                "--case",
                "dotproduct",
                "--case",
                "byte_swap",
                "--case",
                "xor_block",
                "--case",
                "vecmul",
                "--case",
                "prefix_sum",
                "--case",
                "cumsum",
                "--case",
                "prefix_sum_inclusive",
                "--case",
                "mean",
                "--case",
                "vecnorm_l1",
                "--case",
                "vecnorm_l2",
                "--case",
                "matvec",
                "--case",
                "vecadd",
            ],
        )
        for case in (
            "vecsum",
            "reduction",
            "dotproduct",
            "byte_swap",
            "xor_block",
            "vecmul",
            "prefix_sum",
            "cumsum",
            "prefix_sum_inclusive",
            "mean",
            "vecnorm_l1",
            "vecnorm_l2",
            "matvec",
        ):
            assert_sweep_artifact(evidence_dir, case, "dfg.report.json")
            assert_sweep_artifact(evidence_dir, case, "mapping.json")
            assert_sweep_artifact(evidence_dir, case, "cgra.report.json")
            assert_comparison_artifact(evidence_dir, case, "pass")
        assert_mapping_hardware(evidence_dir, "dotproduct", "dotproduct_fmuladd_adg")
        assert_mapping_hardware(evidence_dir, "byte_swap", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "xor_block", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "vecmul", "shared_vector_alu_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "cumsum", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "prefix_sum_inclusive", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "mean", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "vecnorm_l1", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "vecnorm_l2", "shared_reduction_adg")
        assert_mapping_hardware(evidence_dir, "matvec", "shared_reduction_adg")
        assert_mapping_uses_switch_multihop(evidence_dir, "byte_swap")
        assert_mapping_uses_switch_multihop(evidence_dir, "xor_block")
        assert_mapping_uses_switch_multihop(evidence_dir, "vecmul")
        assert_mapping_uses_switch_multihop(evidence_dir, "matvec")
        assert_comparison_artifact(evidence_dir, "vecadd", "blocked")
        assert_component_references_resolve(evidence_dir, "vecadd")

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
            "reduction",
            "dotproduct",
            "byte_swap",
            "xor_block",
            "vecmul",
            "prefix_sum",
            "cumsum",
            "prefix_sum_inclusive",
            "mean",
            "vecnorm_l1",
            "vecnorm_l2",
            "matvec",
        ):
            assert_promoted_row(repo, rows, case)
        dotproduct_row = one_row(rows, "dotproduct")
        if dotproduct_row["hardware_system"] != "dotproduct_fmuladd_adg":
            raise AssertionError(f"dotproduct should use fmuladd hardware: {dotproduct_row}")
        byte_swap_row = one_row(rows, "byte_swap")
        if byte_swap_row["hardware_system"] != "shared_vector_alu_adg":
            raise AssertionError(f"byte_swap should use shared vector hardware: {byte_swap_row}")
        xor_block_row = one_row(rows, "xor_block")
        if xor_block_row["hardware_system"] != "shared_vector_alu_adg":
            raise AssertionError(f"xor_block should use shared vector hardware: {xor_block_row}")
        vecmul_row = one_row(rows, "vecmul")
        if vecmul_row["hardware_system"] != "shared_vector_alu_adg":
            raise AssertionError(f"vecmul should use shared vector hardware: {vecmul_row}")
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
        counts = json.loads(status_json.read_text())["counts"]["app"]
        if counts["pass"] < 13:
            raise AssertionError(f"app pass count should include sweep cases: {counts}")
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
