#!/usr/bin/env python3
"""Regression tests for the intermediate artifact gate scaffold."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import artifact_test_common


CSV_COMMANDS = [
    (
        "test/app/run_source_compat_summary.sh",
        "source-compat-summary.csv",
        ["case", "suite", "native_status", "loom_status", "mode", "diagnostic"],
    ),
    (
        "test/app/run_compiler_pipeline_summary.sh",
        "compiler-pipeline-summary.csv",
        [
            "case",
            "suite",
            "llvm_ir_status",
            "raised_mlir_status",
            "dataflow_status",
            "diagnostic",
        ],
    ),
    (
        "test/dataflow/run_primitive_coverage.sh",
        "dataflow-primitive-coverage.csv",
        ["workload", "primitive", "op_count", "dfg_sim_status", "diagnostic"],
    ),
    (
        "test/fabric/run_adg_hardware_summary.sh",
        "adg-hardware-summary.csv",
        ["hardware", "topology_class", "node_count", "link_count", "verify_status", "diagnostic"],
    ),
    (
        "test/pnr/run_mapping_summary.sh",
        "pnr-mapping-summary.csv",
        [
            "workload",
            "hardware",
            "mapping_id",
            "placed_records",
            "routed_edges",
            "unrouted_edges",
            "unplaced_records",
            "status",
        ],
    ),
    (
        "test/app/run_sim_cycle_summary.sh",
        "sim-cycle-summary.csv",
        ["kernel", "dfg_sim_cycles", "cgra_sim_cycles"],
    ),
    (
        "test/rtl/run_rtl_fpa_summary.sh",
        "rtl-fpa-summary.csv",
        [
            "hardware",
            "workload",
            "rtl_lint_status",
            "rtl_sim_status",
            "synth_status",
            "frequency_mhz",
            "area_um2",
            "dynamic_power_mw",
            "leakage_power_mw",
        ],
    ),
    (
        "test/e2e/run_demonstrator_summary.sh",
        "e2e-demonstrator-summary.csv",
        [
            "demonstrator",
            "compat_status",
            "artifact_status",
            "mapping_status",
            "sim_status",
            "rtl_status",
            "fpa_status",
            "report_status",
        ],
    ),
    (
        "test/dse/run_candidate_summary.sh",
        "dse-candidate-summary.csv",
        [
            "candidate",
            "workload",
            "hardware",
            "mapping_id",
            "objective",
            "cgra_sim_cycles",
            "frequency_mhz",
            "area_um2",
            "dynamic_power_mw",
            "leakage_power_mw",
            "energy_nj",
            "selection_status",
        ],
    ),
    (
        "test/e2e/run_unsupported_scope_ledger.sh",
        "unsupported-scope-ledger.csv",
        ["stage", "case", "artifact", "reason", "owner", "blocking_input"],
    ),
]


JSON_COMMANDS = [
    (
        "test/e2e/run_artifact_manifest.sh",
        "full-stack-artifact-manifest.json",
        {"schema_version", "run_id", "artifacts", "edges", "diagnostics"},
    ),
]

RTL_MANIFEST_REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "manifest_id",
    "source_fabric_adg_identity",
    "emitted_source_files",
    "top_level_modules",
    "diagnostics",
    "status",
}


def run_command(repo: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LOOM_IGNORE_STANDARD_ARTIFACTS"] = "1"
    return subprocess.run(
        argv,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames or [], rows

def semicolon_fingerprints(paths: list[Path]) -> str:
    return ";".join(f"{path}={artifact_test_common.fingerprint(path)}" for path in paths)


def assert_csv_artifact(
    path: Path,
    required_first_columns: list[str],
    *,
    allow_pass_rows: bool = False,
) -> None:
    header, rows = read_csv(path)
    if header[: len(required_first_columns)] != required_first_columns:
        raise AssertionError(
            f"{path.name}: header {header[:len(required_first_columns)]} "
            f"does not match {required_first_columns}"
        )
    if not rows:
        raise AssertionError(f"{path.name}: expected at least one diagnostic row")
    for row in rows:
        if None in row:
            raise AssertionError(f"{path.name}: row has extra unnamed cells: {row}")
        missing = [key for key, value in row.items() if value is None]
        if missing:
            raise AssertionError(f"{path.name}: row is missing values for {missing}")
    statuses = []
    for row in rows:
        statuses.extend(
            value
            for key, value in row.items()
            if key.endswith("_status") or key in {"status", "selection_status"}
        )
    if "pass" in statuses and not allow_pass_rows:
        raise AssertionError(f"{path.name}: scaffold rows must not claim pass evidence")


def assert_json_artifact(path: Path, required_keys: set[str]) -> None:
    data = json.loads(path.read_text())
    missing = sorted(required_keys - set(data))
    if missing:
        raise AssertionError(f"{path.name}: missing keys {missing}")
    if data.get("schema_version") != 1:
        raise AssertionError(f"{path.name}: schema_version must be 1")


def assert_manifest_trace_edges(path: Path) -> None:
    data = json.loads(path.read_text())
    artifact_ids = {artifact.get("id") for artifact in data.get("artifacts", [])}
    required_ids = {
        "dataflow-primitive-coverage",
        "adg-hardware-summary",
        "pnr-mapping-summary",
        "pnr-mapping",
        "vecsum-dfg-sim-report",
        "vecsum-cgra-sim-report",
        "sim-cycle-summary",
        "rtl-manifest",
        "dse-candidate-summary",
    }
    if not required_ids <= artifact_ids:
        missing = sorted(required_ids - artifact_ids)
        raise AssertionError(f"{path.name}: missing trace artifact ids {missing}")

    edge_pairs = {(edge.get("from"), edge.get("to")) for edge in data.get("edges", [])}
    required_edges = {
        ("dataflow-primitive-coverage", "pnr-mapping"),
        ("adg-hardware-summary", "pnr-mapping"),
        ("pnr-mapping-summary", "pnr-mapping"),
        ("dataflow-primitive-coverage", "vecsum-dfg-sim-report"),
        ("pnr-mapping", "vecsum-cgra-sim-report"),
        ("vecsum-dfg-sim-report", "sim-cycle-summary"),
        ("vecsum-cgra-sim-report", "sim-cycle-summary"),
        ("adg-hardware-summary", "rtl-manifest"),
        ("rtl-manifest", "rtl-fpa-summary"),
        ("pnr-mapping", "dse-candidate-summary"),
        ("vecsum-cgra-sim-report", "dse-candidate-summary"),
    }
    if not required_edges <= edge_pairs:
        missing = sorted(required_edges - edge_pairs)
        raise AssertionError(f"{path.name}: missing trace edges {missing}")


def assert_manifest_audit_counts_artifact_records(audit_data: dict[str, object], manifest_path: Path) -> None:
    manifest_data = json.loads(manifest_path.read_text())
    expected_count = len(manifest_data.get("artifacts", []))
    for review in audit_data.get("artifact_reviews", []):
        if not isinstance(review, dict):
            continue
        if review.get("schema") != "artifact_manifest":
            continue
        if Path(str(review.get("artifact"))).resolve() != manifest_path.resolve():
            continue
        if review.get("entries_checked") != expected_count:
            raise AssertionError(
                f"{manifest_path.name}: audit checked {review.get('entries_checked')} "
                f"entries, expected {expected_count} artifact records"
            )
        return
    raise AssertionError(f"{manifest_path.name}: audit review for manifest was not found")


def write_dfg_report(
    path: Path,
    workload: str,
    graph: str,
    cycles: int,
    final_outputs: list[str] | None = None,
    dynamic_work_items: int = 1,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "dfg_sim_report",
                "workload": workload,
                "graph": graph,
                "status": "pass",
                "metric_definition": "optimistic_pipeline_latency_throughput_sum",
                "operation_semantics_source": "loom.sim.operation_semantics.v1",
                "operation_cost_model_source": "loom.sim.operation_cost.v1",
                "optimistic_cycles": cycles,
                "wavefront_steps": min(cycles, 4),
                "event_count": min(cycles, 10),
                "dynamic_work_items": dynamic_work_items,
                "operation_fire_counts": {"dataflow.stream": dynamic_work_items},
                "final_outputs": final_outputs if final_outputs is not None else ["none"],
                "diagnostics": [],
            }
        )
    )


def write_mapping_artifact(path: Path, workload: str, graph: str, mapping_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pnr_mapping",
                "workload": workload,
                "hardware": "fabric0",
                "graph": graph,
                "mapping_id": mapping_id,
                "status": "pass",
                "placed_records": 1,
                "routed_edges": 1,
                "unrouted_edges": 0,
                "unplaced_records": 0,
                "config_records": 0,
                "placements": [
                    {
                        "software": f"{graph}#op0",
                        "operation": "arith.addi",
                        "resource_kind": "fabric.op",
                        "hardware": "fabric0::fabric.op#0",
                        "schedule": "spatial",
                    }
                ],
                "routes": [
                    {
                        "record_id": "route#0",
                        "edge_ref": f"{graph}#op0.result0->{graph}#op1.operand0",
                        "producer_binding": f"placement:{graph}#op0",
                        "consumer_binding": f"placement:{graph}#op1",
                        "payload_kind": "data",
                        "from": f"{graph}#op0",
                        "to": f"{graph}#op1",
                        "status": "routed",
                        "segments": [
                            {
                                "segment_id": "seg0",
                                "segment_kind": "module_path",
                                "source_endpoint": "fabric0::fabric.op#0.out",
                                "sink_endpoint": "fabric0::fabric.op#1.in",
                            }
                        ],
                    }
                ],
                "config_bitstream": [],
            }
        )
    )


def write_cgra_report(
    path: Path,
    workload: str,
    mapping_id: str,
    dfg_cycles: int,
    cgra_cycles: int,
) -> None:
    delta = cgra_cycles - dfg_cycles
    route_cycles = 1 if delta > 0 else 0
    memory_cycles = delta - route_cycles
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "cgra_sim_report",
                "workload": workload,
                "hardware": "fabric0",
                "mapping_id": mapping_id,
                "status": "pass",
                "fidelity_level": "mapping_constraint_estimate",
                "metric_definition": "mapping_constraint_estimate",
                "operation_semantics_source": "loom.sim.operation_semantics.v1",
                "operation_cost_model_source": "loom.sim.operation_cost.v1",
                "difference_classification": "expected_hardware_constraint"
                if delta > 0
                else "no_modeled_hardware_constraints",
                "hardware_bound_classification": "within_modeled_bounds",
                "dfg_cycles": dfg_cycles,
                "modeled_lower_bound_cycles": cgra_cycles,
                "performance_delta_cycles": delta,
                "route_latency_cycles": route_cycles,
                "memory_latency_cycles": memory_cycles,
                "temporal_penalty_cycles": 0,
                "hardware_aware_cycles": cgra_cycles,
                "cycle_breakdown": [
                    {
                        "category": "route_latency",
                        "cycles": route_cycles,
                        "evidence": "mapping.route_segments",
                    },
                    {
                        "category": "memory_latency",
                        "cycles": memory_cycles,
                        "evidence": "fabric.mem placement",
                    },
                ],
                "unmodeled_constraints": ["cache_behavior"],
                "first_principles_checks": [
                    {
                        "name": "cgra_not_more_optimistic_than_dfg",
                        "status": "pass",
                        "evidence": "hardware_aware_cycles >= dfg_cycles",
                    },
                    {
                        "name": "delta_explained_by_modeled_constraints",
                        "status": "pass",
                        "evidence": "performance_delta_cycles = modeled penalties",
                    },
                ],
                "diagnostics": ["synthetic checked CGRA report"],
            }
        )
    )


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-artifacts-") as tmp:
        out_dir = Path(tmp)
        produced: list[Path] = []

        for script, filename, required_columns in CSV_COMMANDS:
            output = out_dir / filename
            result = run_command(repo, ["bash", script, "--output", str(output)])
            if result.returncode != 0:
                raise AssertionError(
                    f"{script} failed with {result.returncode}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            assert_csv_artifact(
                output,
                required_columns,
                allow_pass_rows=filename
                in {
                    "source-compat-summary.csv",
                    "compiler-pipeline-summary.csv",
                    "dataflow-primitive-coverage.csv",
                    "adg-hardware-summary.csv",
                    "sim-cycle-summary.csv",
                    "rtl-fpa-summary.csv",
                    "dse-candidate-summary.csv",
                },
            )
            produced.append(output)
            if filename == "adg-hardware-summary.csv":
                rtl_manifest = out_dir / "rtl-manifest.json"
                result = run_command(
                    repo,
                    [
                        "bash",
                        "test/rtl/run_rtl_manifest.sh",
                        "--hardware-summary",
                        str(output),
                        "--output",
                        str(rtl_manifest),
                    ],
                )
                if result.returncode != 0:
                    raise AssertionError(
                        f"test/rtl/run_rtl_manifest.sh failed with {result.returncode}\n"
                        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                    )
                assert_json_artifact(rtl_manifest, RTL_MANIFEST_REQUIRED_KEYS)
                produced.append(rtl_manifest)
            if filename == "sim-cycle-summary.csv":
                for backing_name in (
                    "pnr-mapping.json",
                    "vecsum-dfg-sim-report.json",
                    "vecsum-cgra-sim-report.json",
                ):
                    backing = out_dir / backing_name
                    if backing.is_file():
                        produced.append(backing)

        for script, filename, required_keys in JSON_COMMANDS:
            output = out_dir / filename
            command = ["bash", script]
            if filename == "full-stack-artifact-manifest.json":
                for artifact in produced:
                    command.extend(["--artifact", str(artifact)])
            command.extend(["--output", str(output)])
            result = run_command(repo, command)
            if result.returncode != 0:
                raise AssertionError(
                    f"{script} failed with {result.returncode}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            assert_json_artifact(output, required_keys)
            if filename == "full-stack-artifact-manifest.json":
                assert_manifest_trace_edges(output)
            produced.append(output)

        audit_pass = out_dir / "artifact-audit-summary.json"
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit_pass),
                *[str(path) for path in produced],
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                f"audit failed with {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        audit_data = json.loads(audit_pass.read_text())
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected pass audit, got {audit_data}")
        assert_manifest_audit_counts_artifact_records(
            audit_data,
            out_dir / "full-stack-artifact-manifest.json",
        )

        invalid = out_dir / "invalid-sim-cycle-summary.csv"
        invalid.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "bad,0,0,pass,\n"
        )
        audit_fail = out_dir / "artifact-audit-summary-fail.json"
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit_fail),
                str(invalid),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("invalid artifact audit unexpectedly exited zero")
        audit_data = json.loads(audit_fail.read_text())
        if audit_data.get("verdict") != "fail":
            raise AssertionError(f"expected fail audit, got {audit_data}")

        valid_primitive = out_dir / "valid-dataflow-primitive-coverage.csv"
        valid_primitive.write_text(
            "workload,primitive,op_count,dfg_sim_status,diagnostic\n"
            "vecadd,stream,1,blocked,DFG-sim is not implemented\n"
        )
        valid_hardware = out_dir / "valid-adg-hardware-summary.csv"
        valid_hardware.write_text(
            "hardware,topology_class,node_count,link_count,verify_status,diagnostic\n"
            "fabric0,fabric_module_template,1,0,pass,verified\n"
        )
        stale_mapping = out_dir / "stale-pnr-mapping-summary.csv"
        stale_mapping.write_text(
            "workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic\n"
            "ghost,missing_hw,,,,,,blocked,stale candidate references\n"
        )
        audit_cross = out_dir / "artifact-audit-summary-cross-fail.json"
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit_cross),
                str(valid_primitive),
                str(valid_hardware),
                str(stale_mapping),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("stale cross-artifact audit unexpectedly exited zero")
        audit_data = json.loads(audit_cross.read_text())
        if audit_data.get("verdict") != "fail":
            raise AssertionError(f"expected cross-artifact fail audit, got {audit_data}")
        findings = audit_data.get("cross_artifact_findings", [])
        if not findings:
            raise AssertionError(f"expected cross-artifact findings, got {audit_data}")
        messages = " ".join(str(finding) for finding in findings)
        if "ghost" not in messages or "missing_hw" not in messages:
            raise AssertionError(f"cross findings should identify stale refs: {findings}")

        zero_pass_primitive = out_dir / "zero-pass-dataflow-primitive-coverage.csv"
        zero_pass_primitive.write_text(
            "workload,primitive,op_count,dfg_sim_status,diagnostic\n"
            "vecadd,stream,0,pass,simulator evidence without operations\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-zero-pass-primitive.json"),
                str(zero_pass_primitive),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("primitive pass row with zero op_count unexpectedly passed audit")

        invalid_optimistic_sim = out_dir / "optimistic-sim-cycle-summary.csv"
        invalid_optimistic_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,9,pass,\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-optimistic-sim.json"),
                str(invalid_optimistic_sim),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA-sim cycles below DFG-sim cycles unexpectedly passed audit")

        duplicate_sim = out_dir / "duplicate-sim-cycle-summary.csv"
        duplicate_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,64,80,pass,synthetic vecadd report\n"
            "conv1d,64,80,pass,synthetic conv1d report\n"
        )
        duplicate_audit = out_dir / "artifact-audit-summary-duplicate-sim.json"
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(duplicate_audit),
                str(duplicate_sim),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("duplicate simulator cycle values unexpectedly passed audit")
        audit_data = json.loads(duplicate_audit.read_text())
        messages = " ".join(str(item) for item in audit_data.get("diagnostics", []))
        if "DFG-sim cycles 64" not in messages or "CGRA-sim cycles 80" not in messages:
            raise AssertionError(f"duplicate simulator diagnostics missing: {audit_data}")

        unequal_extent_reduction_sim = out_dir / "unequal-extent-reduction-sim-cycle-summary.csv"
        unequal_extent_reduction_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecsum,579,589,pass,integer reduction with 64 items\n"
            "reduction,579,589,pass,integer reduction with 128 items\n"
        )
        write_dfg_report(
            out_dir / "vecsum-dfg-sim-report.json",
            "vecsum",
            "g_vecsum",
            579,
            dynamic_work_items=64,
        )
        write_dfg_report(
            out_dir / "reduction-dfg-sim-report.json",
            "reduction",
            "g_reduction",
            579,
            dynamic_work_items=128,
        )
        write_mapping_artifact(
            out_dir / "vecsum-pnr-mapping.json",
            "vecsum",
            "g_vecsum",
            "map_vecsum",
        )
        write_mapping_artifact(
            out_dir / "reduction-pnr-mapping.json",
            "reduction",
            "g_reduction",
            "map_reduction",
        )
        write_cgra_report(
            out_dir / "vecsum-cgra-sim-report.json",
            "vecsum",
            "map_vecsum",
            579,
            589,
        )
        write_cgra_report(
            out_dir / "reduction-cgra-sim-report.json",
            "reduction",
            "map_reduction",
            579,
            589,
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-unequal-extent-reduction.json"),
                str(unequal_extent_reduction_sim),
                str(out_dir / "vecsum-dfg-sim-report.json"),
                str(out_dir / "reduction-dfg-sim-report.json"),
                str(out_dir / "vecsum-pnr-mapping.json"),
                str(out_dir / "reduction-pnr-mapping.json"),
                str(out_dir / "vecsum-cgra-sim-report.json"),
                str(out_dir / "reduction-cgra-sim-report.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError(
                "unequal dynamic extent reduction simulator cycles unexpectedly passed audit"
            )

        monotonic_bad_n64 = out_dir / "monotonic-bad-n64-dfg-sim-report.json"
        monotonic_bad_n128 = out_dir / "monotonic-bad-n128-dfg-sim-report.json"
        write_dfg_report(
            monotonic_bad_n64,
            "scale",
            "g_scale",
            100,
            dynamic_work_items=64,
        )
        write_dfg_report(
            monotonic_bad_n128,
            "scale",
            "g_scale",
            100,
            dynamic_work_items=128,
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-monotonic-bad.json"),
                str(monotonic_bad_n64),
                str(monotonic_bad_n128),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("non-monotonic DFG scale reports unexpectedly passed audit")

        monotonic_good_n64 = out_dir / "monotonic-good-n64-dfg-sim-report.json"
        monotonic_good_n128 = out_dir / "monotonic-good-n128-dfg-sim-report.json"
        write_dfg_report(
            monotonic_good_n64,
            "scale",
            "g_scale",
            100,
            dynamic_work_items=64,
        )
        write_dfg_report(
            monotonic_good_n128,
            "scale",
            "g_scale",
            180,
            dynamic_work_items=128,
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-monotonic-good.json"),
                str(monotonic_good_n64),
                str(monotonic_good_n128),
            ],
        )
        if result.returncode != 0:
            raise AssertionError("monotonic DFG scale reports unexpectedly failed audit")

        aggregate_slice_sim = out_dir / "aggregate-slice-sim-cycle-summary.csv"
        aggregate_slice_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,1603,1631,pass,core graph and checksum reduction slices\n"
        )
        write_dfg_report(out_dir / "vecadd-core-dfg-sim-report.json", "vecadd", "g_vecadd", 960)
        write_dfg_report(
            out_dir / "vecadd-reduction-dfg-sim-report.json",
            "vecadd",
            "g_main_red",
            643,
        )
        write_mapping_artifact(
            out_dir / "vecadd-core-pnr-mapping.json",
            "vecadd",
            "g_vecadd",
            "map_vecadd_core",
        )
        write_mapping_artifact(
            out_dir / "vecadd-reduction-pnr-mapping.json",
            "vecadd",
            "g_main_red",
            "map_vecadd_reduction",
        )
        write_cgra_report(
            out_dir / "vecadd-core-cgra-sim-report.json",
            "vecadd",
            "map_vecadd_core",
            960,
            978,
        )
        write_cgra_report(
            out_dir / "vecadd-reduction-cgra-sim-report.json",
            "vecadd",
            "map_vecadd_reduction",
            643,
            653,
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-aggregate-slices.json"),
                str(aggregate_slice_sim),
                str(out_dir / "vecadd-core-dfg-sim-report.json"),
                str(out_dir / "vecadd-reduction-dfg-sim-report.json"),
                str(out_dir / "vecadd-core-pnr-mapping.json"),
                str(out_dir / "vecadd-reduction-pnr-mapping.json"),
                str(out_dir / "vecadd-core-cgra-sim-report.json"),
                str(out_dir / "vecadd-reduction-cgra-sim-report.json"),
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                "aggregate simulator slices unexpectedly failed audit"
            )

        decimal_sim = out_dir / "decimal-sim-cycle-summary.csv"
        decimal_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10.5,12,pass,synthetic decimal cycle\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-decimal-sim.json"),
                str(decimal_sim),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("decimal simulator cycle values unexpectedly passed audit")

        standalone_dfg_cycle = out_dir / "standalone-dfg-sim-cycle-summary.csv"
        standalone_dfg_cycle.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,,blocked,synthetic standalone DFG cycle\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-standalone-dfg.json"),
                str(standalone_dfg_cycle),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("standalone DFG cycle without evidence unexpectedly passed audit")

        primitive_blocked = out_dir / "blocked-dataflow-primitive-coverage.csv"
        primitive_blocked.write_text(
            "workload,primitive,op_count,dfg_sim_status,diagnostic\n"
            "vecadd,stream,1,blocked,primitive-count proxy only\n"
        )
        dfg_from_blocked_proxy = out_dir / "dfg-from-blocked-proxy-sim-cycle-summary.csv"
        dfg_from_blocked_proxy.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,,blocked,synthetic DFG cycle from blocked primitive coverage\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-dfg-proxy.json"),
                str(primitive_blocked),
                str(dfg_from_blocked_proxy),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DFG cycle derived from blocked primitive coverage unexpectedly passed audit")

        primitive_pass = out_dir / "pass-dataflow-primitive-coverage.csv"
        primitive_pass.write_text(
            "workload,primitive,op_count,dfg_sim_status,diagnostic\n"
            "vecadd,stream,1,pass,primitive covered by simulator\n"
        )
        dfg_from_primitive_pass = out_dir / "dfg-from-pass-primitive-sim-cycle-summary.csv"
        dfg_from_primitive_pass.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,,blocked,synthetic DFG cycle from primitive coverage\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-dfg-pass-primitive.json"),
                str(primitive_pass),
                str(dfg_from_primitive_pass),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DFG cycle backed only by primitive coverage unexpectedly passed audit")

        valid_dfg_report = out_dir / "valid-dfg-sim-report.json"
        write_dfg_report(
            valid_dfg_report,
            "vecadd",
            "g_vecadd",
            10,
            final_outputs=["none", "f32:1"],
        )
        dfg_from_report = out_dir / "dfg-from-report-sim-cycle-summary.csv"
        dfg_from_report.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,,blocked,DFG-sim report available\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-dfg-report.json"),
                str(valid_dfg_report),
                str(dfg_from_report),
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                f"DFG cycle backed by DFG report unexpectedly failed audit\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        noncanonical_dfg_report = out_dir / "vecadd.dfg.report.json"
        write_dfg_report(noncanonical_dfg_report, "vecadd", "g_vecadd", 10)
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-noncanonical-dfg-report.json"),
                str(noncanonical_dfg_report),
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                "DFG report with noncanonical filename unexpectedly failed audit"
            )

        cgra_without_mapping = out_dir / "cgra-without-mapping-sim-cycle-summary.csv"
        cgra_without_mapping.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,12,pass,synthetic CGRA cycle without mapping evidence\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-cgra-no-mapping.json"),
                str(cgra_without_mapping),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA cycle without mapping evidence unexpectedly passed audit")

        valid_mapping = out_dir / "valid-pnr-mapping-summary.csv"
        valid_mapping.write_text(
            "workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic\n"
            "vecadd,fabric0,map0,1,1,0,0,pass,verified mapping\n"
        )
        valid_mapping_artifact = out_dir / "valid-pnr-mapping.json"
        write_mapping_artifact(valid_mapping_artifact, "vecadd", "g_vecadd", "map0")
        cgra_without_report = out_dir / "cgra-without-report-sim-cycle-summary.csv"
        cgra_without_report.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,12,pass,synthetic CGRA cycle without CGRA report\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-cgra-no-report.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(cgra_without_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA cycle without CGRA report unexpectedly passed audit")

        valid_cgra_report = out_dir / "valid-cgra-sim-report.json"
        write_cgra_report(valid_cgra_report, "vecadd", "map0", 10, 12)
        invalid_delta_cgra_report = out_dir / "invalid-delta-cgra-sim-report.json"
        invalid_delta_cgra_report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "cgra_sim_report",
                    "workload": "vecadd",
                    "hardware": "fabric0",
                    "mapping_id": "map0",
                    "status": "pass",
                    "fidelity_level": "mapping_constraint_estimate",
                    "metric_definition": "mapping_constraint_estimate",
                    "operation_semantics_source": "loom.sim.operation_semantics.v1",
                    "operation_cost_model_source": "loom.sim.operation_cost.v1",
                    "difference_classification": "expected_hardware_constraint",
                    "hardware_bound_classification": "within_modeled_bounds",
                    "dfg_cycles": 10,
                    "modeled_lower_bound_cycles": 12,
                    "performance_delta_cycles": 2,
                    "route_latency_cycles": 1,
                    "memory_latency_cycles": 0,
                    "temporal_penalty_cycles": 0,
                    "hardware_aware_cycles": 12,
                    "cycle_breakdown": [
                        {
                            "category": "route_latency",
                            "cycles": 1,
                            "evidence": "mapping.route_segments",
                        }
                    ],
                    "unmodeled_constraints": ["cache_behavior"],
                    "first_principles_checks": [
                        {
                            "name": "delta_explained_by_modeled_constraints",
                            "status": "pass",
                            "evidence": "synthetic invalid report",
                        }
                    ],
                    "diagnostics": ["synthetic invalid delta report"],
                }
            )
        )
        wrong_mapping_cgra_report = out_dir / "wrong-mapping-cgra-sim-report.json"
        write_cgra_report(wrong_mapping_cgra_report, "vecadd", "map1", 10, 12)
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-wrong-cgra-mapping.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(wrong_mapping_cgra_report),
                str(cgra_without_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA report with unrelated mapping_id unexpectedly passed audit")

        ambiguous_hardware = out_dir / "ambiguous-adg-hardware-summary.csv"
        ambiguous_hardware.write_text(
            "hardware,topology_class,node_count,link_count,verify_status,diagnostic\n"
            "test/a.mlir::fabric0,fabric_module_template,1,0,pass,verified\n"
            "test/b.mlir::fabric0,fabric_module_template,1,0,pass,verified\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-ambiguous-hardware.json"),
                str(valid_primitive),
                str(ambiguous_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("ambiguous hardware suffix unexpectedly passed audit")

        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-invalid-cgra-delta.json"),
                str(invalid_delta_cgra_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA report with unexplained delta unexpectedly passed audit")

        truncating_cgra_summary = out_dir / "truncating-cgra-sim-cycle-summary.csv"
        truncating_cgra_summary.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,12.9,pass,synthetic truncating CGRA cycle\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-truncating-cgra.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(truncating_cgra_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("fractional CGRA summary cycle unexpectedly matched integer CGRA report")

        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-cgra-summary-only.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA cycle backed only by mapping summary unexpectedly passed audit")

        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-cgra-report.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                f"CGRA cycle backed by CGRA report unexpectedly failed audit\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        audit_data = json.loads((out_dir / "artifact-audit-summary-cgra-report.json").read_text())
        checks = audit_data.get("cross_artifact_checks")
        if not isinstance(checks, list) or not checks:
            raise AssertionError(f"expected cross-artifact pass checks, got {audit_data}")
        matching_checks = [
            check
            for check in checks
            if check.get("rule") == "sim_cycle_report_mapping_evidence"
            and check.get("workload") == "vecadd"
        ]
        if not matching_checks:
            raise AssertionError(f"expected sim cycle evidence check, got {checks}")
        check = matching_checks[0]
        if check.get("dfg_sim_cycles") != 10 or check.get("cgra_sim_cycles") != 12:
            raise AssertionError(f"sim cycle evidence check missed cycle values: {check}")
        if check.get("dynamic_work_items") != 1:
            raise AssertionError(f"sim cycle evidence check missed dynamic work items: {check}")
        if check.get("mapping_ids") != ["map0"]:
            raise AssertionError(f"sim cycle evidence check missed mapping id: {check}")

        valid_rtl_fpa = out_dir / "valid-rtl-fpa-summary.csv"
        valid_rtl_fpa.write_text(
            "hardware,workload,rtl_lint_status,rtl_sim_status,synth_status,frequency_mhz,area_um2,"
            "dynamic_power_mw,leakage_power_mw,fidelity_level,frequency_source,area_source,power_source,"
            "activity_source,status,diagnostic\n"
            "fabric0,vecadd,skipped,skipped,skipped,100,200,3,1,analytic,analytic_fpa_model,"
            "analytic_fpa_model,analytic_fpa_model,default_toggle,pass,analytic FPA evidence\n"
        )
        dse_provenance_header = (
            "candidate,workload,hardware,mapping_id,objective,cgra_sim_cycles,frequency_mhz,"
            "area_um2,dynamic_power_mw,leakage_power_mw,energy_nj,selection_status,candidate_kind,"
            "input_artifacts,input_artifact_fingerprints,output_artifacts,objective_record,metric_records,policy_id,"
            "ordering_rule,diagnostic\n"
        )
        valid_dse_inputs = [
            valid_mapping,
            valid_mapping_artifact,
            cgra_without_report,
            valid_cgra_report,
            valid_rtl_fpa,
        ]
        valid_dse_input_artifacts = ";".join(str(path) for path in valid_dse_inputs)
        valid_dse_input_fingerprints = semicolon_fingerprints(valid_dse_inputs)
        missing_dse_input_fingerprints = ";".join(
            f"{path}={'0' * 64}" for path in valid_dse_inputs[:-1]
        )
        mismatched_dse_input_fingerprints = valid_dse_input_fingerprints.replace(
            artifact_test_common.fingerprint(valid_mapping),
            "0" * 64,
            1,
        )
        valid_dse_metric_records = (
            "cgra_sim_cycles=12;frequency_mhz=100;area_um2=200;"
            "dynamic_power_mw=3;leakage_power_mw=1;energy_nj=0.480"
        )
        valid_dse = out_dir / "valid-dse-candidate-summary.csv"
        valid_dse_provenance = (
            "combined_full_stack_candidate,"
            f"{valid_dse_input_artifacts},{valid_dse_input_fingerprints},{valid_dse},"
            "objective::minimize_runtime,"
            f"{valid_dse_metric_records},"
            "deterministic_minimize_runtime_v1,"
            "runtime_score_then_candidate_id,"
        )
        valid_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + valid_dse_provenance
            + "cycle-frequency-power-area energy estimate\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-valid-dse.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(valid_dse),
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                f"DSE selected row with matching mapping/sim/FPA unexpectedly failed audit\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        missing_fingerprint_dse = out_dir / "missing-fingerprint-dse-candidate-summary.csv"
        missing_fingerprint_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(missing_fingerprint_dse),
        ).replace(
            valid_dse_input_fingerprints,
            "",
        )
        missing_fingerprint_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + missing_fingerprint_dse_provenance
            + "input artifact provenance omits fingerprints\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-missing-dse-input-fingerprints.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(missing_fingerprint_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with missing input_artifact_fingerprints unexpectedly passed audit")

        missing_fingerprint_entry_dse = out_dir / "missing-fingerprint-entry-dse-candidate-summary.csv"
        missing_fingerprint_entry_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(missing_fingerprint_entry_dse),
        ).replace(
            valid_dse_input_fingerprints,
            missing_dse_input_fingerprints,
        )
        missing_fingerprint_entry_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + missing_fingerprint_entry_dse_provenance
            + "input artifact provenance omits one fingerprint entry\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-missing-dse-input-fingerprint-entry.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(missing_fingerprint_entry_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with incomplete input_artifact_fingerprints unexpectedly passed audit")

        mismatched_fingerprint_dse = out_dir / "mismatched-fingerprint-dse-candidate-summary.csv"
        mismatched_fingerprint_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(mismatched_fingerprint_dse),
        ).replace(
            valid_dse_input_fingerprints,
            mismatched_dse_input_fingerprints,
        )
        mismatched_fingerprint_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + mismatched_fingerprint_dse_provenance
            + "input artifact provenance carries stale fingerprint\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-mismatched-dse-input-fingerprint.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(mismatched_fingerprint_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with stale input_artifact_fingerprints unexpectedly passed audit")

        duplicate_candidate_dse = out_dir / "duplicate-candidate-dse-candidate-summary.csv"
        duplicate_candidate_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(duplicate_candidate_dse),
        )
        duplicate_candidate_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + duplicate_candidate_dse_provenance
            + "cycle-frequency-power-area energy estimate\n"
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,rejected,"
            + duplicate_candidate_dse_provenance
            + "duplicate candidate identity with different selection status\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-duplicate-dse-candidate.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(duplicate_candidate_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("duplicate DSE candidate identity unexpectedly passed audit")

        stale_candidate_dse = out_dir / "stale-candidate-id-dse-candidate-summary.csv"
        stale_candidate_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(stale_candidate_dse),
        )
        stale_candidate_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + stale_candidate_dse_provenance
            + "candidate identity omits immutable mapping id\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-stale-dse-candidate-id.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(stale_candidate_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with stale candidate id unexpectedly passed audit")

        mismatched_objective_dse = out_dir / "mismatched-objective-dse-candidate-summary.csv"
        mismatched_objective_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(mismatched_objective_dse),
        ).replace(
            "objective::minimize_runtime",
            "objective::minimize_energy",
        )
        mismatched_objective_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + mismatched_objective_dse_provenance
            + "objective record contradicts row objective\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-mismatched-dse-objective.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(mismatched_objective_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with mismatched objective record unexpectedly passed audit")

        mismatched_ordering_dse = out_dir / "mismatched-ordering-dse-candidate-summary.csv"
        mismatched_ordering_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(mismatched_ordering_dse),
        ).replace(
            "runtime_score_then_candidate_id",
            "energy_score_then_candidate_id",
        )
        mismatched_ordering_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + mismatched_ordering_dse_provenance
            + "ordering rule contradicts row objective\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-mismatched-dse-ordering.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(mismatched_ordering_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with mismatched ordering rule unexpectedly passed audit")

        mismatched_metric_dse = out_dir / "mismatched-metric-dse-candidate-summary.csv"
        mismatched_metric_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(mismatched_metric_dse),
        ).replace(
            "cgra_sim_cycles=12",
            "cgra_sim_cycles=999",
        )
        mismatched_metric_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + mismatched_metric_dse_provenance
            + "metric records contradict row values\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-mismatched-dse-metric.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(mismatched_metric_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with mismatched metric_records unexpectedly passed audit")

        bogus_input_dse = out_dir / "bogus-input-dse-candidate-summary.csv"
        bogus_input_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(bogus_input_dse),
        ).replace(
            valid_dse_input_artifacts,
            "missing-pnr-mapping-summary.csv;missing-pnr-mapping.json;missing-sim-cycle-summary.csv",
        )
        bogus_input_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + bogus_input_dse_provenance
            + "input artifact provenance points at missing files\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-bogus-dse-inputs.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(bogus_input_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with missing input_artifacts unexpectedly passed audit")

        bogus_output_dse = out_dir / "bogus-output-dse-candidate-summary.csv"
        bogus_output_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            "missing-dse-output.csv",
        )
        bogus_output_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + bogus_output_dse_provenance
            + "output artifact provenance points at a missing file\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-bogus-dse-output.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(bogus_output_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with missing output_artifacts unexpectedly passed audit")

        wrong_output_dse = out_dir / "wrong-output-dse-candidate-summary.csv"
        wrong_output_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(valid_rtl_fpa),
        )
        wrong_output_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + wrong_output_dse_provenance
            + "output artifact provenance points at an unrelated existing file\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-wrong-dse-output.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(wrong_output_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with unrelated output_artifacts unexpectedly passed audit")

        unrelated_input_dse = out_dir / "unrelated-input-dse-candidate-summary.csv"
        unrelated_input_dse_provenance = valid_dse_provenance.replace(
            str(valid_dse),
            str(unrelated_input_dse),
        ).replace(
            valid_dse_input_artifacts,
            f"{valid_primitive};{valid_hardware};{valid_rtl_fpa}",
        )
        unrelated_input_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0::map0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + unrelated_input_dse_provenance
            + "input artifact provenance points at unrelated existing files\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-unrelated-dse-inputs.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(unrelated_input_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE row with unrelated input_artifacts unexpectedly passed audit")

        no_provenance_dse = out_dir / "no-provenance-dse-candidate-summary.csv"
        no_provenance_dse.write_text(
            "candidate,workload,hardware,mapping_id,objective,cgra_sim_cycles,frequency_mhz,area_um2,dynamic_power_mw,leakage_power_mw,energy_nj,selection_status,diagnostic\n"
            "candidate::vecadd::fabric0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,selected,metrics without artifact provenance\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-no-dse-provenance.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(no_provenance_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE selected row without provenance unexpectedly passed audit")

        no_provenance_rejected_dse = out_dir / "no-provenance-rejected-dse-candidate-summary.csv"
        no_provenance_rejected_dse.write_text(
            "candidate,workload,hardware,mapping_id,objective,cgra_sim_cycles,frequency_mhz,area_um2,dynamic_power_mw,leakage_power_mw,energy_nj,selection_status,diagnostic\n"
            "candidate::vecadd::fabric0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,0.480,rejected,metrics without rejected-candidate provenance\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-no-rejected-dse-provenance.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(no_provenance_rejected_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE rejected row without provenance unexpectedly passed audit")

        wrong_mapping_dse_mapping = out_dir / "wrong-mapping-dse-pnr-mapping-summary.csv"
        wrong_mapping_dse_mapping.write_text(
            "workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic\n"
            "vecadd,fabric0,map1,1,1,0,0,pass,stale summary row for a different mapping\n"
        )
        wrong_mapping_dse = out_dir / "wrong-mapping-dse-candidate-summary.csv"
        wrong_mapping_dse_provenance = valid_dse_provenance.replace(str(valid_dse), str(wrong_mapping_dse))
        wrong_mapping_dse.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0,vecadd,fabric0,map1,minimize_runtime,12,100,200,3,1,0.480,selected,"
            + wrong_mapping_dse_provenance
            + "stale mapping id with borrowed simulator cycles\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-wrong-dse-mapping.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(wrong_mapping_dse_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(wrong_mapping_dse),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE selected row with unrelated mapping artifact unexpectedly passed audit")

        invalid_dse_energy = out_dir / "invalid-energy-dse-candidate-summary.csv"
        invalid_dse_energy_provenance = valid_dse_provenance.replace(str(valid_dse), str(invalid_dse_energy))
        invalid_dse_energy.write_text(
            dse_provenance_header
            + "candidate::vecadd::fabric0,vecadd,fabric0,map0,minimize_runtime,12,100,200,3,1,99.000,selected,"
            + invalid_dse_energy_provenance
            + "wrong synthetic energy\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-invalid-dse-energy.json"),
                str(valid_primitive),
                str(valid_hardware),
                str(valid_mapping),
                str(valid_mapping_artifact),
                str(valid_dfg_report),
                str(valid_cgra_report),
                str(cgra_without_report),
                str(valid_rtl_fpa),
                str(invalid_dse_energy),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE selected row with incorrect energy unexpectedly passed audit")

        invalid_mapping = out_dir / "invalid-pnr-mapping-summary.csv"
        invalid_mapping.write_text(
            "workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic\n"
            "vecadd,fabric0,map0,1,0,1,0,pass,\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-invalid-mapping.json"),
                str(invalid_mapping),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("PnR pass row with unrouted edges unexpectedly passed audit")

        invalid_mapping_artifact = out_dir / "invalid-pnr-mapping.json"
        invalid_mapping_artifact.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pnr_mapping",
                    "workload": "vecadd",
                    "hardware": "fabric0",
                    "graph": "g_vecadd",
                    "mapping_id": "map0",
                    "status": "pass",
                    "placed_records": 0,
                    "routed_edges": 0,
                    "unrouted_edges": 0,
                    "unplaced_records": 0,
                    "config_records": 1,
                    "placements": [],
                    "routes": [],
                    "config_bitstream": [],
                }
            )
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-invalid-pnr-json.json"),
                str(invalid_mapping_artifact),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("PnR mapping JSON with mismatched config_records unexpectedly passed audit")

        weak_route_mapping_artifact = out_dir / "weak-route-pnr-mapping.json"
        weak_route_mapping_artifact.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pnr_mapping",
                    "workload": "vecadd",
                    "hardware": "fabric0",
                    "graph": "g_vecadd",
                    "mapping_id": "map0",
                    "status": "pass",
                    "placed_records": 0,
                    "routed_edges": 1,
                    "unrouted_edges": 0,
                    "unplaced_records": 0,
                    "config_records": 2,
                    "placements": [],
                    "routes": [{"from": "arith.addi#0", "to": "arith.muli#0", "status": "routed"}],
                    "config_bitstream": [
                        {
                            "target": "map0::route#0",
                            "register": "from_software_id",
                            "value": "arith.addi#0",
                            "source": "route:arith.addi#0->arith.muli#0",
                        },
                        {
                            "target": "map0::route#0",
                            "register": "to_software_id",
                            "value": "arith.muli#0",
                            "source": "route:arith.addi#0->arith.muli#0",
                        },
                    ],
                }
            )
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-weak-route-pnr-json.json"),
                str(weak_route_mapping_artifact),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("PnR mapping JSON with routes lacking segments unexpectedly passed audit")

        invalid_hardware = out_dir / "invalid-adg-hardware-summary.csv"
        invalid_hardware.write_text(
            "hardware,topology_class,node_count,link_count,verify_status,diagnostic\n"
            "fabric0,arbitrary_graph,0,1,pass,\n"
        )
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "artifact-audit-summary-invalid-hardware.json"),
                str(invalid_hardware),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("ADG hardware pass row with zero nodes unexpectedly passed audit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
