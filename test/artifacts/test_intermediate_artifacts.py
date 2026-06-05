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
                    "adg-hardware-summary.csv",
                },
            )
            produced.append(output)

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
        valid_dfg_report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "dfg_sim_report",
                    "workload": "vecadd",
                    "graph": "g_vecadd",
                    "status": "pass",
                    "metric_definition": "optimistic_operation_latency_sum",
                    "operation_semantics_source": "loom.sim.operation_semantics.v1",
                    "operation_cost_model_source": "loom.sim.operation_cost.v1",
                    "optimistic_cycles": 10,
                    "wavefront_steps": 4,
                    "event_count": 10,
                    "final_outputs": ["none", "f32:1"],
                    "diagnostics": [],
                }
            )
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
        valid_mapping_artifact.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pnr_mapping",
                    "workload": "vecadd",
                    "hardware": "fabric0",
                    "graph": "g_vecadd",
                    "mapping_id": "map0",
                    "status": "pass",
                    "placed_records": 1,
                    "routed_edges": 1,
                    "unrouted_edges": 0,
                    "unplaced_records": 0,
                    "config_records": 0,
                    "placements": [
                        {
                            "software": "arith.addi#0",
                            "operation": "arith.addi",
                            "resource_kind": "fabric.op",
                            "hardware": "fabric0::fabric.op#0",
                            "schedule": "spatial",
                        }
                    ],
                    "routes": [
                        {
                            "record_id": "route#0",
                            "edge_ref": "arith.addi#0.result0->arith.muli#0.operand0",
                            "producer_binding": "placement:arith.addi#0",
                            "consumer_binding": "placement:arith.muli#0",
                            "payload_kind": "data",
                            "from": "arith.addi#0",
                            "to": "arith.muli#0",
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
        valid_cgra_report.write_text(
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
                    "memory_latency_cycles": 1,
                    "temporal_penalty_cycles": 0,
                    "hardware_aware_cycles": 12,
                    "cycle_breakdown": [
                        {
                            "category": "route_latency",
                            "cycles": 1,
                            "evidence": "mapping.route_segments",
                        },
                        {
                            "category": "memory_latency",
                            "cycles": 1,
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
        wrong_mapping_cgra_report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "cgra_sim_report",
                    "workload": "vecadd",
                    "hardware": "fabric0",
                    "mapping_id": "map1",
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
                    "memory_latency_cycles": 1,
                    "temporal_penalty_cycles": 0,
                    "hardware_aware_cycles": 12,
                    "cycle_breakdown": [
                        {
                            "category": "route_latency",
                            "cycles": 1,
                            "evidence": "mapping.route_segments",
                        },
                        {
                            "category": "memory_latency",
                            "cycles": 1,
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
