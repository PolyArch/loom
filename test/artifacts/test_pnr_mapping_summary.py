#!/usr/bin/env python3
"""Regression test for PnR mapping summary candidate rows."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import artifact_test_common


HEADER = [
    "workload",
    "hardware",
    "mapping_id",
    "placed_records",
    "routed_edges",
    "unrouted_edges",
    "unplaced_records",
    "status",
]


def load_aggregate_module(repo: Path):
    module_path = repo / "test/e2e/aggregate_workload_graph_artifacts.py"
    spec = importlib.util.spec_from_file_location("aggregate_workload_graph_artifacts", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load aggregate helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_audit_failure(repo: Path, artifact: Path, audit_output: Path, expected_fragment: str) -> None:
    result = artifact_test_common.run_command(
        repo,
        [
            "python3",
            "test/e2e/audit_intermediate_artifacts.py",
            "--output",
            str(audit_output),
            str(artifact),
        ],
    )
    if result.returncode == 0:
        raise AssertionError(f"{artifact.name} unexpectedly passed audit")
    audit = json.loads(audit_output.read_text())
    diagnostics = json.dumps(audit.get("diagnostics", []), sort_keys=True)
    if expected_fragment not in diagnostics:
        raise AssertionError(f"audit missed {expected_fragment!r}: {audit}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-pnr-mapping-") as tmp:
        out_dir = Path(tmp)
        mapping = out_dir / "pnr-mapping-summary.csv"
        primitive, hardware = artifact_test_common.prepare_candidate_inputs(repo, out_dir)
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/pnr/run_mapping_summary.sh",
            mapping,
            HEADER,
            "--primitive-coverage",
            str(primitive),
            "--hardware-summary",
            str(hardware),
            label="PnR mapping summary",
        )

        matches = [
            row
            for row in rows
            if row["workload"] == "vecadd" and row["hardware"].endswith("::pe_two_pes")
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one vecadd to pe_two_pes row, got {rows}")
        row = matches[0]
        for column in ("mapping_id", "placed_records", "routed_edges", "unrouted_edges", "unplaced_records"):
            if row[column] != "":
                raise AssertionError(f"blocked row must not fake {column}: {row}")
        if row["status"] != "blocked":
            raise AssertionError(f"mapping row should be blocked: {row}")
        if "explicit mapper inputs are required" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")

        dfg_dir = out_dir / "vecsum-dfg"
        result = artifact_test_common.run_command(
            repo,
            [
                "env",
                f"BUILD_DIR={dfg_dir}",
                "bash",
                "test/app/vecsum/dfg_check.sh",
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                "vecsum DFG check with explicit build dir failed\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        mapped = out_dir / "pnr-mapping-summary-pass.csv"
        artifact = out_dir / "pnr-mapping.json"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/pnr/run_mapping_summary.sh",
            mapped,
            HEADER,
            "--dfg-mlir",
            str(dfg_dir / "main_func.dfg.mlir"),
            "--graph",
            "g_t_vecsum_red_0_0",
            "--hardware-mlir",
            "test/pnr/shared_reduction_adg.mlir",
            "--hardware",
            "shared_reduction_adg",
            "--workload",
            "vecsum",
            "--artifact",
            str(artifact),
            label="PnR mapping summary explicit mapper",
        )
        if len(rows) != 1:
            raise AssertionError(f"expected one explicit mapping row, got {rows}")
        row = rows[0]
        expected = {
            "workload": "vecsum",
            "hardware": "shared_reduction_adg",
            "mapping_id": "vecsum__g_t_vecsum_red_0_0__shared_reduction_adg",
            "placed_records": "5",
            "routed_edges": "6",
            "unrouted_edges": "0",
            "unplaced_records": "0",
            "status": "pass",
        }
        for key, value in expected.items():
            if row[key] != value:
                raise AssertionError(f"explicit mapping {key}={row[key]!r}, expected {value!r}")
        if "unrouted software edges lack Fabric ADG connectivity" in row.get("diagnostic", ""):
            raise AssertionError(f"explicit mapping should not report unrouted edges after routing passes: {row}")
        if not artifact.is_file():
            raise AssertionError("explicit mapping did not emit JSON artifact")
        data = json.loads(artifact.read_text())
        expected_json = {
            "kind": "pnr_mapping",
            "workload": "vecsum",
            "graph": "g_t_vecsum_red_0_0",
            "hardware": "shared_reduction_adg",
            "mapping_id": "vecsum__g_t_vecsum_red_0_0__shared_reduction_adg",
            "placed_records": 5,
            "routed_edges": 6,
            "unrouted_edges": 0,
            "unplaced_records": 0,
            "config_records": 97,
            "status": "pass",
        }
        for key, value in expected_json.items():
            if data.get(key) != value:
                raise AssertionError(f"explicit mapping artifact {key}={data.get(key)!r}, expected {value!r}")
        if len(data.get("config_bitstream", [])) != 97:
            raise AssertionError(f"explicit mapping config bitstream size changed: {data}")
        if data.get("unrouted_edge_details") != []:
            raise AssertionError(f"passing mapping should have no unrouted edge details: {data}")
        endpoint_pairs = {
            (segment.get("source_endpoint"), segment.get("sink_endpoint"))
            for route in data.get("routes", [])
            for segment in route.get("segments", [])
        }
        required_endpoints = {
            (
                "shared_reduction_adg::fabric.op#2.result0",
                "shared_reduction_adg::fabric.fu#0.result1",
            ),
            (
                "shared_reduction_adg::fabric.pe#0.result1",
                "shared_reduction_adg::fabric.switch#7.operand0",
            ),
            (
                "shared_reduction_adg::fabric.switch#7.operand0",
                "shared_reduction_adg::fabric.switch#7.result0",
            ),
            (
                "shared_reduction_adg::fabric.switch#7.result0",
                "shared_reduction_adg::fabric.op#1.operand2",
            ),
            (
                "shared_reduction_adg::mem.load#0.result0",
                "shared_reduction_adg::fabric.switch#1.operand0",
            ),
            (
                "shared_reduction_adg::fabric.switch#1.operand0",
                "shared_reduction_adg::fabric.switch#1.result0",
            ),
            (
                "shared_reduction_adg::fabric.switch#1.result0",
                "shared_reduction_adg::fabric.op#2.operand0",
            ),
            (
                "shared_reduction_adg::fabric.op#1.result0",
                "shared_reduction_adg::fabric.op#2.operand1",
            ),
            (
                "shared_reduction_adg::mem.load#0.result1",
                "shared_reduction_adg::fabric.op#32.operand0",
            ),
            (
                "shared_reduction_adg::fabric.pe#0.result0",
                "shared_reduction_adg::mem.load#0.operand0",
            ),
            (
                "shared_reduction_adg::fabric.op#0.result1",
                "shared_reduction_adg::fabric.op#1.operand0",
            ),
        }
        if not required_endpoints.issubset(endpoint_pairs):
            raise AssertionError(f"explicit mapping route endpoints changed: {endpoint_pairs}")
        for source, sink in endpoint_pairs:
            if (source and source.endswith(".out")) or (sink and sink.endswith(".in")):
                raise AssertionError(f"explicit mapping used legacy string endpoint: {(source, sink)}")

        vecadd_dir = out_dir / "vecadd-dfg"
        result = artifact_test_common.run_command(
            repo,
            [
                "env",
                f"BUILD_DIR={vecadd_dir}",
                "bash",
                "test/app/vecadd/dfg_check.sh",
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                "vecadd DFG check with explicit build dir failed\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        graph_mapping_ids: dict[str, str] = {}
        expected_graph_status = {
            "g_t_vecadd_0_0": "pass",
            "g_t_main_red_0_0": "pass",
        }
        for graph_name in ("g_t_vecadd_0_0", "g_t_main_red_0_0"):
            graph_csv = out_dir / f"{graph_name}.mapping.csv"
            graph_artifact = out_dir / f"{graph_name}.mapping.json"
            graph_rows = artifact_test_common.run_csv_summary(
                repo,
                "test/pnr/run_mapping_summary.sh",
                graph_csv,
                HEADER,
                "--dfg-mlir",
                str(vecadd_dir / "main_func.dfg.mlir"),
                "--graph",
                graph_name,
                "--hardware-mlir",
                "test/pnr/shared_reduction_adg.mlir",
                "--hardware",
                "shared_reduction_adg",
                "--workload",
                "vecadd",
                "--artifact",
                str(graph_artifact),
                label=f"PnR mapping summary for {graph_name}",
            )
            if len(graph_rows) != 1:
                raise AssertionError(f"expected one mapping row for {graph_name}, got {graph_rows}")
            graph_row = graph_rows[0]
            expected_status = expected_graph_status[graph_name]
            if graph_row["status"] != expected_status:
                raise AssertionError(f"mapping row for {graph_name} status changed: {graph_row}")
            if expected_status == "fail":
                if "unrouted software edges lack Fabric ADG connectivity" not in graph_row.get("diagnostic", ""):
                    raise AssertionError(f"mapping row for {graph_name} should diagnose unrouted edges: {graph_row}")
            elif graph_row["unrouted_edges"] != "0":
                raise AssertionError(f"passing mapping row for {graph_name} should have no unrouted edges: {graph_row}")
            mapping_id = graph_row["mapping_id"]
            if graph_name not in mapping_id:
                raise AssertionError(
                    f"mapping id must include graph identity {graph_name!r}: {mapping_id!r}"
                )
            graph_data = json.loads(graph_artifact.read_text())
            expected_json = {
                "kind": "pnr_mapping",
                "workload": "vecadd",
                "graph": graph_name,
                "hardware": "shared_reduction_adg",
                "mapping_id": mapping_id,
            }
            for key, value in expected_json.items():
                if graph_data.get(key) != value:
                    raise AssertionError(
                        f"{graph_name} mapping artifact {key}={graph_data.get(key)!r}, expected {value!r}"
                    )
            unrouted_details = graph_data.get("unrouted_edge_details")
            if expected_status == "fail":
                if not isinstance(unrouted_details, list) or len(unrouted_details) != graph_data.get("unrouted_edges"):
                    raise AssertionError(f"{graph_name} should expose exact unrouted edge details: {graph_data}")
            elif unrouted_details != []:
                raise AssertionError(f"{graph_name} passing mapping should not carry unrouted details: {graph_data}")
            graph_mapping_ids[graph_name] = mapping_id

        if len(set(graph_mapping_ids.values())) != len(graph_mapping_ids):
            raise AssertionError(f"multi-graph workload mapping ids collided: {graph_mapping_ids}")

        failed_unrouted_component = copy.deepcopy(data)
        failed_unrouted_component["status"] = "fail"
        failed_unrouted_component["unrouted_edges"] = 1
        failed_unrouted_component["diagnostics"] = ["synthetic unrouted edge"]
        failed_unrouted_component["unrouted_edge_details"] = [
            {
                "edge_ref": "synthetic.producer.result0->synthetic.consumer.operand0",
                "producer_binding": "placement:synthetic.producer",
                "consumer_binding": "placement:synthetic.consumer",
                "payload_kind": "data",
                "from": "synthetic.producer",
                "to": "synthetic.consumer",
                "status": "unrouted",
                "source_endpoint": "shared_reduction_adg::synthetic.result0",
                "sink_endpoint": "shared_reduction_adg::synthetic.operand0",
                "diagnostic": "synthetic missing connectivity",
            }
        ]

        missing_unrouted_details = out_dir / "missing-unrouted-details-pnr-mapping.json"
        missing_unrouted_details_data = copy.deepcopy(failed_unrouted_component)
        missing_unrouted_details_data.pop("unrouted_edge_details", None)
        missing_unrouted_details.write_text(
            json.dumps(missing_unrouted_details_data, indent=2, sort_keys=True) + "\n"
        )
        expect_audit_failure(
            repo,
            missing_unrouted_details,
            out_dir / "missing-unrouted-details-audit.json",
            "unrouted_edge_details",
        )

        empty_unrouted_details = out_dir / "empty-unrouted-details-pnr-mapping.json"
        empty_unrouted_details_data = copy.deepcopy(failed_unrouted_component)
        empty_unrouted_details_data["unrouted_edge_details"] = []
        empty_unrouted_details.write_text(
            json.dumps(empty_unrouted_details_data, indent=2, sort_keys=True) + "\n"
        )
        expect_audit_failure(
            repo,
            empty_unrouted_details,
            out_dir / "empty-unrouted-details-audit.json",
            "unrouted_edge_details",
        )

        aggregate = load_aggregate_module(repo)
        failed_zero_count_component = copy.deepcopy(data)
        failed_zero_count_component["status"] = "fail"
        failed_zero_count_component["diagnostics"] = ["synthetic component failure"]
        aggregate_mapping = aggregate.aggregate_mapping(
            argparse.Namespace(workload="vecadd", hardware="shared_reduction_adg", graph="workload_graph_set",
                               mapping_id="vecadd__workload_graph_set__shared_reduction_adg"),
            [artifact],
            [failed_zero_count_component],
        )
        if aggregate_mapping.get("status") == "pass":
            raise AssertionError(f"aggregate mapping should not pass failed components: {aggregate_mapping}")

        punctuation_mlir = out_dir / "mapping-punctuation.mlir"
        punctuation_mlir.write_text(
            """module {
  dataflow.graph.func private @"punct.foo"(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @"punct-foo"(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  fabric.module @punctuation_adg(%i32a : !fabric.bits<32>,
                                 %i32b : !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> () {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield
      }
    }
    fabric.yield
  }
}
"""
        )
        punctuation_mapping_ids: dict[str, str] = {}
        for graph_name in ("punct.foo", "punct-foo"):
            graph_csv = out_dir / f"{graph_name}.csv"
            graph_artifact = out_dir / f"{graph_name}.json"
            graph_rows = artifact_test_common.run_csv_summary(
                repo,
                "test/pnr/run_mapping_summary.sh",
                graph_csv,
                HEADER,
                "--dfg-mlir",
                str(punctuation_mlir),
                "--graph",
                graph_name,
                "--hardware-mlir",
                str(punctuation_mlir),
                "--hardware",
                "punctuation_adg",
                "--workload",
                "punctuation",
                "--artifact",
                str(graph_artifact),
                label=f"PnR mapping summary for quoted graph {graph_name}",
            )
            if len(graph_rows) != 1:
                raise AssertionError(f"expected one mapping row for {graph_name}, got {graph_rows}")
            graph_row = graph_rows[0]
            if graph_row["status"] != "pass":
                raise AssertionError(f"quoted graph mapping should pass: {graph_row}")
            graph_data = json.loads(graph_artifact.read_text())
            mapping_id = graph_row["mapping_id"]
            if graph_data.get("graph") != graph_name:
                raise AssertionError(f"quoted graph artifact lost graph identity: {graph_data}")
            if graph_data.get("mapping_id") != mapping_id:
                raise AssertionError(f"quoted graph artifact disagrees with CSV row: {graph_data}")
            punctuation_mapping_ids[graph_name] = mapping_id

        if len(set(punctuation_mapping_ids.values())) != len(punctuation_mapping_ids):
            raise AssertionError(
                f"punctuated graph mapping ids collided: {punctuation_mapping_ids}"
            )
        expected_punctuation_ids = {
            "punct.foo": "punctuation__punct%2Efoo__punctuation_adg",
            "punct-foo": "punctuation__punct%2Dfoo__punctuation_adg",
        }
        if punctuation_mapping_ids != expected_punctuation_ids:
            raise AssertionError(
                f"punctuated graph mapping ids changed: {punctuation_mapping_ids}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
