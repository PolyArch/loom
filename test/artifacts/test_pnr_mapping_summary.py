#!/usr/bin/env python3
"""Regression test for PnR mapping summary candidate rows."""

from __future__ import annotations

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
            "config_records": 73,
            "status": "pass",
        }
        for key, value in expected_json.items():
            if data.get(key) != value:
                raise AssertionError(f"explicit mapping artifact {key}={data.get(key)!r}, expected {value!r}")
        if len(data.get("config_bitstream", [])) != 73:
            raise AssertionError(f"explicit mapping config bitstream size changed: {data}")
        endpoint_pairs = {
            (segment.get("source_endpoint"), segment.get("sink_endpoint"))
            for route in data.get("routes", [])
            for segment in route.get("segments", [])
        }
        required_endpoints = {
            (
                "shared_reduction_adg::fabric.op#2.result0",
                "shared_reduction_adg::fabric.op#1.operand2",
            ),
            (
                "shared_reduction_adg::mem.load#0.result0",
                "shared_reduction_adg::fabric.op#2.operand0",
            ),
            (
                "shared_reduction_adg::fabric.op#1.result0",
                "shared_reduction_adg::fabric.op#2.operand1",
            ),
            (
                "shared_reduction_adg::mem.load#0.result1",
                "shared_reduction_adg::fabric.op#29.operand0",
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
            if graph_row["status"] != "fail":
                raise AssertionError(f"mapping row for {graph_name} should fail without Fabric ADG routes: {graph_row}")
            if "unrouted software edges lack Fabric ADG connectivity" not in graph_row.get("diagnostic", ""):
                raise AssertionError(f"mapping row for {graph_name} should diagnose unrouted edges: {graph_row}")
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
            graph_mapping_ids[graph_name] = mapping_id

        if len(set(graph_mapping_ids.values())) != len(graph_mapping_ids):
            raise AssertionError(f"multi-graph workload mapping ids collided: {graph_mapping_ids}")

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
