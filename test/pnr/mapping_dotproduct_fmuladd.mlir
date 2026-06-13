// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/dotproduct LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/dotproduct/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/dotproduct/main_func.dfg.mlir --graph g_t_dotproduct_red_0_0 --hardware-mlir %S/dotproduct_fmuladd_adg.mlir --hardware dotproduct_fmuladd_adg --workload dotproduct --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: dotproduct,dotproduct_fmuladd_adg,dotproduct__g_t_dotproduct_red_0_0__dotproduct_fmuladd_adg,6,9,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "dotproduct"
// JSON-DAG: "hardware": "dotproduct_fmuladd_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 6
// JSON-DAG: "routed_edges": 9
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->llvm.intr.fmuladd#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#0.result1->dataflow.sync#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->llvm.intr.fmuladd#0.operand1"
// JSON-DAG: "edge_ref": "dataflow.carry#0.result0->llvm.intr.fmuladd#0.operand2"
// JSON-DAG: "edge_ref": "llvm.intr.fmuladd#0.result0->dataflow.carry#0.operand2"
// JSON-DAG: "edge_ref": "dataflow.load#1.result1->dataflow.sync#0.operand1"
// JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#0.operand1"
// JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#1.operand1"
// JSON-DAG: "edge_ref": "dataflow.stream#0.result1->dataflow.carry#0.operand0"
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"
