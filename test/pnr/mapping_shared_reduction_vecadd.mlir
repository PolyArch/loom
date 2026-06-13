// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/vecadd LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecadd/main_func.dfg.mlir --graph g_t_vecadd_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecadd --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecadd/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecadd --output %t.dir/reduction.mapping.csv --artifact %t.dir/reduction.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=REDUCTION < %t.dir/reduction.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: vecadd,shared_reduction_adg,vecadd__g_t_vecadd_0_0__shared_reduction_adg,5,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "vecadd"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "mapping_id": "vecadd__g_t_vecadd_0_0__shared_reduction_adg"
// JSON-DAG: "placed_records": 5
// JSON-DAG: "routed_edges": {{[1-9][0-9]*}}
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "status": "pass"
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.addf#0.operand1"
// JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand2"
// JSON-DAG: "edge_ref": "arith.addf#0.result0->dataflow.store#0.operand2"

// REDUCTION-DAG: "workload": "vecadd"
// REDUCTION-DAG: "hardware": "shared_reduction_adg"
// REDUCTION-DAG: "mapping_id": "vecadd__g_t_main_red_0_0__shared_reduction_adg"
// REDUCTION-DAG: "routed_edges": {{[1-9][0-9]*}}
// REDUCTION-DAG: "unrouted_edges": 0
// REDUCTION-DAG: "status": "pass"
// REDUCTION-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand1"
// REDUCTION-DAG: "edge_ref": "dataflow.carry#0.result0->arith.addf#0.operand0"
