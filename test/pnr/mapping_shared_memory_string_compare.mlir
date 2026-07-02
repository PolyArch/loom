// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/string_compare LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/string_compare/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/string_compare/main_func.dfg.mlir --graph g_string_compare_kernel_0 --hardware-mlir %S/shared_memory_reduction_adg.mlir --hardware shared_memory_reduction_adg --workload string_compare --output %t.dir/string_compare.mapping.csv --artifact %t.dir/string_compare.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/string_compare.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/string_compare.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: string_compare,shared_memory_reduction_adg,string_compare__g_string_compare_kernel_0__shared_memory_reduction_adg,22,29,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "workload": "string_compare"
// JSON-DAG: "hardware": "shared_memory_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 22
// JSON-DAG: "routed_edges": 29
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "edge_ref": "dataflow.mux#4.result0->arith.trunci#0.operand0"
// JSON-DAG: "operation": "dataflow.constant"
// JSON-DAG: "operation": "arith.trunci"
// JSON-DAG: "operation": "dataflow.mux"
// JSON-NOT: "unsupported PnR graph operation: ub.poison"
// JSON-NOT: "no Fabric ADG route connects source to sink"
