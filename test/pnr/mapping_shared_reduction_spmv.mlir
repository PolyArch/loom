// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/spmv LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/spmv/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/spmv/main_func.dfg.mlir --graph g_t_spmv_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload spmv --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: spmv,shared_reduction_adg,spmv__g_t_spmv_kernel_red_0_0__shared_reduction_adg,9,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "spmv"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "mapping_id": "spmv__g_t_spmv_kernel_red_0_0__shared_reduction_adg"
// JSON-DAG: "placed_records": 9
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "status": "pass"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->llvm.zext#0.operand0"
// JSON-DAG: "edge_ref": "llvm.zext#0.result0->dataflow.load#2.operand1"
// JSON-DAG: "edge_ref": "dataflow.load#2.result1->dataflow.sync#0.operand2"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"
