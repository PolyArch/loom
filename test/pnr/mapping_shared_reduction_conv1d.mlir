// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/conv1d LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/conv1d/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/conv1d/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload conv1d --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: conv1d,shared_reduction_adg,conv1d__g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0__shared_reduction_adg,6,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "conv1d"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "mapping_id": "conv1d__g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0__shared_reduction_adg"
// JSON-DAG: "placed_records": 6
// JSON-DAG: "routed_edges": {{[1-9][0-9]*}}
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "status": "pass"
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->llvm.intr.fmuladd#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->llvm.intr.fmuladd#0.operand1"
// JSON-DAG: "edge_ref": "dataflow.carry#0.result0->llvm.intr.fmuladd#0.operand2"
// JSON-DAG: "edge_ref": "llvm.intr.fmuladd#0.result0->dataflow.carry#0.operand2"
