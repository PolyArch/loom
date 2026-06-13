// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/byte_swap LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/byte_swap/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %S/byte_swap_store_adg.mlir --hardware byte_swap_store_adg --workload byte_swap --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: byte_swap,byte_swap_store_adg,byte_swap__g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0__byte_swap_store_adg,4,4,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "byte_swap"
// JSON-DAG: "hardware": "byte_swap_store_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 4
// JSON-DAG: "routed_edges": 4
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->llvm.intr.bswap#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#0.result1->dataflow.sync#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand1"
// JSON-DAG: "edge_ref": "llvm.intr.bswap#0.result0->dataflow.store#0.operand2"
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"
