// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/pack_bits LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/pack_bits/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/pack_bits/main_func.dfg.mlir --graph g_t_pack_bits_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload pack_bits --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: pack_bits,shared_reduction_adg,pack_bits__g_t_pack_bits_kernel_red_0_0__shared_reduction_adg,{{[1-9][0-9]*}},{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "pack_bits"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "mapping_id": "pack_bits__g_t_pack_bits_kernel_red_0_0__shared_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "edge_ref": "arith.ori#0.result0->dataflow.store#0.operand2"
// JSON-DAG: "edge_ref": "arith.select#0.result0->arith.ori#0.operand0"
// JSON-DAG: "edge_ref": "arith.shli#1.result0->arith.select#0.operand2"
// JSON-DAG: "edge_ref": "llvm.trunc#0.result0->arith.addi#0.operand0"
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"
