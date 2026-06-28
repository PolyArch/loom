// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/bitonic_stage
// RUN: %loom-cc -emit-llvm -O1 -S %S/../app/bitonic_stage/main_func.c -o %t.dir/bitonic_stage/main_func.ll
// RUN: %loom-raise %t.dir/bitonic_stage/main_func.ll -o %t.dir/bitonic_stage/main_func.scf.mlir
// RUN: %loom-lower %t.dir/bitonic_stage/main_func.scf.mlir -o %t.dir/bitonic_stage/main_func.dfg.mlir
// RUN: loom-pnr-map --dfg-mlir %t.dir/bitonic_stage/main_func.dfg.mlir --graph g_bitonic_stage_0 --hardware-mlir %S/shared_memory_reduction_adg.mlir --hardware shared_memory_reduction_adg --workload bitonic_stage --output %t.dir/bitonic_stage.mapping.csv --artifact %t.dir/bitonic_stage.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/bitonic_stage.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/bitonic_stage.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: bitonic_stage,shared_memory_reduction_adg,bitonic_stage__g_bitonic_stage_0__shared_memory_reduction_adg,23,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "bitonic_stage"
// JSON-DAG: "hardware": "shared_memory_reduction_adg"
// JSON-DAG: "mapping_id": "bitonic_stage__g_bitonic_stage_0__shared_memory_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 23
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "edge_ref": "arith.cmpi#0.result0->arith.select#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.constant#1.result0->arith.index_cast#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.constant#1.result0->arith.index_cast#1.operand0"
// JSON-DAG: "edge_ref": "arith.index_cast#0.result0->arith.shli#1.operand1"
// JSON-DAG: "edge_ref": "arith.index_cast#1.result0->arith.shli#2.operand1"
// JSON-DAG: "edge_ref": "arith.shrui#0.result0->dataflow.load#0.operand1"
// JSON-DAG: "edge_ref": "arith.shrui#1.result0->dataflow.store#0.operand1"
// JSON-DAG: "edge_ref": "arith.select#1.result0->dataflow.store#0.operand2"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"
