// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/sort_bubble
// RUN: %loom-c++ -emit-llvm -O1 -S %S/../app/sort_bubble/main_func.cpp -o %t.dir/sort_bubble/main_func.ll
// RUN: %loom-raise %t.dir/sort_bubble/main_func.ll -o %t.dir/sort_bubble/main_func.scf.mlir
// RUN: %loom-lower %t.dir/sort_bubble/main_func.scf.mlir -o %t.dir/sort_bubble/main_func.dfg.mlir
// RUN: loom-pnr-map --dfg-mlir %t.dir/sort_bubble/main_func.dfg.mlir --graph g_t_sort_bubble_kernel_red_0_0 --hardware-mlir %S/shared_memory_reduction_adg.mlir --hardware shared_memory_reduction_adg --workload sort_bubble --output %t.dir/sort_bubble.mapping.csv --artifact %t.dir/sort_bubble.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/sort_bubble.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/sort_bubble.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: sort_bubble,shared_memory_reduction_adg,sort_bubble__g_t_sort_bubble_kernel_red_0_0__shared_memory_reduction_adg,{{[1-9][0-9]*}},{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "sort_bubble"
// JSON-DAG: "hardware": "shared_memory_reduction_adg"
// JSON-DAG: "mapping_id": "sort_bubble__g_t_sort_bubble_kernel_red_0_0__shared_memory_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.cmpf#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.cmpf#0.operand1"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->dataflow.store#0.operand2"
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->dataflow.store#1.operand2"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"
