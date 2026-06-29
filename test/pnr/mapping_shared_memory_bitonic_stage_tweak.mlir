// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/bitonic_stage_tweak
// RUN: %loom-c++ -emit-llvm -O1 -S %S/../app/bitonic_stage-tweak/main_func.cpp -o %t.dir/bitonic_stage_tweak/main_func.ll
// RUN: %loom-raise %t.dir/bitonic_stage_tweak/main_func.ll -o %t.dir/bitonic_stage_tweak/main_func.scf.mlir
// RUN: %loom-lower %t.dir/bitonic_stage_tweak/main_func.scf.mlir -o %t.dir/bitonic_stage_tweak/main_func.dfg.mlir
// RUN: loom-pnr-map --dfg-mlir %t.dir/bitonic_stage_tweak/main_func.dfg.mlir --graph g_bitonic_stage_tweak_kernel_0 --hardware-mlir %S/shared_memory_reduction_adg.mlir --hardware shared_memory_reduction_adg --workload bitonic_stage-tweak --output %t.dir/bitonic_stage_tweak.mapping.csv --artifact %t.dir/bitonic_stage_tweak.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/bitonic_stage_tweak.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/bitonic_stage_tweak.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: bitonic_stage-tweak,shared_memory_reduction_adg,bitonic_stage%2Dtweak__g_bitonic_stage_tweak_kernel_0__shared_memory_reduction_adg,{{[1-9][0-9]*}},{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "bitonic_stage-tweak"
// JSON-DAG: "hardware": "shared_memory_reduction_adg"
// JSON-DAG: "mapping_id": "bitonic_stage%2Dtweak__g_bitonic_stage_tweak_kernel_0__shared_memory_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "operation": "arith.index_cast"
// JSON-DAG: "operation": "dataflow.constant"
// JSON-NOT: "resource_pressure"
