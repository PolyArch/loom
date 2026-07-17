// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph sum_load --arg 0=0 --arg 1=3 --arg 2=1 --arg 3=0.000000e+00 --memref 4=1.000000e+00,2.000000e+00,3.000000e+00,99.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "sum_load"
// CHECK-DAG: "graph": "sum_load"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "weighted_operations_plus_library_work_diversity_and_address.v1"
// CHECK-DAG: "dataflow.load": 3
// CHECK-DAG: "f32:6"

module {
  dataflow.graph private @sum_load(
      %start: none, %lb: i64, %ub: i64, %step: i64, %init: f32,
      %mem: memref<?xf32>) -> (f32)
      attributes {input_segments = array<i32: 4, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result = scf.for %i = %lb to %ub step %step
        iter_args(%acc = %init) -> (f32) : i64 {
      %idx = arith.index_cast %i : i64 to index
      %data = memref.load %mem[%idx] : memref<?xf32>
      %next = arith.addf %acc, %data : f32
      scf.yield %next : f32
    }
    dataflow.graph.return %start, %result : none, f32
  }
}
