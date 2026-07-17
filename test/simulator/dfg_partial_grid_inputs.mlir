// RUN: loom-dfg-sim %s --graph partial_grid --arg 0=0 --arg 0=1 --arg 0=2 --arg 0=3 --arg 1=7 --arg 1=7 --arg 1=7 --arg 1=7 --arg 2=none --arg 2=none --arg 2=none --arg 2=none --arg 3=false --arg 3=false --arg 3=false --arg 3=true --memref 4=0,0,0,0 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "partial_grid"
// CHECK-DAG: "graph": "partial_grid"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dynamic_work_items": 4
// CHECK-DAG: "dataflow.store": 4
// CHECK-DAG: "arg4": [
// CHECK-DAG: "i32:7"
// CHECK-DAG: "i32:7"
// CHECK-DAG: "i32:7"
// CHECK-DAG: "i32:7"

module {
  dataflow.graph private @partial_grid(
      %ctrl: none, %idx: index, %scalar: i32, %unit: none,
      %last: i1, %mem: memref<?xi32>) -> ()
      attributes {input_segments = array<i32: 0, 4, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %stored = dataflow.store %mem[%idx] %scalar %unit : memref<?xi32>
    %complete:2 = dataflow.demux %last, %stored
        : (i1, none) -> (none, none)
    dataflow.graph.return %complete#1 : none
  }
}
