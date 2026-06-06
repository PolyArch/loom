// RUN: loom-dfg-sim %s --graph partial_grid --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 1=0 --arg 1=1 --arg 1=2 --arg 1=3 --arg 2=7 --memref 3=0,0,0,0 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "partial_grid"
// CHECK-DAG: "graph": "partial_grid"
// CHECK-DAG: "status": "blocked"
// CHECK-DAG: "dynamic_work_items": 4
// CHECK-DAG: "dataflow.graph.return value produced 1 of 4 dynamic work items"

module {
  dataflow.graph.func private @partial_grid(%ctrl: none, %idx: index,
                                            %scalar: i32, %mem: memref<?xi32>)
      -> none {
    %stored = dataflow.store %mem[%idx] %scalar %ctrl : memref<?xi32>
    dataflow.graph.return %stored : none
  }
}
