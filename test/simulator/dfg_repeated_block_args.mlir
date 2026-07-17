// RUN: loom-dfg-sim %s --graph grid_like --arg 0=1 --arg 0=2 --arg 0=3 --arg 0=4 --arg 1=10 --arg 1=20 --arg 1=30 --arg 1=40 --arg 2=none --arg 2=none --arg 2=none --arg 2=none --arg 3=false --arg 3=false --arg 3=false --arg 3=true --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "grid_like"
// CHECK-DAG: "graph": "grid_like"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "operation_cost_score": 19
// CHECK-DAG: "event_count": 12
// CHECK-DAG: "dynamic_work_items": 4
// CHECK-DAG: "final_stream_outputs":
// CHECK-DAG: "i32:44"

module {
  dataflow.graph.func private @grid_like(
      %ctrl: none, %lhs: i32, %rhs: i32, %unit: none, %last: i1)
      -> (none, i32)
      attributes {input_segments = array<i32: 0, 4, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %sum = arith.addi %lhs, %rhs : i32
    %paired:2 = dataflow.sync %sum, %unit
        : (i32, none) -> (i32, none)
    %complete:2 = dataflow.demux %last, %paired#1
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%paired#0 : i32) memories()
        complete(%complete#1 : none)
  }
}
