// RUN: loom-dfg-sim %s --graph grid_like --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 1=1 --arg 1=2 --arg 1=3 --arg 1=4 --arg 2=10 --arg 2=20 --arg 2=30 --arg 2=40 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "grid_like"
// CHECK-DAG: "graph": "grid_like"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "operation_cost_score": 5
// CHECK-DAG: "event_count": 4
// CHECK-DAG: "dynamic_work_items": 4
// CHECK-DAG: "i32:44"

module {
  dataflow.graph.func private @grid_like(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }
}
