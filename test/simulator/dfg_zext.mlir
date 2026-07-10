// RUN: loom-dfg-sim %s --graph zext --arg 0=none --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "zext"
// CHECK-DAG: "graph": "zext"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "operation_cost_score": 4
// CHECK-DAG: "wavefront_steps": 2
// CHECK-DAG: "event_count": 2
// CHECK-DAG: "i64:42"

module {
  dataflow.graph.func private @zext(%ctrl: none) -> (none, i64) {
    %narrow = dataflow.constant %ctrl {const_value = 42 : i32} : i32
    %wide = llvm.zext %narrow : i32 to i64
    dataflow.graph.return %ctrl, %wide : none, i64
  }
}
