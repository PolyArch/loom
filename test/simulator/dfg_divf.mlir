// RUN: loom-dfg-sim %s --graph divf --arg 0=none --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "divf"
// CHECK-DAG: "graph": "divf"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "optimistic_cycles": 14
// CHECK-DAG: "event_count": 3
// CHECK-DAG: "arith.divf": 1
// CHECK-DAG: "f32:3"

module {
  dataflow.graph.func private @divf(%ctrl: none) -> (none, f32) {
    %lhs = dataflow.constant %ctrl {const_value = 7.500000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 2.500000e+00 : f32} : f32
    %result = arith.divf %lhs, %rhs : f32
    dataflow.graph.return %ctrl, %result : none, f32
  }
}
