// RUN: loom-dfg-sim %s --graph divf --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "divf"
// CHECK-DAG: "graph": "divf"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "operation_cost_score": 18
// CHECK-DAG: "event_count": 4
// CHECK-DAG: "arith.divf": 1
// CHECK-DAG: "f32:3"

module {
  dataflow.graph.func private @divf(%ctrl: none) -> (none, f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %ctrl {const_value = 7.500000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 2.500000e+00 : f32} : f32
    %result = arith.divf %lhs, %rhs : f32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }
}
