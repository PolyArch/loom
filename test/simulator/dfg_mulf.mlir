// RUN: loom-dfg-sim %s --graph mulf --arg 0=none --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "mulf"
// CHECK-DAG: "graph": "mulf"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "optimistic_operation_latency_sum"
// CHECK-DAG: "optimistic_cycles": 5
// CHECK-DAG: "wavefront_steps": 2
// CHECK-DAG: "event_count": 3
// CHECK-DAG: "f32:3"

module {
  dataflow.graph.func private @mulf(%ctrl: none) -> (none, f32) {
    %lhs = dataflow.constant %ctrl {const_value = 1.500000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 2.000000e+00 : f32} : f32
    %result = arith.mulf %lhs, %rhs : f32
    dataflow.graph.return %ctrl, %result : none, f32
  }
}
