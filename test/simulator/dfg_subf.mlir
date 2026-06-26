// RUN: loom-dfg-sim %s --graph subf --arg 0=none --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "subf"
// CHECK-DAG: "graph": "subf"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// CHECK-DAG: "optimistic_cycles": 6
// CHECK-DAG: "wavefront_steps": 2
// CHECK-DAG: "event_count": 3
// CHECK-DAG: "f32:3.250000"

module {
  dataflow.graph.func private @subf(%ctrl: none) -> (none, f32) {
    %lhs = dataflow.constant %ctrl {const_value = 5.500000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 2.250000e+00 : f32} : f32
    %result = arith.subf %lhs, %rhs : f32
    dataflow.graph.return %ctrl, %result : none, f32
  }
}
