// RUN: loom-dfg-sim %s --graph muli --arg 0=none --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "muli"
// CHECK-DAG: "graph": "muli"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// CHECK-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// CHECK-DAG: "optimistic_cycles": 5
// CHECK-DAG: "wavefront_steps": 2
// CHECK-DAG: "event_count": 3
// CHECK-DAG: "i32:42"

module {
  dataflow.graph.func private @muli(%ctrl: none) -> (none, i32) {
    %lhs = dataflow.constant %ctrl {const_value = 6 : i32} : i32
    %rhs = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %result = arith.muli %lhs, %rhs : i32
    dataflow.graph.return %ctrl, %result : none, i32
  }
}
