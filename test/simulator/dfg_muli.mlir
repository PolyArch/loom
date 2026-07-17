// RUN: loom-dfg-sim %s --graph muli --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "muli"
// CHECK-DAG: "graph": "muli"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "weighted_operations_plus_library_work_diversity_and_address.v1"
// CHECK-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// CHECK-DAG: "operation_cost_score": 9
// CHECK-DAG: "wavefront_steps": 3
// CHECK-DAG: "event_count": 4
// CHECK-DAG: "i32:42"

module {
  dataflow.graph.func private @muli(%ctrl: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %ctrl {const_value = 6 : i32} : i32
    %rhs = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %result = arith.muli %lhs, %rhs : i32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
