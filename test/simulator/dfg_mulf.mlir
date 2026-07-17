// RUN: loom-dfg-sim %s --graph mulf --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "mulf"
// CHECK-DAG: "graph": "mulf"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "weighted_operations_plus_library_work_diversity_and_address.v1"
// CHECK-DAG: "operation_cost_score": 9
// CHECK-DAG: "wavefront_steps": 3
// CHECK-DAG: "event_count": 4
// CHECK-DAG: "f32:3"

module {
  dataflow.graph private @mulf(%ctrl: none) -> (f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %ctrl {const_value = 1.500000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 2.000000e+00 : f32} : f32
    %result = arith.mulf %lhs, %rhs : f32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }
}
