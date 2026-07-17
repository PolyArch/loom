// RUN: loom-dfg-sim %s --graph fmuladd --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph split_mulf_addf --output %t.split.json
// RUN: FileCheck %s --check-prefix=SPLIT < %t.split.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "fmuladd"
// CHECK-DAG: "graph": "fmuladd"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "weighted_operations_plus_library_work_diversity_and_address.v1"
// CHECK-DAG: "operation_cost_score": 15
// CHECK-DAG: "wavefront_steps": 3
// CHECK-DAG: "event_count": 5
// CHECK-DAG: "f32:10"

// SPLIT-DAG: "kind": "dfg_sim_report"
// SPLIT-DAG: "workload": "split_mulf_addf"
// SPLIT-DAG: "graph": "split_mulf_addf"
// SPLIT-DAG: "status": "pass"
// SPLIT-DAG: "metric_definition": "weighted_operations_plus_library_work_diversity_and_address.v1"
// SPLIT-DAG: "operation_cost_score": 13
// SPLIT-DAG: "wavefront_steps": 4
// SPLIT-DAG: "event_count": 6
// SPLIT-DAG: "f32:10"

module {
  dataflow.graph private @fmuladd(%ctrl: none) -> (f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %ctrl {const_value = 2.000000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 3.000000e+00 : f32} : f32
    %acc = dataflow.constant %ctrl {const_value = 4.000000e+00 : f32} : f32
    %result = llvm.intr.fmuladd(%lhs, %rhs, %acc) : (f32, f32, f32) -> f32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }

  dataflow.graph private @split_mulf_addf(%ctrl: none) -> (f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %ctrl {const_value = 2.000000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 3.000000e+00 : f32} : f32
    %acc = dataflow.constant %ctrl {const_value = 4.000000e+00 : f32} : f32
    %product = arith.mulf %lhs, %rhs : f32
    %result = arith.addf %product, %acc : f32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }
}
