// RUN: loom-dfg-sim %s --graph fmuladd --arg 0=none --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph split_mulf_addf --arg 0=none --output %t.split.json
// RUN: FileCheck %s --check-prefix=SPLIT < %t.split.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "fmuladd"
// CHECK-DAG: "graph": "fmuladd"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// CHECK-DAG: "optimistic_cycles": 10
// CHECK-DAG: "wavefront_steps": 2
// CHECK-DAG: "event_count": 4
// CHECK-DAG: "f32:10"

// SPLIT-DAG: "kind": "dfg_sim_report"
// SPLIT-DAG: "workload": "split_mulf_addf"
// SPLIT-DAG: "graph": "split_mulf_addf"
// SPLIT-DAG: "status": "pass"
// SPLIT-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// SPLIT-DAG: "optimistic_cycles": 8
// SPLIT-DAG: "wavefront_steps": 3
// SPLIT-DAG: "event_count": 5
// SPLIT-DAG: "f32:10"

module {
  dataflow.graph.func private @fmuladd(%ctrl: none) -> (none, f32) {
    %lhs = dataflow.constant %ctrl {const_value = 2.000000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 3.000000e+00 : f32} : f32
    %acc = dataflow.constant %ctrl {const_value = 4.000000e+00 : f32} : f32
    %result = llvm.intr.fmuladd(%lhs, %rhs, %acc) : (f32, f32, f32) -> f32
    dataflow.graph.return %ctrl, %result : none, f32
  }

  dataflow.graph.func private @split_mulf_addf(%ctrl: none) -> (none, f32) {
    %lhs = dataflow.constant %ctrl {const_value = 2.000000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 3.000000e+00 : f32} : f32
    %acc = dataflow.constant %ctrl {const_value = 4.000000e+00 : f32} : f32
    %product = arith.mulf %lhs, %rhs : f32
    %result = arith.addf %product, %acc : f32
    dataflow.graph.return %ctrl, %result : none, f32
  }
}
