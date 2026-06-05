// RUN: loom-dfg-sim %s --graph fmuladd --arg 0=none --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "fmuladd"
// CHECK-DAG: "graph": "fmuladd"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "optimistic_cycles": 2
// CHECK-DAG: "event_count": 4
// CHECK-DAG: "f32:10"

module {
  dataflow.graph.func private @fmuladd(%ctrl: none) -> (none, f32) {
    %lhs = dataflow.constant %ctrl {const_value = 2.000000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 3.000000e+00 : f32} : f32
    %acc = dataflow.constant %ctrl {const_value = 4.000000e+00 : f32} : f32
    %result = llvm.intr.fmuladd(%lhs, %rhs, %acc) : (f32, f32, f32) -> f32
    dataflow.graph.return %ctrl, %result : none, f32
  }
}
