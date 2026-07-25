// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-dfg-sim %t.dir/raw.mlir --graph fmuladd --output %t.json 2>&1 | FileCheck %s --check-prefix=FMULADD-REJECT
// RUN: loom-dfg-sim %t.dir/split.mlir --graph split_mulf_addf --output %t.split.json
// RUN: FileCheck %s --check-prefix=SPLIT < %t.split.json

// FMULADD-REJECT: finalized graph contains unregistered actor 'llvm.intr.fmuladd'

// SPLIT-DAG: "kind": "dfg_sim_report"
// SPLIT-DAG: "workload": "split_mulf_addf"
// SPLIT-DAG: "graph": "split_mulf_addf"
// SPLIT-DAG: "status": "pass"
// SPLIT-DAG: "metric_definition": "weighted_operations_plus_library_work_diversity_and_address.v1"
// SPLIT-DAG: "operation_cost_score": 13
// SPLIT-DAG: "wavefront_steps": 4
// SPLIT-DAG: "event_count": 6
// SPLIT-DAG: "f32:10"

//--- raw.mlir
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
}

//--- split.mlir
module {
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
