// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-dfg-sim %t.dir/raw.mlir --graph fmuladd --output %t.json
// RUN: FileCheck %s --check-prefix=FMA < %t.json
// RUN: loom-dfg-sim %t.dir/split.mlir --graph split_mulf_addf --output %t.split.json
// RUN: FileCheck %s --check-prefix=SPLIT < %t.split.json

// FMA-DAG: "kind": "dfg_sim_report"
// FMA-DAG: "workload": "fmuladd"
// FMA-DAG: "graph": "fmuladd"
// FMA-DAG: "status": "pass"
// FMA-DAG: "wavefront_steps": 3
// FMA-DAG: "event_count": 5
// FMA-DAG: "math.fma": 1
// FMA-DAG: "f32:10"

// SPLIT-DAG: "kind": "dfg_sim_report"
// SPLIT-DAG: "workload": "split_mulf_addf"
// SPLIT-DAG: "graph": "split_mulf_addf"
// SPLIT-DAG: "status": "pass"
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
    %result = math.fma %lhs, %rhs, %acc : f32
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
