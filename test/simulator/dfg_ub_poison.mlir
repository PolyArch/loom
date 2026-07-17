// RUN: loom-dfg-sim %s --graph poison_top --output %t.top.json
// RUN: FileCheck %s --check-prefix=TOP < %t.top.json
// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph poison_structured --arg 0=true --output %t.structured.json
// RUN: FileCheck %s --check-prefix=STRUCTURED < %t.structured.json

// TOP-DAG: "graph": "poison_top"
// TOP-DAG: "status": "pass"
// TOP-DAG: "ub.poison": 1
// TOP-DAG: "final_outputs": [
// TOP-DAG: "none"
// TOP-DAG: "i32:0"

// STRUCTURED-DAG: "graph": "poison_structured"
// STRUCTURED-DAG: "status": "pass"
// STRUCTURED-DAG: "ub.poison": 1
// STRUCTURED-DAG: "final_outputs": [
// STRUCTURED-DAG: "none"
// STRUCTURED-DAG: "i32:0"

module {
  dataflow.graph.func private @poison_top(%ctrl: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %poison = ub.poison : i32
    %published:2 = dataflow.sync %ctrl, %poison
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph.func private @poison_structured(%ctrl: none, %cond: i1)
      -> (none, i32)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = arith.constant 0 : i32
    %value = scf.if %cond -> (i32) {
      %poison = ub.poison : i32
      scf.yield %poison : i32
    } else {
      scf.yield %zero : i32
    }
    dataflow.graph.return %ctrl, %value : none, i32
  }
}
