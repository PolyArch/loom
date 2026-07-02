// RUN: loom-dfg-sim %s --graph poison_top --arg 0=none --output %t.top.json
// RUN: FileCheck %s --check-prefix=TOP < %t.top.json
// RUN: loom-dfg-sim %s --graph poison_structured --arg 0=none --arg 1=true --output %t.structured.json
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
// STRUCTURED-DAG: "scf.if": 1
// STRUCTURED-DAG: "final_outputs": [
// STRUCTURED-DAG: "none"
// STRUCTURED-DAG: "i32:0"

module {
  dataflow.graph.func private @poison_top(%ctrl: none) -> (none, i32) {
    %poison = ub.poison : i32
    dataflow.graph.return %ctrl, %poison : none, i32
  }

  dataflow.graph.func private @poison_structured(%ctrl: none, %cond: i1)
      -> (none, i32) {
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
