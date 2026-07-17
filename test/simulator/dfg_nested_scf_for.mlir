// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph nested_for_accumulate --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "nested_for_accumulate"
// CHECK-DAG: "graph": "nested_for_accumulate"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "final_outputs": [
// CHECK-DAG: "i32:9"
// CHECK-DAG: "arith.addi": 9

module {
  dataflow.graph.func private @nested_for_accumulate(%ctrl: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %three = dataflow.constant %ctrl {const_value = 3 : i32} : i32
    %sum = scf.for %i = %zero to %three step %one iter_args(%outer = %zero) -> (i32) : i32 {
      %inner = scf.for %j = %zero to %three step %one iter_args(%acc = %outer) -> (i32) : i32 {
        %next = arith.addi %acc, %one : i32
        scf.yield %next : i32
      }
      scf.yield %inner : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }
}
