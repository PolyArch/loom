// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-lower %t.lowered.mlir | FileCheck %s

// A nested loop selected inside scf.while.before retires exactly once for
// every before-region activation. Its close must therefore remain aligned
// with the complete outer condition stream, including the final false phase.
// CHECK-LABEL: dataflow.graph private @nested_while_selection
// CHECK: dataflow.gate
// CHECK: dataflow.graph.return
// CHECK-NOT: scf.if
// CHECK-NOT: scf.while
dataflow.graph private @nested_while_selection(
    %start: none, %outer_limit: i32,
    %input: memref<?xi32>, %output: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 2>,
                result_segments = array<i32: 0, 0, 0>} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %outer = scf.while (%i = %zero) : (i32) -> i32 {
    %index = arith.index_cast %i : i32 to index
    %begin = memref.load %input[%index] : memref<?xi32>
    %next = arith.addi %i, %one : i32
    %next_index = arith.index_cast %next : i32 to index
    %end = memref.load %input[%next_index] : memref<?xi32>
    %selected = arith.cmpi ult, %begin, %end : i32
    %selected_value = scf.if %selected -> (i32) {
      %inner:2 = scf.while (%j = %begin, %sum = %zero)
          : (i32, i32) -> (i32, i32) {
        %inner_index = arith.index_cast %j : i32 to index
        %value = memref.load %input[%inner_index] : memref<?xi32>
        %inner_next = arith.addi %j, %one : i32
        %next_sum = arith.addi %sum, %value : i32
        %continue = arith.cmpi ult, %inner_next, %end : i32
        scf.condition(%continue) %inner_next, %next_sum : i32, i32
      } do {
      ^bb0(%j: i32, %sum: i32):
        scf.yield %j, %sum : i32, i32
      }
      scf.yield %inner#1 : i32
    } else {
      scf.yield %zero : i32
    }
    memref.store %selected_value, %output[%index] : memref<?xi32>
    %continue = arith.cmpi slt, %next, %outer_limit : i32
    scf.condition(%continue) %next : i32
  } do {
  ^bb0(%i: i32):
    scf.yield %i : i32
  }
  dataflow.graph.return %start : none
}
