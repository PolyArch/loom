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

// An unconditionally nested loop in scf.while.before also retires once for
// every before-region activation. Selection inside that child loop must not
// hide its close from the complete parent phase.
// CHECK-LABEL: dataflow.graph private @nested_while_with_inner_selection
// CHECK: dataflow.gate
// CHECK: dataflow.graph.return
// CHECK-NOT: scf.if
// CHECK-NOT: scf.while
dataflow.graph private @nested_while_with_inner_selection(
    %start: none, %outer_limit: i32,
    %input: memref<?xi32>, %output: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 2>,
                result_segments = array<i32: 0, 0, 0>} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %four = arith.constant 4 : i32
  %outer = scf.while (%i = %zero) : (i32) -> i32 {
    %inner:2 = scf.while (%j = %zero, %sum = %zero)
        : (i32, i32) -> (i32, i32) {
      %index = arith.index_cast %j : i32 to index
      %selected = arith.cmpi sle, %j, %i : i32
      %next_sum = scf.if %selected -> (i32) {
        %value = memref.load %input[%index] : memref<?xi32>
        %added = arith.addi %sum, %value : i32
        scf.yield %added : i32
      } else {
        scf.yield %sum : i32
      }
      %next_j = arith.addi %j, %one : i32
      %inner_continue = arith.cmpi slt, %next_j, %four : i32
      scf.condition(%inner_continue) %next_j, %next_sum : i32, i32
    } do {
    ^bb0(%j: i32, %sum: i32):
      scf.yield %j, %sum : i32, i32
    }
    %outer_index = arith.index_cast %i : i32 to index
    memref.store %inner#1, %output[%outer_index] : memref<?xi32>
    %next_i = arith.addi %i, %one : i32
    %outer_continue = arith.cmpi slt, %next_i, %outer_limit : i32
    scf.condition(%outer_continue) %next_i : i32
  } do {
  ^bb0(%i: i32):
    scf.yield %i : i32
  }
  dataflow.graph.return %start : none
}

// A lifted CFG may represent the continue decision as an integer selected by
// structured control before converting it back to i1. The selected condition
// still owns one trailing false phase and therefore one completion token.
// CHECK-LABEL: dataflow.graph private @selected_integer_while_condition
// CHECK: dataflow.carry
// CHECK: dataflow.graph.return
// CHECK-NOT: scf.if
// CHECK-NOT: scf.while
dataflow.graph private @selected_integer_while_condition(
    %start: none, %limit: i32) -> ()
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %result:2 = scf.while (%i = %zero, %value = %zero)
      : (i32, i32) -> (i32, i32) {
    %past_end = arith.cmpi sge, %i, %limit : i32
    %state:3 = scf.if %past_end -> (i32, i32, i32) {
      scf.yield %i, %value, %zero : i32, i32, i32
    } else {
      %next = arith.addi %i, %one : i32
      %at_end = arith.cmpi eq, %next, %limit : i32
      %selected_continue = scf.if %at_end -> (i32) {
        scf.yield %zero : i32
      } else {
        scf.yield %one : i32
      }
      scf.yield %next, %value, %selected_continue : i32, i32, i32
    }
    %continue = arith.trunci %state#2 : i32 to i1
    scf.condition(%continue) %state#0, %state#1 : i32, i32
  } do {
  ^bb0(%i: i32, %value: i32):
    scf.yield %i, %value : i32, i32
  }
  dataflow.graph.return %start : none
}
