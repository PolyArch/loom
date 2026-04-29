// RUN: loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// A func.func that contains more than one `dataflow.subgraph` is also
// rejected during input validation. Same annotation + warning as the
// zero-subgraph case.

// CHECK: warning: {{.*}}func.func @two_subgraphs: invalid_input
// CHECK: loom.synth_failed = "invalid_input"

func.func @two_subgraphs(%a: i32, %b: i32) -> (i32, i32) {
  %r0 = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  %r1 = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r0, %r1 : i32, i32
}
