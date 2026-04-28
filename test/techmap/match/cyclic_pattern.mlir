// RUN: loom %s -loom-map-subgraph-to-fus 2>&1 | FileCheck %s

// Pins: a cyclic user pattern (carry + addi self-feedback) should match
// an FU whose body holds the same cyclic structure. The fabric.fu body
// is a graph region, so a textual back-reference (`%next`) is
// well-formed and the enumerator handles the cycle via its two-pass
// materializer.

// CHECK: loom.matched_fu
func.func @hw_carry_loop(%cond: !fabric.bits<1>, %init: !fabric.bits<32>) {
  %r = fabric.fu(%c = %cond : !fabric.bits<1>,
                 %i = %init : !fabric.bits<32>) -> !fabric.bits<32> {
    %acc = fabric.op [@dataflow.carry] (%c, %i, %next)
           : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
             -> !fabric.bits<32>
    %next = fabric.op [@arith.addi] (%acc, %i)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %acc : !fabric.bits<32>
  }
  return
}

func.func @pat_carry_loop(%cond: i1, %init: i32) -> i32 {
  %r = dataflow.subgraph(%c = %cond : i1, %i = %init : i32) -> i32
       attributes {loom.is_pattern} {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %i : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}
