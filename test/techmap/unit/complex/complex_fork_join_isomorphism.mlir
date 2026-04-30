// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: an FU implements a mac (mul + add). The user graph contains a
// mul whose result feeds an add whose OTHER operand position is the
// non-mul value -- i.e., the partitioner's multi-op match must accept a
// commutativity-preserving permutation of the addi operands. The old
// operand[0]-chain matcher would have rejected this shape; the VF2-based
// matcher binds the mul + add pair as a single subgraph.

// CHECK-LABEL: @fu_mac
fabric.module @fu_mac(%cast0_fu_mac : !fabric.bits<32>, %cast1_fu_mac : !fabric.bits<32>, %cast2_fu_mac : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_mac : !fabric.bits<32>, %b = %cast1_fu_mac : !fabric.bits<32>, %c = %cast2_fu_mac : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>,
                 %z = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    %m = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %s = fabric.op [@arith.addi] (%m, %z)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %s : !fabric.bits<32>
  }
  }
  fabric.yield
}


// User graph: addi has the multiply's result on the second operand
// (commutativity swap). The VF2-based multi-op matcher accepts the
// fork/join shape and emits exactly one dataflow.subgraph wrapping both
// the muli and the addi.
// CHECK-LABEL: @graph_mac_swapped
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_mac_swapped(%a: i32, %b: i32, %c: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %m = arith.muli %x, %y : i32
    // Note: muli's result is on the SECOND operand of addi.
    %s = arith.addi %z, %m : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
