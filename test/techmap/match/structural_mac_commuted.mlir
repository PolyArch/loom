// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Pins: structural isomorphism with commuted operands. The user
// subgraph computes (z + (x*y)) where the multiply's result lands on
// the SECOND operand of the addi. The MAC FU has the multiply's result
// on the FIRST operand of its addi. VF2 must accept the swap because
// arith.addi is commutative; the match should still bind both ops to
// the FU and report an annotated dataflow.subgraph.

fabric.module @hw_mac(%cast0_hw_mac : !fabric.bits<32>, %cast1_hw_mac : !fabric.bits<32>, %cast2_hw_mac : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_hw_mac : !fabric.bits<32>, %b = %cast1_hw_mac : !fabric.bits<32>, %c = %cast2_hw_mac : !fabric.bits<32>) -> !fabric.bits<32> {
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


// CHECK-LABEL: @pat_mac_swapped
func.func @pat_mac_swapped(%x: i32, %y: i32, %z: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_mac#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32, %c = %z : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %a, %b : i32
    // muli's result on the second operand of addi.
    %s = arith.addi %c, %m : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
