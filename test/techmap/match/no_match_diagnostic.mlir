// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Pins: when no FU matches a pattern, the matcher annotates the
// dataflow.subgraph with `loom.unmatched`. The user pattern below uses
// arith.muli but the only FU advertises arith.addi.

fabric.module @hw_addi(%cast0_hw_addi : !fabric.bits<32>, %cast1_hw_addi : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_hw_addi : !fabric.bits<32>, %b = %cast1_hw_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @pat_unmatched
func.func @pat_unmatched(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.muli %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
