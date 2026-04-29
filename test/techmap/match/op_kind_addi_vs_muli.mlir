// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Pins: VF2 matches by op-name first. An FU advertising both addi and
// muli (multi-member share group is not legal here so we use two FUs)
// must let an arith.muli user pattern bind to the muli FU and an
// arith.addi user pattern bind to the addi FU. Op-kind selection is
// part of the sw_configs description (op_sel keyword for share groups
// or implicit single-symbol picking otherwise).

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


fabric.module @hw_muli(%cast0_hw_muli : !fabric.bits<32>, %cast1_hw_muli : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_hw_muli : !fabric.bits<32>, %b = %cast1_hw_muli : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @pat_addi
func.func @pat_addi(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_addi#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.addi %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// CHECK-LABEL: @pat_muli
func.func @pat_muli(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_muli#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.muli %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
