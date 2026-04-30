// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Pins: when multiple FUs qualify, the matcher picks deterministically.
// Two FUs both implement arith.addi. The matcher must pick exactly one
// (the first match wins per pass implementation), and the choice must
// be stable across runs.

fabric.module @hw_addi_a(%cast0_hw_addi_a : !fabric.bits<32>, %cast1_hw_addi_a : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_hw_addi_a : !fabric.bits<32>, %b = %cast1_hw_addi_a : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


fabric.module @hw_addi_b(%cast0_hw_addi_b : !fabric.bits<32>, %cast1_hw_addi_b : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_hw_addi_b : !fabric.bits<32>, %b = %cast1_hw_addi_b : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @pat_addi_competition
func.func @pat_addi_competition(%x: i32, %y: i32) -> i32 {
  // First FU wins.
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_addi_a#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.addi %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
