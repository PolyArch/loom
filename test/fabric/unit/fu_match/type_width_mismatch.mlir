// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU is 32-bit only. A 64-bit pattern must not match.

func.func @hw_32bit(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @pat_i32_match
func.func @pat_i32_match(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_32bit#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.addi %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// CHECK-LABEL: @pat_i64_unmatched
func.func @pat_i64_unmatched(%x: i64, %y: i64) -> i64 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%a = %x : i64, %b = %y : i64) -> i64
       attributes {loom.is_pattern} {
    %k = arith.addi %a, %b : i64
    dataflow.yield %k : i64
  }
  return %r : i64
}
