// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU has 2-input arity. Patterns with different arity must not match.

fabric.module @hw_2in(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// 1-input pattern: arity mismatch.
// CHECK-LABEL: @pat_unary_unmatched
func.func @pat_unary_unmatched(%x: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%a = %x : i32) -> i32
       attributes {loom.is_pattern} {
    dataflow.yield %a : i32
  }
  return %r : i32
}

// 3-input pattern: arity mismatch.
// CHECK-LABEL: @pat_ternary_unmatched
func.func @pat_ternary_unmatched(%x: i32, %y: i32, %z: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32, %c = %z : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.addi %a, %b : i32
    %n = arith.addi %m, %c : i32
    dataflow.yield %n : i32
  }
  return %r : i32
}
