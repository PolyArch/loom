// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Pin commutativity-preserving operand swaps under the VF2 matcher.
// addi is commutative, so both (a+b) and (b+a) patterns must bind to the
// FU's addi configuration.

fabric.module @hw_addi {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
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

// CHECK-LABEL: @pat_canonical
func.func @pat_canonical(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_addi#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.addi %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// CHECK-LABEL: @pat_swapped
func.func @pat_swapped(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_addi#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.addi %b, %a : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
