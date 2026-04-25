// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Multiple FUs in the module. Each pattern is matched against the first FU
// (in module walk order) that supports it.

func.func @hw_intonly(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi, @arith.subi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

func.func @hw_floatonly(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addf, @arith.subf] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// Integer pattern -> matches first FU.
// CHECK-LABEL: @pat_addi
func.func @pat_addi(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_intonly#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.addi %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// Float pattern -> matches second FU.
// CHECK-LABEL: @pat_subf
func.func @pat_subf(%x: f32, %y: f32) -> f32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_floatonly#0"
  %r = dataflow.subgraph(%a = %x : f32, %b = %y : f32) -> f32
       attributes {loom.is_pattern} {
    %k = arith.subf %a, %b : f32
    dataflow.yield %k : f32
  }
  return %r : f32
}

// Multiplication: neither FU supports it.
// CHECK-LABEL: @pat_muli_unmatched
func.func @pat_muli_unmatched(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.muli %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
