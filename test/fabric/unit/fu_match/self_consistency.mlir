// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Self-consistency: for a single FU offering all three bitwise ops in one
// hardware-share group, every matching pattern should resolve to the same
// FU with the corresponding op_sel.

fabric.module @hw_bitwise {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.andi, @arith.ori, @arith.xori] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @pat_andi
func.func @pat_andi(%x: i32, %y: i32) -> i32 {
  // CHECK: loom.match_config = "op#0{op_sel=arith.andi}"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.andi %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// CHECK-LABEL: @pat_ori
func.func @pat_ori(%x: i32, %y: i32) -> i32 {
  // CHECK: loom.match_config = "op#0{op_sel=arith.ori}"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.ori %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// CHECK-LABEL: @pat_xori
func.func @pat_xori(%x: i32, %y: i32) -> i32 {
  // CHECK: loom.match_config = "op#0{op_sel=arith.xori}"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.xori %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
