// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU implements either a*b or a*b+c via mux/demux. Two patterns: pure
// multiply and multiply-accumulate.

fabric.module @hw_mac(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %mul = fabric.op [@arith.muli] (%x, %y)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %d0, %d1 = fabric.demux %mul : !fabric.bits<32> -> 2
      %add = fabric.op [@arith.addi] (%d1, %z)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %out = fabric.mux %d0, %add : !fabric.bits<32>
      fabric.yield %out : !fabric.bits<32>
    }
  }
  fabric.yield
}

// Pattern: pure multiply. The FU has 3 inputs but the demux.sel=0 /
// mux.sel=0 configuration leaves the third input dead, so the matching
// template is a 2-input subgraph. The user pattern is 2-input as well.
// CHECK-LABEL: @pat_mul
func.func @pat_mul(%a: i32, %b: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: demux#0{sel=0,discard=false,disconnect=false}; mux#0{sel=0,discard=false,disconnect=false}
  // CHECK-SAME: loom.matched_fu = "@hw_mac#0"
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %x, %y : i32
    dataflow.yield %m : i32
  }
  return %r : i32
}

// Pattern: multiply-accumulate.
// CHECK-LABEL: @pat_mac
func.func @pat_mac(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: demux#0{sel=1,discard=false,disconnect=false}; mux#0{sel=1,discard=false,disconnect=false}
  // CHECK-SAME: loom.matched_fu = "@hw_mac#0"
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %x, %y : i32
    %s = arith.addi %m, %z : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// Pattern: subtract-accumulate (the FU's adder is hard-wired to addi, so
// this should not match).
// CHECK-LABEL: @pat_msac_unmatched
func.func @pat_msac_unmatched(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %x, %y : i32
    %s = arith.subi %m, %z : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
