// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU implements either a*b or a*b+c via mux/demux. Two patterns: pure
// multiply and multiply-accumulate.

func.func @hw_mac(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                  %c: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>,
                 %z = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    %mul = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %d0, %d1 = fabric.demux %mul : !fabric.bits<32> -> 2
    %add = fabric.op [@arith.addi] (%d1, %z)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %out = fabric.mux %d0, %add : !fabric.bits<32>
    fabric.yield %out : !fabric.bits<32>
  }
  return
}

// Pattern: pure multiply.
// CHECK-LABEL: @pat_mul
func.func @pat_mul(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "demux#0{sel=0}; mux#0{sel=0}"
  // CHECK-SAME: loom.matched_fu = "@hw_mac#0"
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32
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
  // CHECK-SAME: loom.match_config = "demux#0{sel=1}; mux#0{sel=1}"
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
