// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Pins (canonical reference): arith.select and dataflow.mux are
// distinct op kinds with distinct semantics. VF2 distinguishes them by
// op-name, so an arith.select pattern must NOT match a dataflow.mux FU
// and vice versa. This test belongs in the match suite as the canonical
// example separating the four entities (claim 4 vs claim 2 in the
// design statement).

func.func @hw_mux2(%sel: !fabric.bits<1>,
                   %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%s = %sel : !fabric.bits<1>,
                 %x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %o = fabric.op [@dataflow.mux] (%s, %x, %y)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
           -> !fabric.bits<32>
    fabric.yield %o : !fabric.bits<32>
  }
  return
}

func.func @hw_select(%c: !fabric.bits<1>,
                     %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%cn = %c : !fabric.bits<1>,
                 %x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %o = fabric.op [@arith.select] (%cn, %x, %y)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
           -> !fabric.bits<32>
    fabric.yield %o : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @pat_select
func.func @pat_select(%c: i1, %a: i32, %b: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_select#0"
  %r = dataflow.subgraph(%cn = %c : i1, %x = %a : i32, %y = %b : i32) -> i32
       attributes {loom.is_pattern} {
    %o = arith.select %cn, %x, %y : i32
    dataflow.yield %o : i32
  }
  return %r : i32
}

// CHECK-LABEL: @pat_mux
func.func @pat_mux(%s: i1, %a: i32, %b: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_mux2#0"
  %r = dataflow.subgraph(%sn = %s : i1, %x = %a : i32, %y = %b : i32) -> i32
       attributes {loom.is_pattern} {
    %o = dataflow.mux %sn, %x, %y : (i1, i32, i32) -> i32
    dataflow.yield %o : i32
  }
  return %r : i32
}
