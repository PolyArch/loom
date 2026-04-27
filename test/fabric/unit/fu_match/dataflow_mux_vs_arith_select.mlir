// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// arith.select and dataflow.mux are different op kinds with different
// semantics:
//   * arith.select: strict-SSA, eager evaluation; consumes both data
//                   inputs regardless of sel.
//   * dataflow.mux: data-dependent gating; consumes only the selected
//                   data input.
// VF2 distinguishes them by op-name; therefore an arith.select user
// pattern must NOT match a dataflow.mux-only FU, and vice versa.

// FU offering a fixed-arity 2-input dataflow.mux (M=2 is a legal lower
// bound for mux: numIns=3 == 1 sel + 2 data).
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

// FU offering a fixed-arity arith.select.
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

// arith.select pattern matches @hw_select but NOT @hw_mux2.
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

// dataflow.mux pattern matches @hw_mux2 but NOT @hw_select.
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
