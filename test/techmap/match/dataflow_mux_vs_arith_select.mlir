// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// Pins (canonical reference): arith.select and dataflow.mux are
// distinct op kinds with distinct semantics. VF2 distinguishes them by
// op-name, so an arith.select pattern must NOT match a dataflow.mux FU
// and vice versa. This test belongs in the match suite as the canonical
// example separating the four entities (claim 4 vs claim 2 in the
// design statement).
//
// To satisfy the pe uniform-W rule both FUs are exposed at
// bits<1> throughout (sel is fixed bits<1> and the data ports accept
// any width via TypeParam(0)).

fabric.module @hw_mux2(%sel : !fabric.bits<1>, %a : !fabric.bits<1>, %b : !fabric.bits<1>) {
  fabric.pe [spatial] (%psel = %sel : !fabric.bits<1>,
                    %pa = %a : !fabric.bits<1>,
                    %pb = %b : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%s = %psel : !fabric.bits<1>,
              %x = %pa : !fabric.bits<1>,
              %y = %pb : !fabric.bits<1>) -> !fabric.bits<1> {
      %o = fabric.op [@dataflow.mux] (%s, %x, %y)
           : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
             -> !fabric.bits<1>
      fabric.yield %o : !fabric.bits<1>
    }
  }
  fabric.yield
}

fabric.module @hw_select(%c : !fabric.bits<1>, %a : !fabric.bits<1>, %b : !fabric.bits<1>) {
  fabric.pe [spatial] (%pc = %c : !fabric.bits<1>,
                    %pa = %a : !fabric.bits<1>,
                    %pb = %b : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%cn = %pc : !fabric.bits<1>,
              %x = %pa : !fabric.bits<1>,
              %y = %pb : !fabric.bits<1>) -> !fabric.bits<1> {
      %o = fabric.op [@arith.select] (%cn, %x, %y)
           : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
             -> !fabric.bits<1>
      fabric.yield %o : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @pat_select
func.func @pat_select(%c: i1, %a: i1, %b: i1) -> i1 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_select#0"
  %r = dataflow.subgraph(%cn = %c : i1, %x = %a : i1, %y = %b : i1) -> i1
       attributes {loom.is_pattern} {
    %o = arith.select %cn, %x, %y : i1
    dataflow.yield %o : i1
  }
  return %r : i1
}

// CHECK-LABEL: @pat_mux
func.func @pat_mux(%s: i1, %a: i1, %b: i1) -> i1 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_mux2#0"
  %r = dataflow.subgraph(%sn = %s : i1, %x = %a : i1, %y = %b : i1) -> i1
       attributes {loom.is_pattern} {
    %o = dataflow.mux %sn, %x, %y : (i1, i1, i1) -> i1
    dataflow.yield %o : i1
  }
  return %r : i1
}
