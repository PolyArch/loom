// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// Pins: some configurable ops (e.g. dataflow.carry / dataflow.invariant)
// naturally need a cycle inside the FU body. The fabric.fu body region
// is a graph region (RegionKind::Graph), so a textual back-reference
// like `%next` below is well-formed. The enumerator's two-pass
// materializer resolves the back-edge by synthesizing placeholder
// operands during its first pass and rewiring them once the producer
// has been built.
//
// To satisfy the pe uniform-W rule we expose the FU at bits<1>
// throughout (carry's TypeParam(0) data ports accept any width).

// CHECK: dataflow.subgraph
fabric.module @fu_self_feedback(%c : !fabric.bits<1>, %i : !fabric.bits<1>) {
  fabric.pe [spatial] (%pc = %c : !fabric.bits<1>,
                    %pi = %i : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%cc = %pc : !fabric.bits<1>,
              %ii = %pi : !fabric.bits<1>) -> !fabric.bits<1> {
      %acc = fabric.op [@dataflow.carry] (%cc, %ii, %next)
             : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
               -> !fabric.bits<1>
      %next = fabric.op [@arith.addi] (%acc, %ii)
              : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %acc : !fabric.bits<1>
    }
  }
  fabric.yield
}
