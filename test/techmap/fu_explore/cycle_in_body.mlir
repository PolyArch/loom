// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// Pins: some configurable ops (e.g. dataflow.carry / dataflow.invariant)
// naturally need a cycle inside the FU body. The fabric.fu body region
// is a graph region (RegionKind::Graph), so a textual back-reference
// like `%next` below is well-formed. The enumerator's two-pass
// materializer resolves the back-edge by synthesizing placeholder
// operands during its first pass and rewiring them once the producer
// has been built.

// CHECK: dataflow.subgraph
func.func @fu_self_feedback(%c: !fabric.bits<1>, %i: !fabric.bits<32>) {
  %r = fabric.fu(%cc = %c : !fabric.bits<1>,
                 %ii = %i : !fabric.bits<32>) -> !fabric.bits<32> {
    %acc = fabric.op [@dataflow.carry] (%cc, %ii, %next)
           : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
             -> !fabric.bits<32>
    %next = fabric.op [@arith.addi] (%acc, %ii)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %acc : !fabric.bits<32>
  }
  return
}
