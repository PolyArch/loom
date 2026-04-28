// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins: an FU advertising a single fabric.op[@arith.select] has no
// sw_configs axis (arith.select is non-configurable per the design
// statement), so the enumerator emits exactly one template wrapping
// arith.select.

// CHECK-LABEL: @fu_select
func.func @fu_select(%s: !fabric.bits<1>, %a: !fabric.bits<32>,
                     %b: !fabric.bits<32>) {
  %r = fabric.fu(%ss = %s : !fabric.bits<1>,
                 %aa = %a : !fabric.bits<32>,
                 %bb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %k = fabric.op [@arith.select] (%ss, %aa, %bb)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  // CHECK: dataflow.subgraph
  // CHECK: arith.select
  // CHECK-NOT: dataflow.subgraph
  return
}
