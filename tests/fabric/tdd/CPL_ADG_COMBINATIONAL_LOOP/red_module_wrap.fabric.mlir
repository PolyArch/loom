// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_ADG_COMBINATIONAL_LOOP

// A switch wrapped in a fabric.module is still combinational.
// Two instances of the wrapper form a zero-delay cycle.
fabric.module @sw_wrap(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %o:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o#0, %o#1 : !dataflow.bits<32>, !dataflow.bits<32>
}

fabric.module @top(%x: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %u:2 = fabric.instance @sw_wrap(%x, %v#0) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
  %v:2 = fabric.instance @sw_wrap(%u#0, %u#1) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
  fabric.yield %v#1 : !dataflow.bits<32>
}
