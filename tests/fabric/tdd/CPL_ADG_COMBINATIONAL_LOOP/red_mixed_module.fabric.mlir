// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_ADG_COMBINATIONAL_LOOP

// A module with both a combinational path (switch output 0 -> yield 0)
// and a sequential path (switch output 1 -> fifo -> yield 1). Feedback
// through the combinational path must still be detected.
fabric.module @mixed(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %sw:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  %seq = fabric.fifo [depth = 2] %sw#1 : !dataflow.bits<32>
  fabric.yield %sw#0, %seq : !dataflow.bits<32>, !dataflow.bits<32>
}

fabric.module @top(%x: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %u:2 = fabric.instance @mixed(%x, %v#0) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
  %v:2 = fabric.instance @mixed(%u#0, %u#1) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
  fabric.yield %v#1 : !dataflow.bits<32>
}
