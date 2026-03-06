// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_TYPE_MISMATCH

// Named FIFO where input and output types do not match.
fabric.fifo @bad_fifo [depth = 2] : (!dataflow.bits<32>) -> (!dataflow.bits<64>)

fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %pe = fabric.instance @bad_fifo(%a) {sym_name = "f0"} : (!dataflow.bits<32>) -> !dataflow.bits<64>
  fabric.yield %a : !dataflow.bits<32>
}
