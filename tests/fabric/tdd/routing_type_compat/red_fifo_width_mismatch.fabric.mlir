// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_TYPE_MISMATCH

// Named FIFO with incompatible bit widths: i32 vs i16.
fabric.fifo @bad_fifo [depth = 2] : (!dataflow.bits<32>) -> (!dataflow.bits<16>)

fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  fabric.fifo @inline_buf [depth = 4] : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  %pe = fabric.instance @inline_buf(%a) {sym_name = "f0"} : (!dataflow.bits<32>) -> !dataflow.bits<32>
  fabric.yield %pe : !dataflow.bits<32>
}
