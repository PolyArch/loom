// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_TYPE_MISMATCH

// Named FIFO with category mismatch: native i32 vs tagged<i32,i4>.
fabric.fifo @bad_fifo [depth = 2] : (!dataflow.bits<32>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>)

fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 4] %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
