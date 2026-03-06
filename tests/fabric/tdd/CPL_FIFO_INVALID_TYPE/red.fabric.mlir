// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_INVALID_TYPE

// Named FIFO with an invalid type (i3 is not bits<N> or none).
fabric.fifo @bad_fifo [depth = 2] : (i3) -> (i3)

fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @bad_fifo(%a) : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}
