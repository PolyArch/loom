// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_INSTANCE_ILLEGAL_TARGET

// Named fifo cannot be used as an instance target.
fabric.fifo @buf [depth = 2] : (!dataflow.bits<32>) -> (!dataflow.bits<32>)

fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @buf(%a) : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}
