// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_INSTANCE_ILLEGAL_TARGET

// Named memory cannot be used as an instance target.
fabric.memory @mem
    [ldCount = 1, stCount = 0]
    : memref<64xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)

fabric.module @test(%ext: memref<64xi32>, %addr: !dataflow.bits<57>) -> (!dataflow.bits<32>) {
  %data, %done = fabric.instance @mem(%ext, %addr)
      : (memref<64xi32>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %data : !dataflow.bits<32>
}
