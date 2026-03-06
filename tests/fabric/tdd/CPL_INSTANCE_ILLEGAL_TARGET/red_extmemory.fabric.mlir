// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_INSTANCE_ILLEGAL_TARGET

// Named extmemory cannot be used as an instance target.
fabric.extmemory @ext
    [ldCount = 1, stCount = 0]
    : memref<?xi32>, (memref<?xi32>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)

fabric.module @test(%m: memref<?xi32>, %addr: !dataflow.bits<57>) -> (!dataflow.bits<32>) {
  %data, %done = fabric.instance @ext(%m, %addr)
      : (memref<?xi32>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %data : !dataflow.bits<32>
}
