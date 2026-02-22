// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_TYPE_MISMATCH

// Named FIFO with category mismatch: native i32 vs tagged<i32,i4>.
fabric.fifo @bad_fifo [depth = 2] : (i32) -> (!dataflow.tagged<i32, i4>)

fabric.module @test(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 4] %a : i32
  fabric.yield %out : i32
}
