// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_TYPE_MISMATCH

// Named FIFO with incompatible bit widths: i32 vs i16.
fabric.fifo @bad_fifo [depth = 2] : (i32) -> (i16)

fabric.module @test(%a: i32) -> (i32) {
  fabric.fifo @inline_buf [depth = 4] : (i32) -> (i32)
  %pe = fabric.instance @inline_buf(%a) {sym_name = "f0"} : (i32) -> i32
  fabric.yield %pe : i32
}
