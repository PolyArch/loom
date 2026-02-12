// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_TYPE_MISMATCH

// Named FIFO where input and output types do not match.
fabric.fifo @bad_fifo [depth = 2] : (i32) -> (i64)

fabric.module @test(%a: i32) -> (i32) {
  %pe = fabric.instance @bad_fifo(%a) {sym_name = "f0"} : (i32) -> i64
  fabric.yield %a : i32
}
