// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_FIFO_INVALID_TYPE

// FIFO with an invalid type (i3 is not a native type).
fabric.module @test_fifo_invalid_type(%a: i3) -> (i3) {
  %out = fabric.fifo [depth = 2] %a : i3
  fabric.yield %out : i3
}
