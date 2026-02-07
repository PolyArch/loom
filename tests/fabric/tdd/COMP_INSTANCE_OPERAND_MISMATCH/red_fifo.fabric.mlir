// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_INSTANCE_OPERAND_MISMATCH

// Named fifo expects 1 input, but instance provides 2.
fabric.fifo @buf [depth = 2] : (i32) -> (i32)

fabric.module @top(%a: i32, %b: i32) -> (i32) {
  %out = fabric.instance @buf(%a, %b) : (i32, i32) -> (i32)
  fabric.yield %out : i32
}
