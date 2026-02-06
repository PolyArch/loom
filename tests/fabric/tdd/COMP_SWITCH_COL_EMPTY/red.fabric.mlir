// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_SWITCH_COL_EMPTY

// Input column 1 (second input) has no connections in any output row.
fabric.module @test(%a: i32, %b: i32) -> (i32, i32) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 0, 1, 0]] %a, %b : i32 -> i32, i32
  fabric.yield %o1, %o2 : i32, i32
}
