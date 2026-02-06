// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_SWITCH_ROW_EMPTY

// Output row 1 (second row) has no connections (all zeros).
fabric.module @test(%a: i32, %b: i32) -> (i32, i32) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 1, 0, 0]] %a, %b : i32 -> i32, i32
  fabric.yield %o1, %o2 : i32, i32
}
