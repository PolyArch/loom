// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_SWITCH_TABLE_SHAPE

// connectivity_table has 3 entries but a 2x2 switch requires 4.
fabric.module @test(%a: i32, %b: i32) -> (i32, i32) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 0, 1]] %a, %b : i32 -> i32, i32
  fabric.yield %o1, %o2 : i32, i32
}
