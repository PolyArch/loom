// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_ADG_COMBINATIONAL_LOOP

// A switch wrapped in a fabric.module is still combinational.
// Two instances of the wrapper form a zero-delay cycle.
fabric.module @sw_wrap(%a: i32, %b: i32) -> (i32, i32) {
  %o:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a, %b : i32 -> i32, i32
  fabric.yield %o#0, %o#1 : i32, i32
}

fabric.module @top(%x: i32) -> (i32) {
  %u:2 = fabric.instance @sw_wrap(%x, %v#0) : (i32, i32) -> (i32, i32)
  %v:2 = fabric.instance @sw_wrap(%u#0, %u#1) : (i32, i32) -> (i32, i32)
  fabric.yield %v#1 : i32
}
