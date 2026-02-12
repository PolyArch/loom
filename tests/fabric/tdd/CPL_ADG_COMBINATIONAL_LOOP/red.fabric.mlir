// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_ADG_COMBINATIONAL_LOOP

// Two switches form a purely combinational cycle (no sequential element).
fabric.module @test_comb_loop(%a: i32) -> (i32) {
  %sw0:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a, %sw1#0 : i32 -> i32, i32
  %sw1:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %sw0#0, %sw0#1 : i32 -> i32, i32
  fabric.yield %sw1#1 : i32
}
