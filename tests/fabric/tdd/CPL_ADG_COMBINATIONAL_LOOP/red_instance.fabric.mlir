// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_ADG_COMBINATIONAL_LOOP

// Named switches form a combinational cycle via fabric.instance.
fabric.switch @xbar [connectivity_table = [1, 1, 1, 1]] : (i32, i32) -> (i32, i32)

fabric.module @test(%a: i32) -> (i32) {
  %sw0:2 = fabric.instance @xbar(%a, %sw1#0) : (i32, i32) -> (i32, i32)
  %sw1:2 = fabric.instance @xbar(%sw0#0, %sw0#1) : (i32, i32) -> (i32, i32)
  fabric.yield %sw1#1 : i32
}
