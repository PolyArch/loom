// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_CONSTANT_BODY

// A constant PE that also contains an arith op, violating constant exclusivity.
fabric.module @test(%ctrl: none, %x: i32) -> (i32) {
  %r = fabric.pe %ctrl, %x : (none, i32) -> (i32) {
  ^bb0(%c_ctrl: none, %c_x: i32):
    %c = handshake.constant %c_ctrl {value = 10 : i32} : i32
    %s = arith.addi %c, %c_x : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}
