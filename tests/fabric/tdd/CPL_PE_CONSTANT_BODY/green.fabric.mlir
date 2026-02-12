// RUN: loom --adg %s

// A valid fabric.pe with arith operations only (no handshake.constant).
// This does not trigger CPL_PE_CONSTANT_BODY because no constant is present.
fabric.module @test(%a: i32, %b: i32) -> (i32) {
  %sum = fabric.pe %a, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : i32
}
