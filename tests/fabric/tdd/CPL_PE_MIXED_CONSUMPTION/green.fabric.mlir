// RUN: loom --adg %s

// A valid PE using only full-consume operations (arith).
// This does not trigger CPL_PE_MIXED_CONSUMPTION because all ops are from
// the same consumption group.
fabric.module @test(%a: i32, %b: i32) -> (i32) {
  %sum = fabric.pe %a, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : i32
}
