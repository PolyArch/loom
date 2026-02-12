// RUN: loom --adg %s

// A valid fabric.module with a non-terminator operation in the body.
fabric.module @nonempty(%a: i32, %b: i32) -> (i32) {
  %sum = fabric.pe %a, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : i32
}
