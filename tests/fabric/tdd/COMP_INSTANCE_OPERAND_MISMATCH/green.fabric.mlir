// RUN: loom --adg %s

// A valid fabric.instance whose operand count and types match the target.
fabric.module @alu(%a: i32, %b: i32) -> (i32) {
  %sum = fabric.pe %a, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : i32
}

fabric.module @top(%p: i32, %q: i32) -> (i32) {
  %out = fabric.instance @alu(%p, %q) : (i32, i32) -> (i32)
  fabric.yield %out : i32
}
