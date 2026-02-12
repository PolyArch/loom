// RUN: loom --adg %s

// A valid fabric.instance referencing an existing fabric.module symbol.
fabric.module @adder(%a: i32, %b: i32) -> (i32) {
  %sum = fabric.pe %a, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : i32
}

fabric.module @top(%p: i32, %q: i32) -> (i32) {
  %out = fabric.instance @adder(%p, %q) : (i32, i32) -> (i32)
  fabric.yield %out : i32
}
