// RUN: loom --adg %s

// A valid fabric.instance whose result count and types match the target.
fabric.module @inc(%a: i32) -> (i32) {
  %r = fabric.pe %a : (i32) -> (i32) {
  ^bb0(%x: i32):
    %c1 = arith.constant 1 : i32
    %s = arith.addi %x, %c1 : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}

fabric.module @top(%v: i32) -> (i32) {
  %out = fabric.instance @inc(%v) : (i32) -> (i32)
  fabric.yield %out : i32
}
