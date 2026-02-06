// RUN: loom --adg %s

// A valid fabric.instance referencing a fabric.module (legal target type).
fabric.module @legal_target(%a: i32) -> (i32) {
  %r = fabric.pe %a : (i32) -> (i32) {
  ^bb0(%x: i32):
    %c1 = arith.constant 1 : i32
    %s = arith.addi %x, %c1 : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}

fabric.module @top(%v: i32) -> (i32) {
  %out = fabric.instance @legal_target(%v) : (i32) -> (i32)
  fabric.yield %out : i32
}
