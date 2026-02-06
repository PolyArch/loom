// RUN: loom --adg %s

// A valid acyclic instance chain: @top -> @middle -> @leaf (no cycle).
fabric.module @leaf(%a: i32) -> (i32) {
  %r = fabric.pe %a : (i32) -> (i32) {
  ^bb0(%x: i32):
    %c1 = arith.constant 1 : i32
    %s = arith.addi %x, %c1 : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}

fabric.module @middle(%a: i32) -> (i32) {
  %out = fabric.instance @leaf(%a) : (i32) -> (i32)
  fabric.yield %out : i32
}

fabric.module @top(%v: i32) -> (i32) {
  %out = fabric.instance @middle(%v) : (i32) -> (i32)
  fabric.yield %out : i32
}
