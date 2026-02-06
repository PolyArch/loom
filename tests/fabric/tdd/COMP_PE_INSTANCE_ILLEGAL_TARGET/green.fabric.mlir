// RUN: loom --adg %s

// A valid fabric.pe body that instances a named fabric.pe (legal target).
fabric.pe @adder(%a: i32, %b: i32) -> (i32)
    [latency = [1 : i16, 1 : i16, 1 : i16]] {
  %s = arith.addi %a, %b : i32
  fabric.yield %s : i32
}

fabric.module @top(%a: i32, %b: i32, %c: i32) -> (i32) {
  %r = fabric.pe %a, %b, %c : (i32, i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32, %z: i32):
    %out = fabric.instance @adder(%x, %y) : (i32, i32) -> (i32)
    %s = arith.addi %out, %z : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}
