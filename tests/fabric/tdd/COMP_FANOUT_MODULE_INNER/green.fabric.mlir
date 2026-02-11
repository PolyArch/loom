// RUN: loom --adg %s

// Each instance output has at most one consumer, thanks to switch broadcast.
fabric.pe @add(%a: i32, %b: i32) -> (i32) {
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @top(%x: i32, %y: i32) -> (i32, i32) {
  %sum = fabric.instance @add(%x, %y) : (i32, i32) -> (i32)
  // Use switch broadcast to duplicate %sum for two yield operands
  %dup:2 = fabric.switch [connectivity_table = [1, 1]] %sum : i32 -> i32, i32
  fabric.yield %dup#0, %dup#1 : i32, i32
}
