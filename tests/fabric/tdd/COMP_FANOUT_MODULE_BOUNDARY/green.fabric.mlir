// RUN: loom --adg %s

// Module input is duplicated via switch broadcast before feeding
// two consumers. Each connection is strictly 1-to-1.
fabric.pe @add(%a: i32, %b: i32) -> (i32) {
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @top(%x: i32) -> (i32) {
  // Use switch broadcast to duplicate %x for two consumers
  %dup:2 = fabric.switch [connectivity_table = [1, 1]] %x : i32 -> i32, i32
  %sum = fabric.instance @add(%dup#0, %dup#1) : (i32, i32) -> (i32)
  fabric.yield %sum : i32
}
