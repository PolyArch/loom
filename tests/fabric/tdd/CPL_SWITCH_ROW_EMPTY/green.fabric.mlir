// RUN: loom --adg %s

// A valid 2x2 switch where every output row has at least one connection.
fabric.module @test(%a: i32, %b: i32) -> (i32, i32) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 0, 0, 1]] %a, %b : i32 -> i32, i32
  fabric.yield %o1, %o2 : i32, i32
}
