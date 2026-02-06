// RUN: loom --adg %s

// A valid 2x2 switch with connectivity_table of correct length (2*2 = 4).
fabric.module @test(%a: i32, %b: i32) -> (i32, i32) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 0, 0, 1]] %a, %b : i32 -> i32, i32
  fabric.yield %o1, %o2 : i32, i32
}
