// RUN: loom --adg %s

// Valid: bypassable fifo with bypassed = true.
fabric.module @test(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 4, bypassable] {bypassed = true} %a : i32
  fabric.yield %out : i32
}
