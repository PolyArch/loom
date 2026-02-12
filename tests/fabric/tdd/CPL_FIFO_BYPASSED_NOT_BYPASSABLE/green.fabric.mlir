// RUN: loom --adg %s

// Valid: bypassable fifo with bypassed = false.
fabric.module @test(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 2, bypassable] {bypassed = false} %a : i32
  fabric.yield %out : i32
}
