// RUN: loom --adg %s

// Valid: non-bypassable fifo without bypassed.
fabric.module @test(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 2] %a : i32
  fabric.yield %out : i32
}
