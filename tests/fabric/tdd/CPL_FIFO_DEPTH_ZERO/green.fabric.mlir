// RUN: loom --adg %s

// FIFO with depth >= 1 is valid.
fabric.module @test_fifo_depth_ok(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 1] %a : i32
  fabric.yield %out : i32
}
