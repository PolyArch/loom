// RUN: loom --adg %s

// Named FIFO with bit-width-compatible native types: i32 in, f32 out.
fabric.fifo @relaxed_buf [depth = 4] : (i32) -> (f32)

fabric.module @test_fifo_named_compat(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 4] %a : i32
  fabric.yield %out : i32
}
