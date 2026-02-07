// RUN: loom --adg %s

// Valid FIFO with matching input/output types.
fabric.module @test_fifo_type_ok(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 4] %a : i32
  fabric.yield %out : i32
}
