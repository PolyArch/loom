// RUN: loom --adg %s

// FIFO with a valid tagged type.
fabric.module @test_fifo_tagged(%a: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.fifo [depth = 2] %a : !dataflow.tagged<i32, i4>
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
