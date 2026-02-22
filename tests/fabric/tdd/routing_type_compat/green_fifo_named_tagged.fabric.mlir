// RUN: loom --adg %s

// Named FIFO with bit-width-compatible tagged types:
// tagged<i32,i4> in, tagged<f32,i4> out.
fabric.fifo @relaxed_tagged_buf [depth = 4] : (!dataflow.tagged<i32, i4>) -> (!dataflow.tagged<f32, i4>)

fabric.module @test(%a: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.fifo [depth = 4] %a : !dataflow.tagged<i32, i4>
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
