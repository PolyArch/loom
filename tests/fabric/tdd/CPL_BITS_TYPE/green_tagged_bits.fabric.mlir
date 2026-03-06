// RUN: loom --adg %s

// Tagged type with bits value: tagged<bits<32>, i4> is valid.
fabric.module @test_tagged_bits(
    %a: !dataflow.tagged<!dataflow.bits<32>, i4>
) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %out = fabric.fifo [depth = 2] %a : !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i4>
}
