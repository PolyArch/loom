// RUN: loom --adg %s

// A valid fabric.map_tag with table length matching table_size.
fabric.module @test_map_tag_table_length(%in: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i2>) {
  %out = fabric.map_tag %in {
    table_size = 2,
    table = [
      [1 : i1, 0 : i4, 0 : i2],
      [1 : i1, 1 : i4, 1 : i2]
    ]
  } : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<32>, i2>
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i2>
}
