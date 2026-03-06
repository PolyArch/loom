// RUN: loom --adg %s

// A valid fabric.map_tag where input and output value types both are i32.
fabric.module @test_map_tag_value_type(%in: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i2>) {
  %out = fabric.map_tag %in {table_size = 4} : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<32>, i2>
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i2>
}
