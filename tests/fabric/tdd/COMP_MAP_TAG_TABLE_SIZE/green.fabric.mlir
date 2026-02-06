// RUN: loom --adg %s

// A valid fabric.map_tag with table_size in [1, 256].
fabric.module @test_map_tag_table_size(%in: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i2>) {
  %out = fabric.map_tag %in {table_size = 4} : !dataflow.tagged<i32, i4> -> !dataflow.tagged<i32, i2>
  fabric.yield %out : !dataflow.tagged<i32, i2>
}
