// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MAP_TAG_TABLE_SIZE

// table_size = 0 is out of the valid range [1, 256].
fabric.module @test_map_tag_table_size_bad(%in: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i2>) {
  %out = fabric.map_tag %in {table_size = 0} : !dataflow.tagged<i32, i4> -> !dataflow.tagged<i32, i2>
  fabric.yield %out : !dataflow.tagged<i32, i2>
}
