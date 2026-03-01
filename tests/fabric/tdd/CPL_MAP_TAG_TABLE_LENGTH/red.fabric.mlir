// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MAP_TAG_TABLE_LENGTH

// table_size = 3 but table has only 2 entries.
fabric.module @test_map_tag_table_length_bad(%in: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i2>) {
  %out = fabric.map_tag %in {
    table_size = 3,
    table = [
      [1 : i1, 0 : i4, 0 : i2],
      [1 : i1, 1 : i4, 1 : i2]
    ]
  } : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<32>, i2>
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i2>
}
