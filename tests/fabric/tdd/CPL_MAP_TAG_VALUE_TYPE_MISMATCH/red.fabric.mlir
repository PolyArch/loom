// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MAP_TAG_VALUE_TYPE_MISMATCH

// Input value type is bits<32> but output value type is bits<16>.
fabric.module @test_map_tag_value_type_bad(%in: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<16>, i2>) {
  %out = fabric.map_tag %in {table_size = 4} : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<16>, i2>
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<16>, i2>
}
