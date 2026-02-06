// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MAP_TAG_VALUE_TYPE_MISMATCH

// Input value type is i32 but output value type is f32.
fabric.module @test_map_tag_value_type_bad(%in: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<f32, i2>) {
  %out = fabric.map_tag %in {table_size = 4} : !dataflow.tagged<i32, i4> -> !dataflow.tagged<f32, i2>
  fabric.yield %out : !dataflow.tagged<f32, i2>
}
