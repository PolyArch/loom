// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_DEL_TAG_VALUE_TYPE_MISMATCH

// Input value type is i32 but output type is f32.
fabric.module @test_del_tag_value_type_bad(%in: !dataflow.tagged<i32, i4>) -> (f32) {
  %val = fabric.del_tag %in : !dataflow.tagged<i32, i4> -> f32
  fabric.yield %val : f32
}
