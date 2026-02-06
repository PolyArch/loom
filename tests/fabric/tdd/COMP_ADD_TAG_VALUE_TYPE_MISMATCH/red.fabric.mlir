// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_ADD_TAG_VALUE_TYPE_MISMATCH

// Input type is i32 but result value type is f32.
fabric.module @test_add_tag_value_type_bad(%a: i32) -> (!dataflow.tagged<f32, i4>) {
  %tagged = fabric.add_tag %a {tag = 5 : i4} : i32 -> !dataflow.tagged<f32, i4>
  fabric.yield %tagged : !dataflow.tagged<f32, i4>
}
