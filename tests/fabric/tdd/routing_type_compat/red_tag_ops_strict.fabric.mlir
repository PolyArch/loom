// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_ADD_TAG_VALUE_TYPE_MISMATCH

// Verify that tag ops still enforce strict type checking.
// add_tag with i32 input but f32 value type in result: ILLEGAL even though
// i32 and f32 have the same bit width. Tag ops are NOT routing nodes and must
// retain semantic type matching.
fabric.module @test_tag_ops_reject(%a: i32) -> (!dataflow.tagged<f32, i4>) {
  %tagged = fabric.add_tag %a {tag = 5 : i4} : i32 -> !dataflow.tagged<f32, i4>
  fabric.yield %tagged : !dataflow.tagged<f32, i4>
}
