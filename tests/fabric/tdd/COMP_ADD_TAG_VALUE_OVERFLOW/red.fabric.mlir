// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_ADD_TAG_VALUE_OVERFLOW

// Tag attribute type is i4 but result tag type is i2 (width mismatch).
fabric.module @test_add_tag_overflow_bad(%a: i32) -> (!dataflow.tagged<i32, i2>) {
  %tagged = fabric.add_tag %a {tag = 7 : i4} : i32 -> !dataflow.tagged<i32, i2>
  fabric.yield %tagged : !dataflow.tagged<i32, i2>
}
