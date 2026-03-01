// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_ADD_TAG_VALUE_OVERFLOW

// Tag attribute type is i4 but result tag type is i2 (width mismatch).
fabric.module @test_add_tag_overflow_bad(%a: !dataflow.bits<32>) -> (!dataflow.tagged<!dataflow.bits<32>, i2>) {
  %tagged = fabric.add_tag %a {tag = 7 : i4} : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i2>
  fabric.yield %tagged : !dataflow.tagged<!dataflow.bits<32>, i2>
}
