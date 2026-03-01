// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_ADD_TAG_VALUE_TYPE_MISMATCH

// Input type is bits<32> but result value type is bits<64> -- width mismatch.
fabric.module @test_add_tag_value_type_bad(%a: !dataflow.bits<32>) -> (!dataflow.tagged<!dataflow.bits<64>, i4>) {
  %tagged = fabric.add_tag %a {tag = 5 : i4} : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<64>, i4>
  fabric.yield %tagged : !dataflow.tagged<!dataflow.bits<64>, i4>
}
