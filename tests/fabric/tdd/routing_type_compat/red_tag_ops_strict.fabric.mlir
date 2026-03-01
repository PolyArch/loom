// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_ADD_TAG_VALUE_TYPE_MISMATCH

// Verify that tag ops still enforce strict type checking.
// add_tag with bits<32> input but bits<16> value type in result: ILLEGAL
// because the input value type must exactly match the result value type.
fabric.module @test_tag_ops_reject(%a: !dataflow.bits<32>) -> (!dataflow.tagged<!dataflow.bits<16>, i4>) {
  %tagged = fabric.add_tag %a {tag = 5 : i4} : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<16>, i4>
  fabric.yield %tagged : !dataflow.tagged<!dataflow.bits<16>, i4>
}
