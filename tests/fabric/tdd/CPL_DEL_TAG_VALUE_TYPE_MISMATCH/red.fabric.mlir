// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_DEL_TAG_VALUE_TYPE_MISMATCH

// Input value type is bits<32> but output type is bits<16>.
fabric.module @test_del_tag_value_type_bad(%in: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.bits<16>) {
  %val = fabric.del_tag %in : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<16>
  fabric.yield %val : !dataflow.bits<16>
}
