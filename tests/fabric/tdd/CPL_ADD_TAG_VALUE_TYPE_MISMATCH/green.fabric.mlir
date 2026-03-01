// RUN: loom --adg %s

// A valid fabric.add_tag where input type matches result value type (both i32).
fabric.module @test_add_tag_value_type(%a: !dataflow.bits<32>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %tagged = fabric.add_tag %a {tag = 5 : i4} : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %tagged : !dataflow.tagged<!dataflow.bits<32>, i4>
}
