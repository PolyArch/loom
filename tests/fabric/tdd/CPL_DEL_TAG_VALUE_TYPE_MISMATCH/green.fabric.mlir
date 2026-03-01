// RUN: loom --adg %s

// A valid fabric.del_tag where output type matches input value type (both i32).
fabric.module @test_del_tag_value_type(%in: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.bits<32>) {
  %val = fabric.del_tag %in : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
  fabric.yield %val : !dataflow.bits<32>
}
