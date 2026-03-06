// RUN: loom --adg %s

// A valid fabric.add_tag where tag value (1) fits in i2 (range 0-3).
fabric.module @test_add_tag_overflow(%a: !dataflow.bits<32>) -> (!dataflow.tagged<!dataflow.bits<32>, i2>) {
  %tagged = fabric.add_tag %a {tag = 3 : i2} : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i2>
  fabric.yield %tagged : !dataflow.tagged<!dataflow.bits<32>, i2>
}
