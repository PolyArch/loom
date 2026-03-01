// RUN: loom --adg %s

// A valid fabric.add_tag with tag type i4 (width 4 is within [1, 16]).
fabric.module @test_tag_width(%a: !dataflow.bits<32>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %tagged = fabric.add_tag %a {tag = 0 : i4} : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %tagged : !dataflow.tagged<!dataflow.bits<32>, i4>
}
