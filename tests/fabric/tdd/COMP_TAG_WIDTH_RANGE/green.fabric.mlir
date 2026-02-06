// RUN: loom --adg %s

// A valid fabric.add_tag with tag type i4 (width 4 is within [1, 16]).
fabric.module @test_tag_width(%a: i32) -> (!dataflow.tagged<i32, i4>) {
  %tagged = fabric.add_tag %a {tag = 0 : i4} : i32 -> !dataflow.tagged<i32, i4>
  fabric.yield %tagged : !dataflow.tagged<i32, i4>
}
