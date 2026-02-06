// RUN: loom --adg %s

// A valid fabric.add_tag where tag value (1) fits in i2 (range 0-3).
fabric.module @test_add_tag_overflow(%a: i32) -> (!dataflow.tagged<i32, i2>) {
  %tagged = fabric.add_tag %a {tag = 3 : i2} : i32 -> !dataflow.tagged<i32, i2>
  fabric.yield %tagged : !dataflow.tagged<i32, i2>
}
