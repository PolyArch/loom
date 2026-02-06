// RUN: loom --adg %s

// A valid fabric.add_tag where input type matches result value type (both i32).
fabric.module @test_add_tag_value_type(%a: i32) -> (!dataflow.tagged<i32, i4>) {
  %tagged = fabric.add_tag %a {tag = 5 : i4} : i32 -> !dataflow.tagged<i32, i4>
  fabric.yield %tagged : !dataflow.tagged<i32, i4>
}
