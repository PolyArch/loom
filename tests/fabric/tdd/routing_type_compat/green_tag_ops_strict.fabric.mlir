// RUN: loom --adg %s

// Verify that tag ops still accept exact type matching (strict checking).
// add_tag: i32 -> tagged<i32, i4> (value types must match exactly).
// del_tag: tagged<i32, i4> -> i32 (value types must match exactly).
fabric.module @test_tag_ops_strict(%a: i32) -> (i32) {
  %tagged = fabric.add_tag %a {tag = 5 : i4} : i32 -> !dataflow.tagged<i32, i4>
  %result = fabric.del_tag %tagged : !dataflow.tagged<i32, i4> -> i32
  fabric.yield %result : i32
}
