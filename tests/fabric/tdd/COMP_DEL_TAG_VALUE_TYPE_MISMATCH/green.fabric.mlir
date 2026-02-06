// RUN: loom --adg %s

// A valid fabric.del_tag where output type matches input value type (both i32).
fabric.module @test_del_tag_value_type(%in: !dataflow.tagged<i32, i4>) -> (i32) {
  %val = fabric.del_tag %in : !dataflow.tagged<i32, i4> -> i32
  fabric.yield %val : i32
}
