// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_TAG_WIDTH_RANGE

// Tag type i32 has width 32, which exceeds the allowed range [1, 16].
// The parser rejects the type before the verifier runs.
fabric.module @test_tag_width_bad(%a: i32) -> (!dataflow.tagged<i32, i32>) {
  %tagged = fabric.add_tag %a {tag = 0 : i32} : i32 -> !dataflow.tagged<i32, i32>
  fabric.yield %tagged : !dataflow.tagged<i32, i32>
}
