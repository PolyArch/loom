// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_PE_OUTPUT_TAG_MISSING

// A tagged non-load/store fabric.pe missing the required output_tag attribute.
fabric.module @test(
    %a: !dataflow.tagged<i32, i4>,
    %b: !dataflow.tagged<i32, i4>
) -> (!dataflow.tagged<i32, i4>) {
  %r = fabric.pe %a, %b
      : (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>)
      -> (!dataflow.tagged<i32, i4>) {
  ^bb0(%x: i32, %y: i32):
    %s = arith.addi %x, %y : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : !dataflow.tagged<i32, i4>
}
