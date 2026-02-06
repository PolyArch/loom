// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_PE_OUTPUT_TAG_NATIVE

// A native fabric.pe that incorrectly has an output_tag attribute.
fabric.module @test(%a: i32, %b: i32) -> (i32) {
  %r = fabric.pe %a, %b
      [output_tag = [0 : i4]]
      : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %s = arith.addi %x, %y : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}
