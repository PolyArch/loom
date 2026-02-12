// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_MIXED_INTERFACE

// A fabric.pe mixing native (i32) and tagged ports.
fabric.module @test(%a: i32, %b: !dataflow.tagged<i32, i4>) -> (i32) {
  %r = fabric.pe %a, %b : (i32, !dataflow.tagged<i32, i4>) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %s = arith.addi %x, %y : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}
