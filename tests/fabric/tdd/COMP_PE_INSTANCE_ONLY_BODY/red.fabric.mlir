// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_PE_INSTANCE_ONLY_BODY

// A fabric.pe whose body is solely a single fabric.instance with no other ops.
fabric.pe @target(%a: i32, %b: i32) -> (i32)
    [latency = [1 : i16, 1 : i16, 1 : i16]] {
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @test(%a: i32, %b: i32) -> (i32) {
  %r = fabric.pe %a, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %t = fabric.instance @target(%x, %y) : (i32, i32) -> (i32)
    fabric.yield %t : i32
  }
  fabric.yield %r : i32
}
