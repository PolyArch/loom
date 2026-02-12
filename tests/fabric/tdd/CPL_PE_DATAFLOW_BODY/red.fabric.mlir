// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_DATAFLOW_BODY

// A dataflow PE that also contains an arith op, violating dataflow exclusivity.
fabric.module @test(%d: i1, %a: i32, %b: i32) -> (i32) {
  %r = fabric.pe %d, %a, %b : (i1, i32, i32) -> (i32) {
  ^bb0(%bd: i1, %ba: i32, %bb: i32):
    %inv = dataflow.invariant %bd, %ba : i1, i32 -> i32
    %s = arith.addi %inv, %bb : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}
