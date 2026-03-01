// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_MIXED_CONSUMPTION

// A PE mixing full-consume (arith.addi) with partial-consume (handshake.mux).
fabric.module @test(%sel: !dataflow.bits<1>, %a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %sel, %a, %b : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%bsel: i1, %ba: i32, %bb: i32):
    %sum = arith.addi %ba, %bb : i32
    %m = handshake.mux %bsel [%sum, %bb] : i1, i32
    fabric.yield %m : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}
