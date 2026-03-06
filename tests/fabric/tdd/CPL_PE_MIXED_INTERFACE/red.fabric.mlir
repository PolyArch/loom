// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_MIXED_INTERFACE

// A fabric.pe mixing native (i32) and tagged ports.
fabric.module @test(%a: !dataflow.bits<32>, %b: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a, %b : (!dataflow.bits<32>, !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32):
    %s = arith.addi %x, %y : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}
