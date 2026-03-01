// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_EMPTY_BODY

// A fabric.pe whose body contains only the terminator (no non-terminator ops).
fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a : (!dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32):
    fabric.yield %x : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}
