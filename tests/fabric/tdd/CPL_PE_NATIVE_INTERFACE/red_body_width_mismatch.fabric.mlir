// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: bits-interface PE body argument #0 has width 64

// PE with bits<32> interface port but i64 body arg (width mismatch).
fabric.pe @bad_body_width(%in: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
^bb0(%x: i64):
  %c = arith.constant 1 : i64
  %r = arith.addi %x, %c : i64
  fabric.yield %r : i64
}
