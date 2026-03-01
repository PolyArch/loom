// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: bits-interface PE body argument #0 has width 16

// PE with tagged<bits<32>,i4> interface port but i16 body arg (width mismatch).
fabric.pe @bad_tagged_body(%in: !dataflow.tagged<!dataflow.bits<32>, i4>)
    -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
^bb0(%x: i16):
  %c = arith.constant 1 : i16
  %r = arith.addi %x, %c : i16
  fabric.yield %r : i16
}
