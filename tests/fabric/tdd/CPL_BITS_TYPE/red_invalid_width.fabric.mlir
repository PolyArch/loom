// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: bits width must be 1-4096

// bits<0> is invalid: width must be >= 1.
fabric.module @test_bits_zero(%a: !dataflow.bits<0>) -> (!dataflow.bits<0>) {
  fabric.yield %a : !dataflow.bits<0>
}
