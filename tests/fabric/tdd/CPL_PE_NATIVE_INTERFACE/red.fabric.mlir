// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_NATIVE_INTERFACE

// Top-level named PE with native i32 interface (must use bits<32>).
fabric.pe @bad_native(%arg0: i32, %arg1: i32) -> (i32) {
  %r = arith.addi %arg0, %arg1 : i32
  fabric.yield %r : i32
}

fabric.module @test(%x: !dataflow.bits<32>, %y: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.instance @bad_native(%x, %y) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %r : !dataflow.bits<32>
}
