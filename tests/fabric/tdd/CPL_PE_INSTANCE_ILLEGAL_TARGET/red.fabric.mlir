// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_INSTANCE_ILLEGAL_TARGET

// A fabric.pe body that instances a fabric.module (illegal inside PE).
fabric.module @inner(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a, %b : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32):
    %s = arith.addi %x, %y : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}

fabric.module @top(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>, %c: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a, %b, %c : (!dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32, %z: i32):
    %out = fabric.instance @inner(%x, %y) : (i32, i32) -> (i32)
    %s = arith.addi %out, %z : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}
