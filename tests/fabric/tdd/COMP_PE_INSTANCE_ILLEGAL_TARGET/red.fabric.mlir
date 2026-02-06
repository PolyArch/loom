// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_PE_INSTANCE_ILLEGAL_TARGET

// A fabric.pe body that instances a fabric.module (illegal inside PE).
fabric.module @inner(%a: i32, %b: i32) -> (i32) {
  %r = fabric.pe %a, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %s = arith.addi %x, %y : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}

fabric.module @top(%a: i32, %b: i32, %c: i32) -> (i32) {
  %r = fabric.pe %a, %b, %c : (i32, i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32, %z: i32):
    %out = fabric.instance @inner(%x, %y) : (i32, i32) -> (i32)
    %s = arith.addi %out, %z : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}
