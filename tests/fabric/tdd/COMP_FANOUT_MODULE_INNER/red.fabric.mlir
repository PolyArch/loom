// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_FANOUT_MODULE_INNER

// An instance output is used by two consumers (switch + yield) without
// switch broadcast. This must be rejected.
fabric.pe @add(%a: i32, %b: i32) -> (i32) {
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @top(%x: i32, %y: i32) -> (i32, i32) {
  %sum = fabric.instance @add(%x, %y) : (i32, i32) -> (i32)
  // %sum used twice: feeds both yield operands
  fabric.yield %sum, %sum : i32, i32
}
