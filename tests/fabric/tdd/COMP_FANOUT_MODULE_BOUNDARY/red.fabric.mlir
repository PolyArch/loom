// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_FANOUT_MODULE_BOUNDARY

// A module input argument feeds two instance input ports without
// switch broadcast. This must be rejected.
fabric.pe @add(%a: i32, %b: i32) -> (i32) {
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @top(%x: i32) -> (i32) {
  // %x used twice: feeds both inputs of @add
  %sum = fabric.instance @add(%x, %x) : (i32, i32) -> (i32)
  fabric.yield %sum : i32
}
