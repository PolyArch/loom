// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MODULE_NATIVE_PORT

// Module with native i32 ports (must use bits<N>).
fabric.module @bad_native_port(%x: i32) -> (i32) {
  %r = fabric.pe %x : (i32) -> (i32) {
  ^bb0(%a: i32):
    fabric.yield %a : i32
  }
  fabric.yield %r : i32
}
