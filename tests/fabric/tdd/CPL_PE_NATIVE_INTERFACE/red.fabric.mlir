// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_NATIVE_INTERFACE

// Module-level PE with native i32 interface (must use bits<32>).
fabric.module @bad_native_pe(%in: i32) -> (i32) {
  %r = fabric.pe %in : (i32) -> (i32) {
  ^bb0(%x: i32):
    fabric.yield %x : i32
  }
  fabric.yield %r : i32
}
