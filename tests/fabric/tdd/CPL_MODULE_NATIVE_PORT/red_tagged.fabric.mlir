// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MODULE_NATIVE_PORT

// Module with tagged port using native value type (must use tagged<bits, iK>).
fabric.module @bad_tagged_native(%x: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %r = fabric.pe %x : (!dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  ^bb0(%a: i32):
    fabric.yield %a : i32
  }
  fabric.yield %r : !dataflow.tagged<i32, i4>
}
