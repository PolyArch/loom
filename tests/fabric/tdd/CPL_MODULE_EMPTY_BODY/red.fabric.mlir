// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MODULE_EMPTY_BODY

// A fabric.module whose body contains only the terminator (no non-terminator ops).
fabric.module @empty(%in: i32) -> (i32) {
  fabric.yield %in : i32
}
