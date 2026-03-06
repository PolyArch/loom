// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MODULE_EMPTY_BODY

// A fabric.module whose body contains only the terminator (no non-terminator ops).
fabric.module @empty(%in: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  fabric.yield %in : !dataflow.bits<32>
}
