// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_INSTANCE_UNRESOLVED

// fabric.instance references a symbol that does not exist.
fabric.module @top(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @nonexistent(%a) : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}
