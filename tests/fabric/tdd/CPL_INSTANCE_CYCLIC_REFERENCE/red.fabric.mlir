// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_INSTANCE_CYCLIC_REFERENCE

// @mod_a instantiates @mod_b and @mod_b instantiates @mod_a, forming a cycle.
fabric.module @mod_a(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @mod_b(%a) : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}

fabric.module @mod_b(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @mod_a(%a) : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}
