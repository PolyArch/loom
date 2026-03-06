// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_INSTANCE_OPERAND_MISMATCH

// Named fifo expects 1 input, but instance provides 2.
fabric.fifo @buf [depth = 2] : (!dataflow.bits<32>) -> (!dataflow.bits<32>)

fabric.module @top(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @buf(%a, %b) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}
