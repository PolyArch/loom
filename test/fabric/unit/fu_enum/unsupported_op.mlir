// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// FU contains an op outside the enumerator's allowlist (variadic
// dataflow.sync has no materializer yet). The pass should emit a warning
// and not produce any subgraphs.

// CHECK: warning: fabric.fu enumeration skipped: contains unsupported op 'dataflow.sync'

// CHECK-LABEL: @fu_unsupported
func.func @fu_unsupported(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %x, %y = fabric.fu(%p = %a : !fabric.bits<32>, %q = %b : !fabric.bits<32>)
                    -> (!fabric.bits<32>, !fabric.bits<32>) {
    %u, %v = fabric.op [@dataflow.sync] (%p, %q)
             : (!fabric.bits<32>, !fabric.bits<32>)
               -> (!fabric.bits<32>, !fabric.bits<32>)
    fabric.yield %u, %v : !fabric.bits<32>, !fabric.bits<32>
  }
  return
}

// CHECK-NOT: dataflow.subgraph
