// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// FU contains an op outside the v2 enumerator's allowlist
// (dataflow.constant has no materializer yet). The pass should emit a
// warning and not produce any subgraphs.

// CHECK: warning: fabric.fu enumeration skipped: contains unsupported op 'dataflow.constant'

// CHECK-LABEL: @fu_unsupported
func.func @fu_unsupported(%ctrl: !fabric.bits<0>) {
  %r = fabric.fu(%c = %ctrl : !fabric.bits<0>) -> !fabric.bits<32> {
    %k = fabric.op [@dataflow.constant] (%c)
         : (!fabric.bits<0>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// CHECK-NOT: dataflow.subgraph
