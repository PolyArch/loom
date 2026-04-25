// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// FU contains an op outside the v1 enumerator's allowlist. The pass should
// emit a warning and not produce any subgraphs.

// CHECK: warning: fabric.fu enumeration skipped: contains unsupported op 'arith.addf'

// CHECK-LABEL: @fu_unsupported
func.func @fu_unsupported(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addf] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// CHECK-NOT: dataflow.subgraph
