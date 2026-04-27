// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// FU contains a variadic dataflow.sync whose hardware port count M
// exceeds the enumerator's hard cap (kVariadicMaxM = 8). The
// enumerator should emit a warning and skip the FU rather than try to
// iterate 2^M-1 bitmasks.

// CHECK: warning: variadic fabric.op port count M=9 exceeds the enumerator's hard cap (8)
// CHECK: warning: fabric.fu enumeration skipped: contains unsupported op 'dataflow.sync'

// CHECK-LABEL: @fu_unsupported
func.func @fu_unsupported(%a0: !fabric.bits<32>, %a1: !fabric.bits<32>,
                          %a2: !fabric.bits<32>, %a3: !fabric.bits<32>,
                          %a4: !fabric.bits<32>, %a5: !fabric.bits<32>,
                          %a6: !fabric.bits<32>, %a7: !fabric.bits<32>,
                          %a8: !fabric.bits<32>) {
  %r:9 = fabric.fu(%p0 = %a0 : !fabric.bits<32>, %p1 = %a1 : !fabric.bits<32>,
                   %p2 = %a2 : !fabric.bits<32>, %p3 = %a3 : !fabric.bits<32>,
                   %p4 = %a4 : !fabric.bits<32>, %p5 = %a5 : !fabric.bits<32>,
                   %p6 = %a6 : !fabric.bits<32>, %p7 = %a7 : !fabric.bits<32>,
                   %p8 = %a8 : !fabric.bits<32>)
                  -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                      !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                      !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    %s:9 = fabric.op [@dataflow.sync]
           (%p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7, %p8)
           : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
              !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
              !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                 !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                 !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    fabric.yield %s#0, %s#1, %s#2, %s#3, %s#4, %s#5, %s#6, %s#7, %s#8
                 : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                   !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                   !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
  }
  return
}

// CHECK-NOT: dataflow.subgraph
