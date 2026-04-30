// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// FU contains a variadic dataflow.sync whose hardware port count M
// exceeds the enumerator's hard cap (kVariadicMaxM = 8). The
// enumerator should emit a warning and skip the FU rather than try to
// iterate 2^M-1 bitmasks.

// CHECK: warning: variadic fabric.op port count M=9 exceeds the enumerator's hard cap (8)
// CHECK: warning: fabric.fu enumeration skipped: contains unsupported op 'dataflow.sync'

// CHECK-LABEL: fabric.module @fu_unsupported
fabric.module @fu_unsupported(%a0 : !fabric.bits<32>, %a1 : !fabric.bits<32>, %a2 : !fabric.bits<32>, %a3 : !fabric.bits<32>, %a4 : !fabric.bits<32>, %a5 : !fabric.bits<32>, %a6 : !fabric.bits<32>, %a7 : !fabric.bits<32>, %a8 : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa0 = %a0 : !fabric.bits<32>,
                    %pa1 = %a1 : !fabric.bits<32>,
                    %pa2 = %a2 : !fabric.bits<32>,
                    %pa3 = %a3 : !fabric.bits<32>,
                    %pa4 = %a4 : !fabric.bits<32>,
                    %pa5 = %a5 : !fabric.bits<32>,
                    %pa6 = %a6 : !fabric.bits<32>,
                    %pa7 = %a7 : !fabric.bits<32>,
                    %pa8 = %a8 : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                 !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                 !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%p0 = %pa0 : !fabric.bits<32>, %p1 = %pa1 : !fabric.bits<32>,
              %p2 = %pa2 : !fabric.bits<32>, %p3 = %pa3 : !fabric.bits<32>,
              %p4 = %pa4 : !fabric.bits<32>, %p5 = %pa5 : !fabric.bits<32>,
              %p6 = %pa6 : !fabric.bits<32>, %p7 = %pa7 : !fabric.bits<32>,
              %p8 = %pa8 : !fabric.bits<32>)
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
  }
  fabric.yield
}

// CHECK-NOT: dataflow.subgraph
