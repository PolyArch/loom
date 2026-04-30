// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins: variadic dataflow.demux with M=3 data outputs. The bitmask sw
// config picks the live output subset; sel is i1 for N=2 active outputs
// and index for N>=3 (per dataflow.demux verifier).

// CHECK-LABEL: @fu_demux3
fabric.module @fu_demux3(%cast0_fu_demux3 : !fabric.bits<32>, %cast1_fu_demux3 : !fabric.bits<32>) {
  fabric.pe [spatial] (%s = %cast0_fu_demux3 : !fabric.bits<32>, %d = %cast1_fu_demux3 : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
  %a, %b, %c = fabric.fu(%ss = %s : !fabric.bits<32>,
                         %dd = %d : !fabric.bits<32>)
                        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    %o0, %o1, %o2 = fabric.op [@dataflow.demux] (%ss, %dd)
                    : (!fabric.bits<32>, !fabric.bits<32>)
                      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    fabric.yield %o0, %o1, %o2 : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
  }
  // CHECK-DAG: dataflow.demux
  }
  fabric.yield
}

