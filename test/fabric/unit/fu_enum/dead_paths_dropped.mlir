// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// FU with a fabric.demux fanning out two outputs. Only the demux.sel=0
// configuration keeps %d0 alive; demux.sel=1 leaves the yielded value dead
// and the candidate is dropped.

// CHECK-LABEL: fabric.module @fu_demux_drop_dead_yield
fabric.module @fu_demux_drop_dead_yield(%a : !fabric.bits<8>, %b : !fabric.bits<8>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%x = %pa : !fabric.bits<8>, %y = %pb : !fabric.bits<8>)
                  -> !fabric.bits<8> {
      %k = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      %d0, %d1 = fabric.demux %k : !fabric.bits<8> -> 2
      fabric.yield %d0 : !fabric.bits<8>
    }
  }
  fabric.yield
}

// Only one config (demux.sel=0) keeps %d0 alive.
// CHECK: dataflow.subgraph
// CHECK-SAME: demux#0{sel=0,discard=false,disconnect=false}
// CHECK:   arith.muli
// CHECK:   dataflow.yield

// The demux.sel=1 case must not produce a candidate.
// CHECK-NOT: demux#0{sel=1,discard=false,disconnect=false}
