// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// FU with a fabric.demux fanning out two outputs, but only one is consumed.
// The other demux branch is unconnected - it would need to feed something or
// the config is simply "use that demux output". Here it's the yield value
// itself that may be dead. Demux with demux.sel=0 makes %d0 alive (ok),
// while demux.sel=1 makes %d0 dead -> yield drops the candidate.

// CHECK-LABEL: @fu_demux_drop_dead_yield
func.func @fu_demux_drop_dead_yield(%a: !fabric.bits<8>, %b: !fabric.bits<8>) {
  %r = fabric.fu(%x = %a : !fabric.bits<8>, %y = %b : !fabric.bits<8>)
                -> !fabric.bits<8> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
    %d0, %d1 = fabric.demux %k : !fabric.bits<8> -> 2
    fabric.yield %d0 : !fabric.bits<8>
  }

  // Only one config makes %d0 alive (demux.sel=0).
  // CHECK: dataflow.subgraph
  // CHECK-SAME: "op#0=arith.muli; demux#0.sel=0"
  // CHECK:   arith.muli
  // CHECK:   dataflow.yield

  // The demux.sel=1 case must NOT appear.
  // CHECK-NOT: demux#0.sel=1

  return
}
