// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// SSA fan-out: %mul is broadcast both to %add (real consumer) and to a
// fabric.demux side path. Without demux discard mode the demux's input
// wire would never get a consumer (its outputs are unused), causing a
// hardware deadlock. The enumerator must emit configurations that put the
// demux into discard mode so the broadcast value flows correctly.

// CHECK-LABEL: @fu_broadcast_with_drain
func.func @fu_broadcast_with_drain(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                                    %c: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>,
                 %z = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    %mul = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    // Side path: %mul is also fed into a demux that drives nowhere.
    // Using SSA fan-out (no extra fabric op).
    %d0, %d1 = fabric.demux %mul : !fabric.bits<32> -> 2
    %add = fabric.op [@arith.addi] (%mul, %z)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %add : !fabric.bits<32>
  }

  // The only valid configurations have the demux in discard mode (one
  // entry per sel value since the drain works the same way regardless).
  // CHECK-DAG: demux#0{sel=0,discard=true,disconnect=false}
  // CHECK-DAG: demux#0{sel=1,discard=true,disconnect=false}

  // Configurations that leave the demux in normal mode would deadlock the
  // %mul wire and must be dropped.
  // CHECK-NOT: demux#0{sel=0,discard=false,disconnect=false}
  // CHECK-NOT: demux#0{sel=1,discard=false,disconnect=false}

  return
}
