// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// SSA fan-out: %mul is broadcast both to %add (real consumer) and to a
// fabric.demux side path. Without demux discard mode the demux's input
// wire would never get a consumer (its outputs are unused), causing a
// hardware deadlock. The enumerator must emit configurations that put the
// demux into discard mode so the broadcast value flows correctly.

// CHECK-LABEL: fabric.module @fu_broadcast_with_drain
fabric.module @fu_broadcast_with_drain {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %c = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %mul = fabric.op [@arith.muli] (%x, %y)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      // Side path: %mul is also fed into a demux that drives nowhere.
      // Using SSA fan-out (no extra fabric op).
      %d0, %d1 = fabric.demux %mul : !fabric.bits<32> -> 2
      %add = fabric.op [@arith.addi] (%mul, %z)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %add : !fabric.bits<32>
    }
  }
  fabric.yield
}

// The only valid configurations have the demux in discard mode. The
// sel value does not change the materialized compute (the demux drains
// its input identically), so dedup keeps just the lex-smallest entry.
// CHECK: demux#0{sel=0,discard=true,disconnect=false}
// CHECK-NOT: demux#0{sel=1,discard=true,disconnect=false}

// Configurations that leave the demux in normal mode would deadlock the
// %mul wire and must be dropped.
// CHECK-NOT: demux#0{sel=0,discard=false,disconnect=false}
// CHECK-NOT: demux#0{sel=1,discard=false,disconnect=false}
