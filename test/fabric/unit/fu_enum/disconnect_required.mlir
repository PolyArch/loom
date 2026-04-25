// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU has a fabric.demux whose input source has no other live consumer in
// some configurations. Those configurations need the demux in disconnect
// mode (or discard); normal mode would force the input source to fire when
// nothing downstream consumes its outputs.

// CHECK-LABEL: @fu_unused_side
func.func @fu_unused_side(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %mul = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %d0, %d1 = fabric.demux %mul : !fabric.bits<32> -> 2
    fabric.yield %mul : !fabric.bits<32>
  }

  // The enumerator emits a configuration where the demux drains (discard
  // mode) so the broadcast wire to the demux completes its handshake.
  // CHECK-DAG: demux#0{sel=0,discard=true,disconnect=false}
  // CHECK-DAG: demux#0{sel=1,discard=true,disconnect=false}

  // Disconnect mode requires the demux input source to be dead, but %mul is
  // alive (yielded). So the disconnect configuration must be rejected.
  // CHECK-NOT: demux#0{sel=0,discard=false,disconnect=true}

  return
}
