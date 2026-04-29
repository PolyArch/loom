// RUN: loom %s | loom | FileCheck %s

// Note: fabric.demux must live inside fabric.fu (per architecture: fabric.module
// body only admits fabric.spatial_pe, fabric.fifo and fabric.yield; the
// fabric.fu body is the only place that admits fabric.demux). PE/FU ports must
// be !fabric.bits<W>, so the bits_tag and tag demux variants from the previous
// suite are dropped here: the type-only round-trip is covered by fabric.fifo
// on bits_tag and tag in the fifo unit suite.

// -----------------------------------------------------------------------------
// Pure hardware (no software params programmed): bits<8>, 2 outputs.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @demux_hw_bits
fabric.module @demux_hw_bits {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>)
                   -> (!fabric.bits<8>, !fabric.bits<8>) {
    fabric.fu(%fa = %pa : !fabric.bits<8>)
              -> (!fabric.bits<8>, !fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      // CHECK: fabric.demux %{{.*}} : !fabric.bits<8> -> 2
      %x, %y = fabric.demux %v : !fabric.bits<8> -> 2
      fabric.yield %x, %y : !fabric.bits<8>, !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Pure hardware (no software params programmed): bits<0>, 3 outputs.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @demux_hw_bits_zero
fabric.module @demux_hw_bits_zero {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<0>
  fabric.spatial_pe(%pa = %a : !fabric.bits<0>)
                   -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
    fabric.fu(%fa = %pa : !fabric.bits<0>)
              -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
      %v = fabric.op [@dataflow.constant] (%fa)
           {sw_configs = {const_hex_value = "0"}}
           : (!fabric.bits<0>) -> !fabric.bits<0>
      // CHECK: fabric.demux %{{.*}} : !fabric.bits<0> -> 3
      %x, %y, %z = fabric.demux %v : !fabric.bits<0> -> 3
      fabric.yield %x, %y, %z : !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Programmed: route to selected output.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @demux_sw_route
fabric.module @demux_sw_route {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<16>
  fabric.spatial_pe(%pa = %a : !fabric.bits<16>)
                   -> (!fabric.bits<16>, !fabric.bits<16>, !fabric.bits<16>) {
    fabric.fu(%fa = %pa : !fabric.bits<16>)
              -> (!fabric.bits<16>, !fabric.bits<16>, !fabric.bits<16>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      // CHECK: fabric.demux %{{.*}} {sel = 1 : i32, discard = false, disconnect = false} : !fabric.bits<16> -> 3
      %x, %y, %z = fabric.demux %v {sel = 1 : i32, discard = false, disconnect = false}
                    : !fabric.bits<16> -> 3
      fabric.yield %x, %y, %z : !fabric.bits<16>, !fabric.bits<16>, !fabric.bits<16>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Programmed: discard mode (input drained, no output token).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @demux_sw_discard
fabric.module @demux_sw_discard {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<4>
  fabric.spatial_pe(%pa = %a : !fabric.bits<4>)
                   -> (!fabric.bits<4>, !fabric.bits<4>) {
    fabric.fu(%fa = %pa : !fabric.bits<4>)
              -> (!fabric.bits<4>, !fabric.bits<4>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<4>, !fabric.bits<4>) -> !fabric.bits<4>
      // CHECK: fabric.demux %{{.*}} {sel = 0 : i32, discard = true, disconnect = false} : !fabric.bits<4> -> 2
      %x, %y = fabric.demux %v {sel = 0 : i32, discard = true, disconnect = false}
                : !fabric.bits<4> -> 2
      fabric.yield %x, %y : !fabric.bits<4>, !fabric.bits<4>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Programmed: disconnect mode (sel forced to 0).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @demux_sw_disconnect
fabric.module @demux_sw_disconnect {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<4>
  fabric.spatial_pe(%pa = %a : !fabric.bits<4>)
                   -> (!fabric.bits<4>, !fabric.bits<4>) {
    fabric.fu(%fa = %pa : !fabric.bits<4>)
              -> (!fabric.bits<4>, !fabric.bits<4>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<4>, !fabric.bits<4>) -> !fabric.bits<4>
      // CHECK: fabric.demux %{{.*}} {sel = 0 : i32, discard = false, disconnect = true} : !fabric.bits<4> -> 2
      %x, %y = fabric.demux %v {sel = 0 : i32, discard = false, disconnect = true}
                : !fabric.bits<4> -> 2
      fabric.yield %x, %y : !fabric.bits<4>, !fabric.bits<4>
    }
  }
  fabric.yield
}
