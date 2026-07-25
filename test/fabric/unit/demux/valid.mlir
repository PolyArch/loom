// RUN: loom %s | loom | FileCheck %s

// Note: fabric.demux operand/result types are !fabric.bits<W>; see the
// Fabric_DemuxOp declaration in include/Fabric/IR/FabricOps.td. fabric.demux
// must live inside fabric.fu (the fabric.module body whitelist admits only
// fabric.pe, fabric.fifo and fabric.yield).

// -----------------------------------------------------------------------------
// Pure hardware (no software params programmed): bits<8>, 2 outputs.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @demux_hw_bits
fabric.module @demux_hw_bits(%a : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>)
                   -> (!fabric.bits<8>, !fabric.bits<8>) {
    fabric.fu(%fa = %pa : !fabric.bits<8>)
              -> (!fabric.bits<8>, !fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      // CHECK: fabric.demux %{{.*}} : !fabric.bits<8> -> 2
      %x, %y = fabric.demux %v : !fabric.bits<8> -> 2
      fabric.yield %x, %y : !fabric.bits<8>, !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Programmed: route to selected output.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @demux_sw_route
fabric.module @demux_sw_route(%a : !fabric.bits<16>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<16>)
                   -> (!fabric.bits<16>, !fabric.bits<16>, !fabric.bits<16>) {
    fabric.fu(%fa = %pa : !fabric.bits<16>)
              -> (!fabric.bits<16>, !fabric.bits<16>, !fabric.bits<16>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
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
fabric.module @demux_sw_discard(%a : !fabric.bits<4>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<4>)
                   -> (!fabric.bits<4>, !fabric.bits<4>) {
    fabric.fu(%fa = %pa : !fabric.bits<4>)
              -> (!fabric.bits<4>, !fabric.bits<4>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<4>, !fabric.bits<4>) -> !fabric.bits<4>
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
fabric.module @demux_sw_disconnect(%a : !fabric.bits<4>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<4>)
                   -> (!fabric.bits<4>, !fabric.bits<4>) {
    fabric.fu(%fa = %pa : !fabric.bits<4>)
              -> (!fabric.bits<4>, !fabric.bits<4>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<4>, !fabric.bits<4>) -> !fabric.bits<4>
      // CHECK: fabric.demux %{{.*}} {sel = 0 : i32, discard = false, disconnect = true} : !fabric.bits<4> -> 2
      %x, %y = fabric.demux %v {sel = 0 : i32, discard = false, disconnect = true}
                : !fabric.bits<4> -> 2
      fabric.yield %x, %y : !fabric.bits<4>, !fabric.bits<4>
    }
  }
  fabric.yield
}
