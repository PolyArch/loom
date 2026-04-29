// RUN: loom %s | loom | FileCheck %s

// Note: fabric.mux operand/result types are !fabric.bits<W>; see the
// Fabric_MuxOp declaration in include/Fabric/IR/FabricOps.td. fabric.mux
// must live inside fabric.fu (the fabric.module body whitelist admits only
// fabric.spatial_pe, fabric.fifo and fabric.yield).

// -----------------------------------------------------------------------------
// Pure hardware (no software params programmed): 2 inputs, bits<8>.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mux_hw_bits
fabric.module @mux_hw_bits(%a : !fabric.bits<8>, %b : !fabric.bits<8>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
      // CHECK: fabric.mux %{{.*}}, %{{.*}} : !fabric.bits<8>
      %0 = fabric.mux %fa, %fb : !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Pure hardware (no software params programmed): 3 inputs, bits<0>.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mux_hw_bits_zero
fabric.module @mux_hw_bits_zero(%a : !fabric.bits<0>, %b : !fabric.bits<0>, %c : !fabric.bits<0>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<0>,
                    %pb = %b : !fabric.bits<0>,
                    %pc = %c : !fabric.bits<0>) -> !fabric.bits<0> {
    fabric.fu(%fa = %pa : !fabric.bits<0>,
              %fb = %pb : !fabric.bits<0>,
              %fc = %pc : !fabric.bits<0>) -> !fabric.bits<0> {
      // CHECK: fabric.mux %{{.*}}, %{{.*}}, %{{.*}} : !fabric.bits<0>
      %0 = fabric.mux %fa, %fb, %fc : !fabric.bits<0>
      %k = fabric.op [@dataflow.constant] (%0)
           {sw_configs = {const_hex_value = "0"}}
           : (!fabric.bits<0>) -> !fabric.bits<0>
      fabric.yield %k : !fabric.bits<0>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Programmed: normal pass-through.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mux_sw_passthrough
fabric.module @mux_sw_passthrough(%a : !fabric.bits<16>, %b : !fabric.bits<16>, %c : !fabric.bits<16>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<16>,
                    %pb = %b : !fabric.bits<16>,
                    %pc = %c : !fabric.bits<16>) -> !fabric.bits<16> {
    fabric.fu(%fa = %pa : !fabric.bits<16>,
              %fb = %pb : !fabric.bits<16>,
              %fc = %pc : !fabric.bits<16>) -> !fabric.bits<16> {
      // CHECK: fabric.mux %{{.*}}, %{{.*}}, %{{.*}} {sel = 1 : i32, discard = false, disconnect = false} : !fabric.bits<16>
      %0 = fabric.mux %fa, %fb, %fc {sel = 1 : i32, discard = false, disconnect = false} : !fabric.bits<16>
      %k = fabric.op [@arith.addi] (%0, %0)
           : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      fabric.yield %k : !fabric.bits<16>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Programmed: discard mode.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mux_sw_discard
fabric.module @mux_sw_discard(%a : !fabric.bits<4>, %b : !fabric.bits<4>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<4>,
                    %pb = %b : !fabric.bits<4>) -> !fabric.bits<4> {
    fabric.fu(%fa = %pa : !fabric.bits<4>,
              %fb = %pb : !fabric.bits<4>) -> !fabric.bits<4> {
      // CHECK: fabric.mux %{{.*}}, %{{.*}} {sel = 0 : i32, discard = true, disconnect = false} : !fabric.bits<4>
      %0 = fabric.mux %fa, %fb {sel = 0 : i32, discard = true, disconnect = false} : !fabric.bits<4>
      %k = fabric.op [@arith.addi] (%0, %0)
           : (!fabric.bits<4>, !fabric.bits<4>) -> !fabric.bits<4>
      fabric.yield %k : !fabric.bits<4>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Programmed: disconnect mode (sel forced to 0).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mux_sw_disconnect
fabric.module @mux_sw_disconnect(%a : !fabric.bits<4>, %b : !fabric.bits<4>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<4>,
                    %pb = %b : !fabric.bits<4>) -> !fabric.bits<4> {
    fabric.fu(%fa = %pa : !fabric.bits<4>,
              %fb = %pb : !fabric.bits<4>) -> !fabric.bits<4> {
      // CHECK: fabric.mux %{{.*}}, %{{.*}} {sel = 0 : i32, discard = false, disconnect = true} : !fabric.bits<4>
      %0 = fabric.mux %fa, %fb {sel = 0 : i32, discard = false, disconnect = true} : !fabric.bits<4>
      %k = fabric.op [@arith.addi] (%0, %0)
           : (!fabric.bits<4>, !fabric.bits<4>) -> !fabric.bits<4>
      fabric.yield %k : !fabric.bits<4>
    }
  }
  fabric.yield
}
