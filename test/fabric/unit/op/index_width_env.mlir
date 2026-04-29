// Verifies that LOOM_INDEX_WIDTH overrides the default index width (32).
// With LOOM_INDEX_WIDTH=16, a 4-input mux must use bits<16> for sel; bits<32>
// should be rejected.

// RUN: env LOOM_INDEX_WIDTH=16 loom %s -split-input-file -verify-diagnostics

fabric.module @mux_index_env_ok {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %c = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %d = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>,
                    %pc = %c : !fabric.bits<8>,
                    %pd = %d : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>,
              %fc = %pc : !fabric.bits<8>,
              %fd = %pd : !fabric.bits<8>) -> () {
      %sel = fabric.op [@arith.sitofp] (%fa)
             : (!fabric.bits<8>) -> !fabric.bits<16>
      %0 = fabric.op [@dataflow.mux] (%sel, %fa, %fb, %fc, %fd)
           : (!fabric.bits<16>, !fabric.bits<8>, !fabric.bits<8>,
              !fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield
    }
  }
  fabric.yield
}

// -----

fabric.module @mux_index_env_rejects_default {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %c = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %d = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>,
                    %pc = %c : !fabric.bits<8>,
                    %pd = %d : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>,
              %fc = %pc : !fabric.bits<8>,
              %fd = %pd : !fabric.bits<8>) -> () {
      %sel = fabric.op [@arith.sitofp] (%fa)
             : (!fabric.bits<8>) -> !fabric.bits<32>
      // expected-error @+1 {{sel port (input #0) width 32 must be 16}}
      %0 = fabric.op [@dataflow.mux] (%sel, %fa, %fb, %fc, %fd)
           : (!fabric.bits<32>, !fabric.bits<8>, !fabric.bits<8>,
              !fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield
    }
  }
  fabric.yield
}
