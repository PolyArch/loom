// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Fewer than 2 inputs is illegal. The mux is wrapped in a PE/FU shell since
// fabric.mux must live inside fabric.fu.
fabric.module @mux_too_few {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{requires at least 2 inputs}}
      %0 = "fabric.mux"(%fa) : (!fabric.bits<8>) -> !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// Partial software parameters (violates all-or-nothing rule).
fabric.module @mux_partial_params {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{software parameters must be all set or all unset}}
      %0 = fabric.mux %fa, %fb {sel = 0 : i32, discard = false} : !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// discard and disconnect both true is illegal.
fabric.module @mux_discard_and_disconnect {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{'discard' and 'disconnect' cannot both be true}}
      %0 = fabric.mux %fa, %fb {sel = 0 : i32, discard = true, disconnect = true} : !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// When disconnect is true, sel must be 0.
fabric.module @mux_disconnect_nonzero_sel {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{when 'disconnect' is true, 'sel' must be 0}}
      %0 = fabric.mux %fa, %fb {sel = 1 : i32, discard = false, disconnect = true} : !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// sel out of [0, N).
fabric.module @mux_sel_out_of_range {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{'sel' (2) must be in [0, 2)}}
      %0 = fabric.mux %fa, %fb {sel = 2 : i32, discard = false, disconnect = false} : !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// bits_tag requires width > 0. Triggered by parsing the type itself; the
// type appears here on a top-level cast since fabric.module's body
// whitelist admits builtin.unrealized_conversion_cast.
fabric.module @bad_bits_tag_zero_width {
  // expected-error @+1 {{fabric.bits_tag requires width > 0}}
  %a = builtin.unrealized_conversion_cast to !fabric.bits_tag<0, 2>
  fabric.yield
}

// -----
// tag requires tagWidth > 0. Triggered by parsing the type itself.
fabric.module @bad_tag_zero_width {
  // expected-error @+1 {{fabric.tag requires tagWidth > 0}}
  %a = builtin.unrealized_conversion_cast to !fabric.tag<0>
  fabric.yield
}
