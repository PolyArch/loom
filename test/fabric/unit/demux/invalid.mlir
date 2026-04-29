// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Fewer than 2 outputs is illegal.
fabric.module @demux_too_few {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>) -> !fabric.bits<8> {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      // expected-error @+1 {{requires at least 2 outputs}}
      %x = "fabric.demux"(%v) : (!fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %x : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// Partial software parameters.
fabric.module @demux_partial_params {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>)
                   -> (!fabric.bits<8>, !fabric.bits<8>) {
    fabric.fu(%fa = %pa : !fabric.bits<8>)
              -> (!fabric.bits<8>, !fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      // expected-error @+1 {{software parameters must be all set or all unset}}
      %x, %y = fabric.demux %v {sel = 0 : i32, discard = false} : !fabric.bits<8> -> 2
      fabric.yield %x, %y : !fabric.bits<8>, !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// discard and disconnect both true is illegal.
fabric.module @demux_discard_and_disconnect {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>)
                   -> (!fabric.bits<8>, !fabric.bits<8>) {
    fabric.fu(%fa = %pa : !fabric.bits<8>)
              -> (!fabric.bits<8>, !fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      // expected-error @+1 {{'discard' and 'disconnect' cannot both be true}}
      %x, %y = fabric.demux %v {sel = 0 : i32, discard = true, disconnect = true}
                : !fabric.bits<8> -> 2
      fabric.yield %x, %y : !fabric.bits<8>, !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// When disconnect is true, sel must be 0.
fabric.module @demux_disconnect_nonzero_sel {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>)
                   -> (!fabric.bits<8>, !fabric.bits<8>) {
    fabric.fu(%fa = %pa : !fabric.bits<8>)
              -> (!fabric.bits<8>, !fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      // expected-error @+1 {{when 'disconnect' is true, 'sel' must be 0}}
      %x, %y = fabric.demux %v {sel = 1 : i32, discard = false, disconnect = true}
                : !fabric.bits<8> -> 2
      fabric.yield %x, %y : !fabric.bits<8>, !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// sel out of [0, N).
fabric.module @demux_sel_out_of_range {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>)
                   -> (!fabric.bits<8>, !fabric.bits<8>) {
    fabric.fu(%fa = %pa : !fabric.bits<8>)
              -> (!fabric.bits<8>, !fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      // expected-error @+1 {{'sel' (2) must be in [0, 2)}}
      %x, %y = fabric.demux %v {sel = 2 : i32, discard = false, disconnect = false}
                : !fabric.bits<8> -> 2
      fabric.yield %x, %y : !fabric.bits<8>, !fabric.bits<8>
    }
  }
  fabric.yield
}
