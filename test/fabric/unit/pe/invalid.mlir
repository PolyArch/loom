// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// K=0: pe must have at least one input port.
fabric.module @pe_no_inputs() {
  // expected-error @+1 {{requires at least 1 input port (K >= 1)}}
  %r = fabric.pe [spatial] () -> (!fabric.bits<32>) {
    fabric.fu() -> (!fabric.bits<32>) {
      %v = fabric.op [@dataflow.constant] ()
           {sw_configs = {const_hex_value = "0"}}
           : () -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// L=0: pe must have at least one output port.
fabric.module @pe_no_outputs(%a : !fabric.bits<32>) {
  // expected-error @+1 {{requires at least 1 output port (L >= 1)}}
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> () {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Mixed widths on PE inputs.
fabric.module @pe_mixed_input_widths(%a : !fabric.bits<32>, %b : !fabric.bits<16>) {
  // expected-error @+1 {{requires uniform 'bits<W>' on all PE ports}}
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                            %pb = %b : !fabric.bits<16>) -> (!fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Mixed widths between PE inputs and PE outputs.
fabric.module @pe_input_output_width_mismatch(%a : !fabric.bits<32>) {
  // expected-error @+1 {{requires uniform 'bits<W>' on all PE ports}}
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<16> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Non-bits PE port type. The PE's operand type constraint is fabric.bits<W>;
// feeding a !fabric.bits_tag value is rejected by the op's type system.
fabric.module @pe_non_bits_port(%a : !fabric.bits_tag<8, 2>) {
  // expected-error @+1 {{pe' op operand}}
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits_tag<8, 2>) -> (!fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits_tag<8, 2>) -> (!fabric.bits<32>) {
      %v = fabric.op [@dataflow.constant] ()
           : () -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Empty body: no inner FU.
fabric.module @pe_empty_body(%a : !fabric.bits<32>) {
  // expected-error @+1 {{body requires at least one fabric.fu}}
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
  }
  fabric.yield
}

// -----
// Body contains fabric.op directly.
fabric.module @pe_body_has_op(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> (!fabric.bits<32>) {
    // expected-error @+1 {{body may only contain fabric.fu}}
    %v = fabric.op [@arith.addi] (%pa, %pa)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %w = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %w : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Body contains fabric.yield (no terminator allowed).
fabric.module @pe_body_has_yield(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> (!fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %w = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %w : !fabric.bits<32>
    }
    // expected-error @+1 {{fabric.yield is not allowed in an anonymous fabric.pe body}}
    fabric.yield
  }
  fabric.yield
}

// -----
// Inner FU has more inputs than the PE's K.
fabric.module @pe_inner_fu_too_many_inputs(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> (!fabric.bits<32>) {
    // expected-error @+1 {{inner fabric.fu has 2 inputs which exceeds fabric.pe input count K=1}}
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Inner FU has more outputs than the PE's L.
fabric.module @pe_inner_fu_too_many_outputs(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    // expected-error @+1 {{inner fabric.fu has 2 outputs which exceeds fabric.pe output count L=1}}
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %d:2 = fabric.demux %v {sel = 0 : i32, discard = false, disconnect = false}
             : !fabric.bits<32> -> 2
      fabric.yield %d#0, %d#1 : !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Inner FU input width does not match PE width.
fabric.module @pe_inner_fu_input_width_mismatch(%a : !fabric.bits<32>, %b : !fabric.bits<16>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    // expected-error @+1 {{inner fabric.fu boundary width must equal fabric.pe width W=32}}
    fabric.fu(%fa = %b : !fabric.bits<16>) -> (!fabric.bits<32>) {
      %v = fabric.op [@dataflow.constant] ()
           : () -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Inner FU output width does not match PE width.
fabric.module @pe_inner_fu_output_width_mismatch(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    // expected-error @+1 {{inner fabric.fu boundary width must equal fabric.pe width W=32}}
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<16>) {
      %v = fabric.op [@dataflow.constant] ()
           : () -> !fabric.bits<16>
      fabric.yield %v : !fabric.bits<16>
    }
  }
  fabric.yield
}

// -----
// pe at the top of builtin.module (parent must be fabric.module).
%a_top = builtin.unrealized_conversion_cast to !fabric.bits<32>
// expected-error @+1 {{'fabric.pe' op expects parent op 'fabric.module'}}
%r = fabric.pe [spatial] (%pa = %a_top : !fabric.bits<32>) -> (!fabric.bits<32>) {
  fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
    %v = fabric.op [@arith.addi] (%fa, %fa)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %v : !fabric.bits<32>
  }
}

// -----
// pe at the top of builtin.module (no enclosing fabric.module).
%a0 = builtin.unrealized_conversion_cast to !fabric.bits<32>
// expected-error @+1 {{'fabric.pe' op expects parent op 'fabric.module'}}
%r0 = fabric.pe [spatial] (%pa = %a0 : !fabric.bits<32>) -> (!fabric.bits<32>) {
  fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
    %v = fabric.op [@arith.addi] (%fa, %fa)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %v : !fabric.bits<32>
  }
}

// -----
// Nested pe: one inside another's body. The body whitelist rejects
// the inner pe before the parent rule has a chance to fire.
fabric.module @pe_nested(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> (!fabric.bits<32>) {
    // expected-error @+1 {{body may only contain fabric.fu}}
    %inner = fabric.pe [spatial] (%qa = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%fa = %qa : !fabric.bits<32>) -> (!fabric.bits<32>) {
        %w = fabric.op [@arith.addi] (%fa, %fa)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %w : !fabric.bits<32>
      }
    }
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Schedule = temporal: parsable but the verifier rejects with a
// "not yet implemented" diagnostic. The temporal branch will be
// completed in the next task.
fabric.module @pe_temporal_not_yet_implemented(%a : !fabric.bits<32>) {
  // expected-error @+1 {{fabric.pe in 'temporal' schedule is not yet implemented}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits<32>) -> (!fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}
