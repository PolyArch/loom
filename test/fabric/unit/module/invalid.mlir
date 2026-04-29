// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// fabric.module requires a sym_name; the parser rejects an anonymous module.
// expected-error @+1 {{expected valid '@'-identifier for symbol name}}
fabric.module {
}

// -----
// fabric.module body must be a single block.
// expected-error @below {{0 or 1 blocks}}
fabric.module @m_multi_block {
  cf.br ^bb1
^bb1:
  fabric.yield
}

// -----
// fabric.yield inside fabric.module must have zero operands.
fabric.module @m_yield_with_operands {
  %0 = builtin.unrealized_conversion_cast to !fabric.bits<32>
  // expected-error @+1 {{yield inside fabric.module must have no operands}}
  fabric.yield %0 : !fabric.bits<32>
}

// -----
// Strict body whitelist: only fabric.spatial_pe, fabric.fifo, and the
// implicit fabric.yield terminator are permitted in fabric.module's body.
// A raw fabric.fu directly in the module body is rejected.
fabric.module @m_raw_fu_rejected {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  // expected-error @+1 {{is not allowed inside fabric.module; only fabric.spatial_pe and fabric.fifo are permitted}}
  fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>) -> () {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield
  }
  fabric.yield
}

// -----
// Strict body whitelist: a raw fabric.op directly in the module body is
// rejected.
fabric.module @m_raw_op_rejected {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  // expected-error @+1 {{is not allowed inside fabric.module; only fabric.spatial_pe and fabric.fifo are permitted}}
  %k = fabric.op [@arith.addi] (%a, %b)
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  fabric.yield
}

// -----
// fabric.fu placed inside func.func is rejected by the FU verifier
// (parent must be fabric.spatial_pe).
func.func @fu_in_func(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  // expected-error @+1 {{must be inside a fabric.spatial_pe (parent must be fabric.spatial_pe)}}
  fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>) -> () {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield
  }
  return
}
