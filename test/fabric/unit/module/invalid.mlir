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
// Existing rule: yield value type must match the parent fabric.fu result type.
func.func @fu_yield_type_mismatch_regression(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
    -> !fabric.bits<16> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<16> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    // expected-error @+1 {{yield value #0 type '!fabric.bits<32>' must match parent fabric.fu result type '!fabric.bits<16>'}}
    fabric.yield %k : !fabric.bits<32>
  }
  return %r : !fabric.bits<16>
}
