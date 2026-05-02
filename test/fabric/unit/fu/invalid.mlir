// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Body must contain at least one fabric.op.
fabric.module @fu_no_op(%a : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>) -> !fabric.bits<8> {
    // expected-error @+1 {{fabric.fu body requires at least one fabric.op}}
    fabric.fu(%x = %pa : !fabric.bits<8>) -> () {
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// fabric.fu cannot be nested.
fabric.module @fu_nested(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> () {
      %k = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      // expected-error @+1 {{is not allowed inside fabric.fu}}
      fabric.fu(%xx = %x : !fabric.bits<32>) -> () {
        %inner = fabric.op [@arith.muli] (%xx, %xx)
                 : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield
      }
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// fabric.fifo is not allowed inside fabric.fu.
fabric.module @fu_with_fifo(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> () {
      %k = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      // expected-error @+1 {{is not allowed inside fabric.fu}}
      %f = fabric.fifo %k [max_depth = 4, bypassable = false] : !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// arith ops are not allowed in fabric.fu (only fabric ops).
fabric.module @fu_with_arith(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> () {
      %k = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      // expected-error @+1 {{is not allowed inside fabric.fu}}
      %s = arith.constant 0 : i32
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// yield value count mismatch.
fabric.module @fu_yield_count_mismatch(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>) {
    %r:2 = fabric.fu(%x = %pa : !fabric.bits<32>,
                     %y = %pb : !fabric.bits<32>)
                    -> (!fabric.bits<32>, !fabric.bits<32>) {
      %k = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      // expected-error @+1 {{yield value count (1) must match parent fabric.fu result count (2)}}
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// yield value type mismatch.
fabric.module @fu_yield_type_mismatch(%a : !fabric.bits<16>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<16>) -> !fabric.bits<16> {
    %r = fabric.fu(%x = %pa : !fabric.bits<16>) -> !fabric.bits<16> {
      %k = fabric.op [@arith.sitofp] (%x)
           : (!fabric.bits<16>) -> !fabric.bits<32>
      // expected-error @+1 {{yield value #0 type '!fabric.bits<32>' must match parent fabric.fu result type '!fabric.bits<16>'}}
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// fabric.yield placed at the top of builtin.module (no enclosing fabric.fu /
// fabric.module).
// expected-error @+1 {{expects parent op 'fabric.fu', 'fabric.pe' (named), or 'fabric.module'}}
fabric.yield

// -----
// FU boundary truncation: inner block-arg width must be <= outer operand
// width. Wider inner is illegal because hardware only supports high-bit
// truncation, not zero/sign extension.
fabric.module @fu_outer_lt_inner(%a : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>) -> !fabric.bits<8> {
    // expected-error @+1 {{operand #0 bits-width 8 is less than block-argument bits-width 32; the FU boundary only supports high-bit truncation (outer >= inner)}}
    fabric.fu(%fa = %pa : !fabric.bits<8> to !fabric.bits<32>) -> !fabric.bits<8> {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %z = fabric.op [@dataflow.constant] ()
           : () -> !fabric.bits<8>
      fabric.yield %z : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// FU output boundary widening: inner yield value width must be <= outer
// FU result width. Wider inner is illegal because hardware only supports
// low-bit-aligned widening (high bits zero-filled), not truncation on
// the output side. Here the inner body op produces a bits<32> result
// that the yield illegally tries to widen-down to a narrower outer
// bits<8> destination.
fabric.module @fu_yield_inner_gt_outer(%a : !fabric.bits<16>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<16>) -> !fabric.bits<16> {
    %r = fabric.fu(%x = %pa : !fabric.bits<16>) -> !fabric.bits<16> {
      %k = fabric.op [@arith.sitofp] (%x)
           : (!fabric.bits<16>) -> !fabric.bits<32>
      // expected-error @+1 {{yield value #0 inner bits-width 32 is greater than outer bits-width 16; the FU output boundary only supports low-bit-aligned widening (inner <= outer, high bits zero-filled)}}
      fabric.yield %k : !fabric.bits<32> to !fabric.bits<16>
    }
  }
  fabric.yield
}

// -----
// FU output boundary widening: per-value `to <type>` clause must equal
// the FU's declared outer result type. Mismatched destination type is
// rejected.
fabric.module @fu_yield_decl_mismatch(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %p = fabric.op [@arith.cmpi] (%x, %y)
           {hw_params = [{predicate = ["eq", "ne"]}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      // expected-error @+1 {{yield value #0: declared destination type '!fabric.bits<16>' does not match parent fabric.fu result type '!fabric.bits<32>'}}
      fabric.yield %p : !fabric.bits<1> to !fabric.bits<16>
    }
  }
  fabric.yield
}
