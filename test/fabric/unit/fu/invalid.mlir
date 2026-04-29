// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Body must contain at least one fabric.op.
fabric.module @fu_no_op {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<8>
  fabric.spatial_pe(%pa = %a : !fabric.bits<8>) -> !fabric.bits<8> {
    // expected-error @+1 {{fabric.fu body requires at least one fabric.op}}
    fabric.fu(%x = %pa : !fabric.bits<8>) -> () {
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// fabric.fu cannot be nested.
fabric.module @fu_nested {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
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
fabric.module @fu_with_fifo {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
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
fabric.module @fu_with_arith {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
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
fabric.module @fu_yield_count_mismatch {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
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
fabric.module @fu_yield_type_mismatch {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<16>
  fabric.spatial_pe(%pa = %a : !fabric.bits<16>) -> !fabric.bits<16> {
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
// yield outside of fabric.fu / fabric.module.
func.func @fu_yield_outside() {
  // expected-error @+1 {{expects parent op 'fabric.fu' or 'fabric.module'}}
  fabric.yield
}
