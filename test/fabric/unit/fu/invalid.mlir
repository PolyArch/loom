// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Body must contain at least one fabric.op.
func.func @fu_no_op(%a: !fabric.bits<8>) {
  // expected-error @+1 {{fabric.fu body requires at least one fabric.op}}
  fabric.fu(%x = %a : !fabric.bits<8>) -> () {
    fabric.yield
  }
  return
}

// -----
// fabric.fu cannot be nested.
func.func @fu_nested(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>) -> () {
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
  return
}

// -----
// fabric.fifo is not allowed inside fabric.fu.
func.func @fu_with_fifo(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>) -> () {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    // expected-error @+1 {{is not allowed inside fabric.fu}}
    %f = fabric.fifo %k [max_depth = 4, bypassable = false] : !fabric.bits<32>
    fabric.yield
  }
  return
}

// -----
// arith ops are not allowed in fabric.fu (only fabric ops).
func.func @fu_with_arith(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>) -> () {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    // expected-error @+1 {{is not allowed inside fabric.fu}}
    %s = arith.constant 0 : i32
    fabric.yield
  }
  return
}

// -----
// yield value count mismatch.
func.func @fu_yield_count_mismatch(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>) {
  %r:2 = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                  -> (!fabric.bits<32>, !fabric.bits<32>) {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    // expected-error @+1 {{yield value count (1) must match parent fabric.fu result count (2)}}
    fabric.yield %k : !fabric.bits<32>
  }
  return %r#0, %r#1 : !fabric.bits<32>, !fabric.bits<32>
}

// -----
// yield value type mismatch.
func.func @fu_yield_type_mismatch(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
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

// -----
// yield outside of fabric.fu.
func.func @fu_yield_outside() {
  // expected-error @+1 {{expects parent op 'fabric.fu'}}
  fabric.yield
}
