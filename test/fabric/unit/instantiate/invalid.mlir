// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Undefined symbol: instantiate references a name that is not defined in
// any reachable SymbolTable.
%a = builtin.unrealized_conversion_cast to !fabric.bits<32>
// expected-error @+1 {{references undefined symbol '@missing'}}
%r = fabric.instantiate @missing(%a : !fabric.bits<32>) -> (!fabric.bits<32>)

// -----
// Wrong-kind target: instantiating a fabric.module symbol from inside a
// fabric.pe body is illegal (pe-body sites may target only fabric.fu).
fabric.module @leaf_mod(%x : !fabric.bits<32>) -> (!fabric.bits<32>) {
  fabric.yield %x : !fabric.bits<32>
}
fabric.module @host_wrong_kind(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    // expected-error @+1 {{inside a fabric.pe body may only target 'fabric.fu'}}
    %g = fabric.instantiate @leaf_mod(%pa : !fabric.bits<32>)
         -> (!fabric.bits<32>)
  }
  fabric.yield
}

// -----
// Self-reference: a fabric.module body cannot instantiate its own
// enclosing fabric.module symbol (recursion is forbidden).
fabric.module @recursive_self(%a : !fabric.bits<32>) {
  // expected-error @+1 {{cannot instantiate the symbol that encloses it (self-reference of '@recursive_self')}}
  %r = fabric.instantiate @recursive_self(%a : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield
}

// -----
// Forward reference: instantiate appears textually before the named pe
// definition in the same fabric.module body.
fabric.module @host_forward(%a : !fabric.bits<32>) {
  // expected-error @+1 {{forward reference to symbol '@LATER'}}
  %s = fabric.instantiate @LATER(%a : !fabric.bits<32>) -> (!fabric.bits<32>)
  %r = fabric.pe @LATER [spatial] (%pa = %a : !fabric.bits<32>)
                              -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Out-of-scope reference: a top-level fabric.instantiate cannot reach a
// pe symbol that is nested inside another fabric.module's body.
fabric.module @scope_leak_host(%a : !fabric.bits<32>) {
  %r = fabric.pe @INNER [spatial] (%pa = %a : !fabric.bits<32>)
                              -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}
%t = builtin.unrealized_conversion_cast to !fabric.bits<32>
// expected-error @+1 {{references undefined symbol '@INNER'}}
%u = fabric.instantiate @INNER(%t : !fabric.bits<32>) -> (!fabric.bits<32>)

// -----
// Operand count mismatch.
fabric.module @leaf_two_in(%x : !fabric.bits<32>, %y : !fabric.bits<32>)
    -> (!fabric.bits<32>) {
  fabric.yield %x : !fabric.bits<32>
}
fabric.module @host_count_mismatch(%a : !fabric.bits<32>) {
  // expected-error @+1 {{operand count (1) does not match callee '@leaf_two_in' input port count (2)}}
  %r = fabric.instantiate @leaf_two_in(%a : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield
}

// -----
// Output type mismatch: result type does not equal callee's declared
// output port type. Output direction is strict in this iteration.
fabric.module @leaf_out16(%x : !fabric.bits<32>) -> (!fabric.bits<16>) {
  %r = fabric.fifo %x [max_depth = 1, bypassable = false]
       : !fabric.bits<32> to !fabric.bits<16>
  fabric.yield %r : !fabric.bits<16>
}
fabric.module @host_out_mismatch(%a : !fabric.bits<32>) {
  // expected-error @+1 {{result #0 type '!fabric.bits<32>' must equal callee '@leaf_out16' output port type '!fabric.bits<16>'}}
  %r = fabric.instantiate @leaf_out16(%a : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield
}

// -----
// memref operands cannot use the 'to <inner-type>' clause: memref types
// must match exactly (no width relaxation on memref).
fabric.module @leaf_mem(%m : memref<8xi32>) -> (memref<8xi32>) {
  fabric.yield %m : memref<8xi32>
}
fabric.module @host_mem_relax(%m : memref<8xi32>) {
  // expected-error @+1 {{memref operands cannot use the 'to <inner-type>' clause}}
  %r = fabric.instantiate @leaf_mem(%m : memref<8xi32> to memref<4xi32>)
       -> (memref<8xi32>)
  fabric.yield
}
