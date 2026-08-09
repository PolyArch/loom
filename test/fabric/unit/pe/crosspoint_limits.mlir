// RUN: loom %s -verify-diagnostics

fabric.module @spatial_anonymous_quiet(
    %a0 : !fabric.bits<8>, %a1 : !fabric.bits<8>,
    %a2 : !fabric.bits<8>, %a3 : !fabric.bits<8>) {
  %r:4 = fabric.pe [spatial]
      (%p0 = %a0 : !fabric.bits<8>, %p1 = %a1 : !fabric.bits<8>,
       %p2 = %a2 : !fabric.bits<8>, %p3 = %a3 : !fabric.bits<8>)
      -> (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>,
          !fabric.bits<8>) {
    fabric.fu(%f0 = %p0 : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%f0, %f0)
          {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [8 : i32]}}
          : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----

fabric.module @spatial_named_warning() {
  // expected-warning @+1 {{fabric.pe boundary selectors have 20 crosspoints; values above 16 may be implementation-inefficient}}
  fabric.pe @SpatialNamedWarning [spatial]
      (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>)
      -> (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>,
          !fabric.bits<8>, !fabric.bits<8>) {
  ^bb0(%p0 : !fabric.bits<8>, %p1 : !fabric.bits<8>,
       %p2 : !fabric.bits<8>, %p3 : !fabric.bits<8>):
    fabric.fu(%f0 = %p0 : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%f0, %f0)
          {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [8 : i32]}}
          : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
    fabric.yield
  }
  fabric.yield
}

// -----

fabric.module @spatial_anonymous_boundary(
    %a0 : !fabric.bits<8>, %a1 : !fabric.bits<8>,
    %a2 : !fabric.bits<8>, %a3 : !fabric.bits<8>,
    %a4 : !fabric.bits<8>, %a5 : !fabric.bits<8>,
    %a6 : !fabric.bits<8>, %a7 : !fabric.bits<8>) {
  // expected-warning @+1 {{fabric.pe boundary selectors have 64 crosspoints; values above 16 may be implementation-inefficient}}
  %r:8 = fabric.pe [spatial]
      (%p0 = %a0 : !fabric.bits<8>, %p1 = %a1 : !fabric.bits<8>,
       %p2 = %a2 : !fabric.bits<8>, %p3 = %a3 : !fabric.bits<8>,
       %p4 = %a4 : !fabric.bits<8>, %p5 = %a5 : !fabric.bits<8>,
       %p6 = %a6 : !fabric.bits<8>, %p7 = %a7 : !fabric.bits<8>)
      -> (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>,
          !fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>,
          !fabric.bits<8>, !fabric.bits<8>) {
    fabric.fu(%f0 = %p0 : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%f0, %f0)
          {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [8 : i32]}}
          : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----

fabric.module @spatial_named_oversized() {
  // expected-error @+1 {{fabric.pe boundary selectors have 72 crosspoints, exceeding maximum 64}}
  fabric.pe @SpatialNamedOversized [spatial]
      (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>,
       !fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>,
       !fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>)
      -> (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>,
          !fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>,
          !fabric.bits<8>, !fabric.bits<8>) {
  ^bb0(%p0 : !fabric.bits<8>, %p1 : !fabric.bits<8>,
       %p2 : !fabric.bits<8>, %p3 : !fabric.bits<8>,
       %p4 : !fabric.bits<8>, %p5 : !fabric.bits<8>,
       %p6 : !fabric.bits<8>, %p7 : !fabric.bits<8>,
       %p8 : !fabric.bits<8>):
    fabric.fu(%f0 = %p0 : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%f0, %f0)
          {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [8 : i32]}}
          : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
    fabric.yield
  }
  fabric.yield
}
