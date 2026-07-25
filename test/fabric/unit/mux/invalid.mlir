// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Fewer than 2 inputs is illegal. The mux is wrapped in a PE/FU shell since
// fabric.mux must live inside fabric.fu.
fabric.module @mux_too_few(%a : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{requires at least 2 inputs}}
      %0 = "fabric.mux"(%fa) : (!fabric.bits<8>) -> !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// Partial software parameters (violates all-or-nothing rule).
fabric.module @mux_partial_params(%a : !fabric.bits<8>, %b : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{software parameters must be all set or all unset}}
      %0 = fabric.mux %fa, %fb {sel = 0 : i32, discard = false} : !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// discard and disconnect both true is illegal.
fabric.module @mux_discard_and_disconnect(%a : !fabric.bits<8>, %b : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{'discard' and 'disconnect' cannot both be true}}
      %0 = fabric.mux %fa, %fb {sel = 0 : i32, discard = true, disconnect = true} : !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// When disconnect is true, sel must be 0.
fabric.module @mux_disconnect_nonzero_sel(%a : !fabric.bits<8>, %b : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{when 'disconnect' is true, 'sel' must be 0}}
      %0 = fabric.mux %fa, %fb {sel = 1 : i32, discard = false, disconnect = true} : !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// sel out of [0, N).
fabric.module @mux_sel_out_of_range(%a : !fabric.bits<8>, %b : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>,
                    %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>,
              %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
      // expected-error @+1 {{'sel' (2) must be in [0, 2)}}
      %0 = fabric.mux %fa, %fb {sel = 2 : i32, discard = false, disconnect = false} : !fabric.bits<8>
      %k = fabric.op [@arith.addi] (%0, %0)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %k : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----
// bits_tag requires tagWidth > 0. Triggered by parsing the type itself.
// expected-error @+1 {{fabric.bits_tag requires tagWidth > 0}}
fabric.module @bad_bits_tag_zero_tag_width(%a : !fabric.bits_tag<8, 0>) {
  fabric.yield
}

// -----
// fabric.mux operand/result type is restricted to !fabric.bits<W>.
// Feeding a !fabric.bits_tag value is rejected by the op's type system
// before the FU/PE body whitelist or parent rules ever apply. Hosted at
// the top of builtin.module via unrealized_conversion_cast so the type
// constraint fires independently of any fabric.fu / fabric.module gate.
%a_mux_bt = builtin.unrealized_conversion_cast to !fabric.bits_tag<8, 2>
%b_mux_bt = builtin.unrealized_conversion_cast to !fabric.bits_tag<8, 2>
// expected-error @+1 {{must be variadic of fabric bits type}}
%mux_bt = "fabric.mux"(%a_mux_bt, %b_mux_bt)
     : (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>)
     -> !fabric.bits_tag<8, 2>

// -----
// fabric.mux operand/result type is restricted to !fabric.bits<W>.
// Feeding a tag-only !fabric.bits_tag<0,T> value is also rejected by the
// op's type system.
%a_mux_tag = builtin.unrealized_conversion_cast to !fabric.bits_tag<0, 4>
%b_mux_tag = builtin.unrealized_conversion_cast to !fabric.bits_tag<0, 4>
// expected-error @+1 {{must be variadic of fabric bits type}}
%mux_tag = "fabric.mux"(%a_mux_tag, %b_mux_tag)
     : (!fabric.bits_tag<0, 4>, !fabric.bits_tag<0, 4>)
     -> !fabric.bits_tag<0, 4>
