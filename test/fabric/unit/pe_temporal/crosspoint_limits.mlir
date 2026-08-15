// RUN: loom %s -verify-diagnostics

fabric.module @temporal_anonymous_quiet(
    %a0 : !fabric.bits_tag<8, 2>, %a1 : !fabric.bits_tag<8, 2>,
    %a2 : !fabric.bits_tag<8, 2>, %a3 : !fabric.bits_tag<8, 2>) {
  %r:4 = fabric.pe [temporal]
      (%p0 = %a0 : !fabric.bits_tag<8, 2>,
       %p1 = %a1 : !fabric.bits_tag<8, 2>,
       %p2 = %a2 : !fabric.bits_tag<8, 2>,
       %p3 = %a3 : !fabric.bits_tag<8, 2>)
      -> (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>)
      attributes {
        tag_width = 2 : i32,
        num_instruction = 1 : i32,
        fu_config_mode = #fabric.fu_config_mode<per_fu_config>,
        operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
        operand_buffer_size = 2 : i32
      } {
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

fabric.module @temporal_named_warning() {
  // expected-warning @+1 {{fabric.pe boundary selectors have 20 crosspoints; values above 16 may be implementation-inefficient}}
  fabric.pe @TemporalNamedWarning [temporal]
      (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
       !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>)
      -> (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>)
      attributes {
        tag_width = 2 : i32,
        num_instruction = 1 : i32,
        fu_config_mode = #fabric.fu_config_mode<per_fu_config>,
        operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
        operand_buffer_size = 2 : i32
      } {
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

fabric.module @temporal_anonymous_boundary(
    %a0 : !fabric.bits_tag<8, 2>, %a1 : !fabric.bits_tag<8, 2>,
    %a2 : !fabric.bits_tag<8, 2>, %a3 : !fabric.bits_tag<8, 2>,
    %a4 : !fabric.bits_tag<8, 2>, %a5 : !fabric.bits_tag<8, 2>,
    %a6 : !fabric.bits_tag<8, 2>, %a7 : !fabric.bits_tag<8, 2>) {
  // expected-warning @+1 {{fabric.pe boundary selectors have 64 crosspoints; values above 16 may be implementation-inefficient}}
  %r:8 = fabric.pe [temporal]
      (%p0 = %a0 : !fabric.bits_tag<8, 2>,
       %p1 = %a1 : !fabric.bits_tag<8, 2>,
       %p2 = %a2 : !fabric.bits_tag<8, 2>,
       %p3 = %a3 : !fabric.bits_tag<8, 2>,
       %p4 = %a4 : !fabric.bits_tag<8, 2>,
       %p5 = %a5 : !fabric.bits_tag<8, 2>,
       %p6 = %a6 : !fabric.bits_tag<8, 2>,
       %p7 = %a7 : !fabric.bits_tag<8, 2>)
      -> (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>)
      attributes {
        tag_width = 2 : i32,
        num_instruction = 1 : i32,
        fu_config_mode = #fabric.fu_config_mode<per_fu_config>,
        operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
        operand_buffer_size = 2 : i32
      } {
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

fabric.module @temporal_named_oversized() {
  // expected-error @+1 {{fabric.pe boundary selectors have 72 crosspoints, exceeding maximum 64}}
  fabric.pe @TemporalNamedOversized [temporal]
      (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
       !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
       !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
       !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
       !fabric.bits_tag<8, 2>)
      -> (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>)
      attributes {
        tag_width = 2 : i32,
        num_instruction = 1 : i32,
        fu_config_mode = #fabric.fu_config_mode<per_fu_config>,
        operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
        operand_buffer_size = 2 : i32
      } {
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
