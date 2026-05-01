// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Spatial PE rejecting any temporal-only attribute (forbidden cross-branch).
fabric.module @spatial_with_tag_width(%a : !fabric.bits<32>) {
  // expected-error @+1 {{spatial fabric.pe must not carry temporal-only attribute 'tag_width'}}
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32>
       attributes { tag_width = 4 : i32 } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: PE port must be bits_tag, not bits.
fabric.module @temp_non_tag_port(%a : !fabric.bits<32>) {
  // expected-error @+1 {{temporal fabric.pe boundary type must be '!fabric.bits_tag<W, T>'}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: mismatched W between input and output ports.
fabric.module @temp_mismatched_W(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{requires uniform 'bits_tag<W, T>' on all PE ports}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<16, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: mismatched T (tag width) between ports.
fabric.module @temp_mismatched_T(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{requires uniform 'bits_tag<W, T>' on all PE ports}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 2>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: tag_width attribute must equal the boundary T.
fabric.module @temp_tag_width_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'tag_width' attribute (3) must equal PE boundary tag width T (4)}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 3 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: num_instruction must be >= 1.
fabric.module @temp_zero_instructions(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'num_instruction' must be >= 1, got 0}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 0 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: reg_fifo_depth must be present when num_reg_fifo > 0.
fabric.module @temp_reg_fifo_no_depth(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'reg_fifo_depth' is required when 'num_reg_fifo' > 0}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         num_reg_fifo = 4 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: reg_fifo_depth present but num_reg_fifo == 0.
fabric.module @temp_reg_fifo_depth_no_regs(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'reg_fifo_depth' must be absent (or 0) when 'num_reg_fifo' is 0}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         reg_fifo_depth = 4 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: operand_buffer_size present with operand_buffer_mode = per_instruction.
fabric.module @temp_buffer_size_per_instruction(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'operand_buffer_size' must be absent when 'operand_buffer_mode' is 'per_instruction'}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction",
         operand_buffer_size = 4 : i32
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: operand_buffer_size missing when mode != per_instruction.
fabric.module @temp_buffer_size_missing(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'operand_buffer_size' is required when 'operand_buffer_mode' is not 'per_instruction'}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_input_port"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Programmed temporal PE: instruction entry with both discard and disconnect.
fabric.module @temp_discard_disconnect(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{instruction[0] operand_sel[0] cannot have both 'discard' and 'disconnect' true}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction",
         pe_enable = true,
         instruction_mem = [
           {
             enable = true,
             opcode = 0 : i1,
             operand_sel = [
               { src_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = true, disconnect = true }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ]
           }
         ],
         per_fu_sw_configs = [ {} ]
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Programmed temporal PE: instruction entry with is_port=false but no reg fifos.
fabric.module @temp_reg_src_no_regs(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{instruction[0] operand_sel[0] uses 'is_port' = false but 'num_reg_fifo' is 0}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction",
         pe_enable = true,
         instruction_mem = [
           {
             enable = true,
             opcode = 0 : i1,
             operand_sel = [
               { src_sel = 0 : i32, tag = 0 : i4, is_port = false,
                 discard = false, disconnect = false }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ]
           }
         ],
         per_fu_sw_configs = [ {} ]
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Programmed temporal PE: src_sel out of [0, K) range.
fabric.module @temp_src_sel_oor(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{instruction[0] operand_sel[0] 'src_sel' (3) must be < K (1)}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction",
         pe_enable = true,
         instruction_mem = [
           {
             enable = true,
             opcode = 0 : i1,
             operand_sel = [
               { src_sel = 3 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ]
           }
         ],
         per_fu_sw_configs = [ {} ]
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Programmed temporal PE: opcode out of range (opcode >= num_fu).
fabric.module @temp_opcode_oor(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{instruction[0] 'opcode' (5) must be < num_fu (1)}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction",
         pe_enable = true,
         instruction_mem = [
           {
             enable = true,
             opcode = 5 : i8,
             operand_sel = [
               { src_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ]
           }
         ],
         per_fu_sw_configs = [ {} ]
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Programmed temporal PE: instruction_mem length != num_instruction.
fabric.module @temp_inst_count_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'instruction_mem' length (1) must equal 'num_instruction' (2)}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 2 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction",
         pe_enable = true,
         instruction_mem = [
           {
             enable = true,
             opcode = 0 : i1,
             operand_sel = [
               { src_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ]
           }
         ],
         per_fu_sw_configs = [ {} ]
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: pe_enable present but instruction_mem absent (all-or-nothing).
fabric.module @temp_partial_sw_configs(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{all-or-nothing violation: 'pe_enable' is present but 'instruction_mem' is missing}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction",
         pe_enable = true
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: per_fu_sw_configs length != num_fu.
fabric.module @temp_per_fu_count_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'per_fu_sw_configs' length (3) must equal num_fu (1)}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction",
         pe_enable = true,
         instruction_mem = [
           {
             enable = true,
             opcode = 0 : i1,
             operand_sel = [
               { src_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ]
           }
         ],
         per_fu_sw_configs = [ {}, {}, {} ]
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: per_instruction_fu_config requires fu_sw_configs in each entry.
fabric.module @temp_per_inst_no_fu_cfg(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{instruction[0] is missing 'fu_sw_configs' (required for 'per_instruction_fu_config')}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_instruction_fu_config",
         operand_buffer_mode = "per_instruction",
         pe_enable = true,
         instruction_mem = [
           {
             enable = true,
             opcode = 0 : i1,
             operand_sel = [
               { src_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false }
             ]
           }
         ]
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Temporal PE: empty body (no inner FU).
fabric.module @temp_empty_body(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{body requires at least one fabric.fu or fabric.instantiate}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
  }
  fabric.yield
}

// -----
// Temporal PE: missing required hw param (num_instruction).
fabric.module @temp_missing_num_instruction(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{temporal fabric.pe requires 'num_instruction' attribute}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Named temporal PE: entry block arg type bits_tag<W, T> is forbidden
// (boundary auto-strip means body-level args must be bits<W'>).
fabric.module @temp_named_arg_is_tag() {
  // expected-error @+1 {{named PE entry block arg #0 type '!fabric.bits_tag<32, 4>' is bits_tag (forbidden)}}
  fabric.pe @TempBadArg [temporal] (!fabric.bits_tag<32, 4>)
                                    -> (!fabric.bits_tag<32, 4>)
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
  ^bb0(%pa: !fabric.bits_tag<32, 4>):
    fabric.fu() -> (!fabric.bits<32>) {
      %k = fabric.op [@dataflow.constant] ()
           {sw_configs = {const_hex_value = "0xdeadbeef"}}
           : () -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
    fabric.yield %pa : !fabric.bits_tag<32, 4>
  }
  fabric.yield
}

// -----
// Named temporal PE: entry block arg bits-width exceeds the port
// bits-data-width (truncation only narrows, never widens).
fabric.module @temp_named_arg_too_wide() {
  // expected-error @+1 {{named PE entry block arg #0 bits-width 32 > port bits-data-width 16}}
  fabric.pe @TempBadWidth [temporal] (!fabric.bits_tag<16, 2>)
                                      -> (!fabric.bits_tag<16, 2>)
       attributes {
         tag_width = 2 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
  ^bb0(%pa: !fabric.bits<32>):
    fabric.fu() -> (!fabric.bits<16>) {
      %k = fabric.op [@dataflow.constant] ()
           {sw_configs = {const_hex_value = "0xbeef"}}
           : () -> !fabric.bits<16>
      fabric.yield %k : !fabric.bits<16>
    }
    fabric.yield %pa : !fabric.bits<32>
  }
  fabric.yield
}

// -----
// Named temporal PE: yield value bits-width must equal the port
// bits-data-width (tag is reattached at the boundary, but the data
// part width must match exactly).
fabric.module @temp_named_yield_width_mismatch() {
  // expected-error @+1 {{yield value #0 bits-width 16 must equal port bits-data-width 32}}
  fabric.pe @TempBadYield [temporal] (!fabric.bits_tag<32, 4>)
                                      -> (!fabric.bits_tag<32, 4>)
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
  ^bb0(%pa: !fabric.bits<16>):
    fabric.fu(%fa = %pa : !fabric.bits<16>) -> (!fabric.bits<32>) {
      %k = fabric.op [@dataflow.constant] (%fa)
           {sw_configs = {const_hex_value = "0xdeadbeef"}}
           : (!fabric.bits<16>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
    fabric.yield %pa : !fabric.bits<16>
  }
  fabric.yield
}

// -----
// Anonymous temporal PE: inner block arg width exceeds the outer
// bits-data-width (truncation only narrows).
fabric.module @temp_anon_inner_too_wide(%a : !fabric.bits_tag<16, 2>) {
  // expected-error @+1 {{anonymous PE inner block arg #0 width (32) > outer bits part width (16) (truncation only narrows)}}
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<16, 2>
                                       to !fabric.bits<32>)
                            -> !fabric.bits_tag<16, 2>
       attributes {
         tag_width = 2 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<16>) {
      %k = fabric.op [@dataflow.constant] (%fa)
           {sw_configs = {const_hex_value = "0xbeef"}}
           : (!fabric.bits<32>) -> !fabric.bits<16>
      fabric.yield %k : !fabric.bits<16>
    }
  }
  fabric.yield
}
