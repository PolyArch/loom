// RUN: loom %s | loom | FileCheck %s

// -----------------------------------------------------------------------------
// Minimal hw-only temporal PE (anonymous form): K=1, L=1, single inner FU,
// no reg FIFO, per-instruction operand buffer.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @temp_min
fabric.module @temp_min(%a : !fabric.bits_tag<32, 4>)
                       -> (!fabric.bits_tag<32, 4>) {
  // CHECK: fabric.pe [temporal]
  // CHECK-SAME: !fabric.bits_tag<32, 4>
  // CHECK: operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>
  // CHECK-SAME: operand_buffer_size = 2 : i32
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
         operand_buffer_size = 2 : i32
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield %r : !fabric.bits_tag<32, 4>
}

// -----------------------------------------------------------------------------
// The same dedicated organization at depth 1. The depth is canonical Fabric
// content, so this PE is not the one above.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @temp_min_depth_one
fabric.module @temp_min_depth_one(%a : !fabric.bits_tag<32, 4>)
                                 -> (!fabric.bits_tag<32, 4>) {
  // CHECK: operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>
  // CHECK-SAME: operand_buffer_size = 1 : i32
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
         operand_buffer_size = 1 : i32
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield %r : !fabric.bits_tag<32, 4>
}

// -----------------------------------------------------------------------------
// Hw-only temporal PE with a reg FIFO bank (num_reg_fifo > 0).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @temp_with_regs
fabric.module @temp_with_regs(%a : !fabric.bits_tag<16, 3>,
                              %b : !fabric.bits_tag<16, 3>)
                             -> (!fabric.bits_tag<16, 3>) {
  // CHECK: fabric.pe [temporal]
  // CHECK: operand_buffer_mode = #fabric.operand_buffer_mode<per_input_port>
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<16, 3>,
                             %pb = %b : !fabric.bits_tag<16, 3>)
                            -> !fabric.bits_tag<16, 3>
       attributes {
         tag_width = 3 : i32,
         num_instruction = 4 : i32,
         num_reg_fifo = 4 : i32,
         reg_fifo_depth = 8 : i32,
         reg_fifo_ports = 2 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = #fabric.operand_buffer_mode<per_input_port>,
         operand_buffer_size = 4 : i32
       } {
    fabric.fu(%fa = %pa : !fabric.bits<16>,
              %fb = %pb : !fabric.bits<16>) -> (!fabric.bits<16>) {
      %v = fabric.op [@arith.addi] (%fa, %fb)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      fabric.yield %v : !fabric.bits<16>
    }
  }
  fabric.yield %r : !fabric.bits_tag<16, 3>
}

// -----------------------------------------------------------------------------
// Hw-only temporal PE with operand_buffer_mode = all_fu_share.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @temp_share_buffer
fabric.module @temp_share_buffer(%a : !fabric.bits_tag<8, 2>)
                                -> (!fabric.bits_tag<8, 2>) {
  // CHECK: fabric.pe [temporal]
  // CHECK: operand_buffer_mode = #fabric.operand_buffer_mode<all_fu_share>
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<8, 2>)
                            -> !fabric.bits_tag<8, 2>
       attributes {
         tag_width = 2 : i32,
         num_instruction = 2 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = #fabric.operand_buffer_mode<all_fu_share>,
         operand_buffer_size = 16 : i32
       } {
    fabric.fu(%fa = %pa : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
  }
  fabric.yield %r : !fabric.bits_tag<8, 2>
}

// -----------------------------------------------------------------------------
// Programmed temporal PE: per_instruction_fu_config mode, full sw_configs
// trio. num_instruction = 2, 1 inner FU (so opcode width is 0 but the
// attribute is still required as 0 : i1).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @temp_programmed
fabric.module @temp_programmed(%a : !fabric.bits_tag<32, 4>,
                               %b : !fabric.bits_tag<32, 4>)
                              -> (!fabric.bits_tag<32, 4>) {
  // CHECK: fabric.pe [temporal]
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>,
                             %pb = %b : !fabric.bits_tag<32, 4>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 2 : i32,
         fu_config_mode = "per_instruction_fu_config",
         operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
         operand_buffer_size = 2 : i32,
         pe_enable = true,
         instruction_mem = [
           {
             enable = true,
             opcode = 0 : i1,
             operand_sel = [
               { src_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = false },
               { src_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = true }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 1 : i4, is_port = true,
                 discard = false, disconnect = false }
             ],
             fu_sw_configs = { dummy = 0 : i32 }
           },
           {
             enable = false,
             opcode = 0 : i1,
             operand_sel = [
               { src_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = true },
               { src_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = true }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 0 : i4, is_port = true,
                 discard = false, disconnect = true }
             ],
             fu_sw_configs = {}
           }
         ]
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fb)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield %r : !fabric.bits_tag<32, 4>
}

// -----------------------------------------------------------------------------
// Programmed temporal PE: per_fu_config mode. Two inner FUs (so opcode
// width = 1 bit). Reg FIFO bank present (single-port). One instruction
// uses a reg-fifo source.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @temp_per_fu
fabric.module @temp_per_fu(%a : !fabric.bits_tag<16, 2>,
                           %b : !fabric.bits_tag<16, 2>)
                          -> (!fabric.bits_tag<16, 2>) {
  // CHECK: fabric.pe [temporal]
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<16, 2>,
                             %pb = %b : !fabric.bits_tag<16, 2>)
                            -> !fabric.bits_tag<16, 2>
       attributes {
         tag_width = 2 : i32,
         num_instruction = 1 : i32,
         num_reg_fifo = 2 : i32,
         reg_fifo_depth = 4 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = #fabric.operand_buffer_mode<per_input_port>,
         operand_buffer_size = 2 : i32,
         pe_enable = true,
         instruction_mem = [
           {
             enable = true,
             opcode = 1 : i1,
             operand_sel = [
               { src_sel = 0 : i32, tag = 0 : i2, is_port = true,
                 discard = false, disconnect = false },
               { src_sel = 1 : i32, tag = 0 : i2, is_port = false,
                 discard = false, disconnect = false }
             ],
             result_sel = [
               { dst_sel = 0 : i32, tag = 0 : i2, is_port = false,
                 discard = false, disconnect = false }
             ]
           }
         ],
         per_fu_sw_configs = [
           { dummy = 0 : i32 },
           { dummy = 0 : i32 }
         ]
       } {
    fabric.fu(%fa = %pa : !fabric.bits<16>) -> (!fabric.bits<16>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      fabric.yield %v : !fabric.bits<16>
    }
    fabric.fu(%fa = %pa : !fabric.bits<16>,
              %fb = %pb : !fabric.bits<16>) -> (!fabric.bits<16>) {
      %v = fabric.op [@arith.muli] (%fa, %fb)
           {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      fabric.yield %v : !fabric.bits<16>
    }
  }
  fabric.yield %r : !fabric.bits_tag<16, 2>
}

// -----------------------------------------------------------------------------
// Named template form for a temporal PE. `function_type` alone owns the
// result ports, so the body closes with a zero-operand fabric.yield.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @temp_named_host
// CHECK: fabric.pe @TempPe [temporal]
// CHECK: fabric.yield %{{.*}} : !fabric.bits<32>
// CHECK-NEXT: }
// CHECK-NEXT: fabric.yield{{[[:space:]]*$}}
fabric.module @temp_named_host() {
  fabric.pe @TempPe [temporal] (!fabric.bits_tag<32, 4>)
                                -> (!fabric.bits_tag<32, 4>)
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
         operand_buffer_size = 2 : i32
       } {
  ^bb0(%pa: !fabric.bits<32>):
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    fabric.yield
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Anonymous temporal PE without an explicit `to <inner-type>` clause:
// the implicit boundary tag-strip exposes !fabric.bits<W> to the body.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @temp_implicit_strip
fabric.module @temp_implicit_strip(%a : !fabric.bits_tag<32, 4>)
                                  -> (!fabric.bits_tag<32, 4>) {
  // CHECK: fabric.pe [temporal]
  // The printer must omit the implicit `to bits<32>` default.
  // CHECK-NOT: to !fabric.bits<32>
  %r = fabric.pe [temporal] (%pa = %a : !fabric.bits_tag<32, 4>)
                            -> !fabric.bits_tag<32, 4>
       attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
         operand_buffer_size = 2 : i32
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield %r : !fabric.bits_tag<32, 4>
}

// -----------------------------------------------------------------------------
// Named temporal PE with entry block args at full bits<W> width (no `to`
// syntax in named form; user writes the bits<W> type directly in ^bb0).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @temp_named_full_width
// CHECK: fabric.pe @TempFull [temporal]
fabric.module @temp_named_full_width() {
  fabric.pe @TempFull [temporal] (!fabric.bits_tag<16, 2>,
                                  !fabric.bits_tag<16, 2>)
                                  -> (!fabric.bits_tag<16, 2>)
       attributes {
         tag_width = 2 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
         operand_buffer_size = 2 : i32
       } {
  ^bb0(%pa: !fabric.bits<16>, %pb: !fabric.bits<16>):
    fabric.fu(%fa = %pa : !fabric.bits<16>,
              %fb = %pb : !fabric.bits<16>) -> (!fabric.bits<16>) {
      %v = fabric.op [@arith.addi] (%fa, %fb)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      fabric.yield %v : !fabric.bits<16>
    }
    fabric.yield
  }
  fabric.yield
}
