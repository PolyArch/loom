// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// fabric.s2t: 2-operand form with TW mismatch (operand[1] width != result tag width).
fabric.module @s2t_tag_width_mismatch(%d : !fabric.bits<32>, %t : !fabric.bits<3>) {
  // expected-error @+1 {{operand #1 bits-width 3 must equal result tag-width 4}}
  %0 = fabric.s2t %d, %t : (!fabric.bits<32>, !fabric.bits<3>) -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.s2t: 2-operand form with BW mismatch.
fabric.module @s2t_data_width_mismatch(%d : !fabric.bits<16>, %t : !fabric.bits<4>) {
  // expected-error @+1 {{operand #0 bits-width 16 must equal result data-width 32}}
  %0 = fabric.s2t %d, %t : (!fabric.bits<16>, !fabric.bits<4>) -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.s2t: 1-operand form missing sw_configs.tag.
fabric.module @s2t_const_missing_tag(%d : !fabric.bits<32>) {
  // expected-error @+1 {{constant-tag form requires 'sw_configs.tag' integer attribute}}
  %0 = fabric.s2t %d : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.s2t: 1-operand form sw_configs.tag width != result TW.
fabric.module @s2t_const_tag_width_bad(%d : !fabric.bits<32>) {
  // expected-error @+1 {{'sw_configs.tag' integer attribute width 8 must equal result tag-width 4}}
  %0 = fabric.s2t %d {sw_configs = {tag = 3 : i8}}
       : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.t2t: BW1 != BW2.
fabric.module @t2t_bw_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{operand data-width 32 must equal result data-width 16}}
  %0 = fabric.t2t %a
       {hw_params = [{lookup_table = [{input_tag = 0 : i4, output_tag = 0 : i4}]}]}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<16, 4>
  fabric.yield
}

// -----
// fabric.t2t: missing hw_params.
fabric.module @t2t_missing_hw_params(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{requires attribute 'hw_params'}}
  %0 = fabric.t2t %a : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.t2t: empty lookup_table.
fabric.module @t2t_empty_lut(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'lookup_table' must be non-empty}}
  %0 = fabric.t2t %a
       {hw_params = [{lookup_table = []}]}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.t2t: input_tag width mismatch.
fabric.module @t2t_input_tag_width_bad(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'input_tag' integer width 8 must equal operand tag-width 4}}
  %0 = fabric.t2t %a
       {hw_params = [{lookup_table = [{input_tag = 0 : i8, output_tag = 0 : i4}]}]}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.t2t: output_tag width mismatch.
fabric.module @t2t_output_tag_width_bad(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'output_tag' integer width 4 must equal result tag-width 8}}
  %0 = fabric.t2t %a
       {hw_params = [{lookup_table = [{input_tag = 0 : i4, output_tag = 0 : i4}]}]}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 8>
  fabric.yield
}

// -----
// fabric.t2t: duplicate input_tag values.
fabric.module @t2t_duplicate_keys(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'lookup_table' has duplicate 'input_tag' value 1}}
  %0 = fabric.t2t %a
       {hw_params = [{lookup_table = [{input_tag = 1 : i4, output_tag = 0 : i4},
                                       {input_tag = 1 : i4, output_tag = 2 : i4}]}]}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.t2t: hw_params is not length-1.
fabric.module @t2t_hw_params_bad_length(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'hw_params' must be a length-1 array wrapping a dictionary}}
  %0 = fabric.t2t %a
       {hw_params = []}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.t2s: 2-result form with data width mismatch.
fabric.module @t2s_split_data_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{result #0 bits-width 16 must equal operand data-width 32}}
  %d, %t = fabric.t2s %a : !fabric.bits_tag<32, 4> -> (!fabric.bits<16>, !fabric.bits<4>)
  fabric.yield
}

// -----
// fabric.t2s: 2-result form with tag width mismatch.
fabric.module @t2s_split_tag_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{result #1 bits-width 8 must equal operand tag-width 4}}
  %d, %t = fabric.t2s %a : !fabric.bits_tag<32, 4> -> (!fabric.bits<32>, !fabric.bits<8>)
  fabric.yield
}

// -----
// fabric.t2s: drop-tag form data width mismatch.
fabric.module @t2s_drop_data_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{result #0 bits-width 16 must equal operand data-width 32}}
  %d = fabric.t2s %a : !fabric.bits_tag<32, 4> -> !fabric.bits<16>
  fabric.yield
}

// -----
// fabric.s2t inside fabric.fu body: rejected by fu body whitelist.
// (s2t in constant-tag form so its operand has the FU's bits<W> type.)
fabric.module @s2t_in_fu(%d : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      // expected-error @+1 {{is not allowed inside fabric.fu}}
      %0 = fabric.s2t %fa {sw_configs = {tag = 0 : i4}}
           : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// fabric.s2t inside fabric.pe body (anonymous): rejected by pe body whitelist.
// (s2t in constant-tag form so the operand type matches the PE's uniform W.)
fabric.module @s2t_in_pe(%d : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    // expected-error @+1 {{'fabric.pe' op body may only contain fabric.fu and fabric.instantiate}}
    %0 = fabric.s2t %pa {sw_configs = {tag = 0 : i4}}
         : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}
