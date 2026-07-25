// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// fabric.boundary direction parse error: wrong keyword.
fabric.module @boundary_bad_direction(%d : !fabric.bits<32>, %t : !fabric.bits<4>) {
  // expected-error @+1 {{expected fabric boundary direction keyword 's2t', 't2t' or 't2s', got 'wibble'}}
  %0 = fabric.boundary [wibble] %d, %t : (!fabric.bits<32>, !fabric.bits<4>) -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [s2t]: 2-operand form with TW mismatch.
fabric.module @s2t_tag_width_mismatch(%d : !fabric.bits<32>, %t : !fabric.bits<3>) {
  // expected-error @+1 {{[s2t] operand #1 bits-width 3 must equal result tag-width 4}}
  %0 = fabric.boundary [s2t] %d, %t : (!fabric.bits<32>, !fabric.bits<3>) -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [s2t]: 2-operand form with BW mismatch.
fabric.module @s2t_data_width_mismatch(%d : !fabric.bits<16>, %t : !fabric.bits<4>) {
  // expected-error @+1 {{[s2t] operand #0 bits-width 16 must equal result data-width 32}}
  %0 = fabric.boundary [s2t] %d, %t : (!fabric.bits<16>, !fabric.bits<4>) -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [s2t]: the two-operand form has no configured projection.
fabric.module @s2t_general_with_sw_configs(%d : !fabric.bits<32>,
                                           %t : !fabric.bits<4>) {
  // expected-error @+1 {{[s2t] two-operand form must not carry 'sw_configs'}}
  %0 = fabric.boundary [s2t] %d, %t {sw_configs = {unexpected = 0 : i1}}
       : (!fabric.bits<32>, !fabric.bits<4>) -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// Memory capabilities cannot be declared as boundary destination ports.
fabric.module @boundary_memref_input_normalization(%a : !fabric.bits<32>) {
  // expected-error @+1 {{memref capabilities cannot use the 'to <destination-type>' clause}}
  %0 = fabric.boundary [s2t] %a {sw_configs = {tag = 0 : i4}}
       : !fabric.bits<32> to memref<4xi32> -> !fabric.bits_tag<16, 4>
  fabric.yield
}

// -----
// A same-name discardable attribute must not shadow a valid inherent property.
fabric.module @boundary_inner_input_types_collision(
    %a : !fabric.bits<32>) {
  // expected-error @+1 {{discardable attribute 'inner_input_types' conflicts with the inherent property of the same name}}
  %0 = "fabric.boundary"(%a) <{
    direction = 0 : i32,
    inner_input_types = [!fabric.bits<16>],
    sw_configs = {tag = 0 : i4}
  }> {inner_input_types = "not-an-array"}
      : (!fabric.bits<32>) -> !fabric.bits_tag<16, 4>
  fabric.yield
}

// -----
// A present wrong-typed sw_configs must not be normalized to absence.
fabric.module @boundary_sw_configs_wrong_outer_type(%d : !fabric.bits<32>) {
  // expected-error @+2 {{Invalid attribute `sw_configs` in property conversion}}
  // expected-error @+1 {{[s2t] present 'sw_configs' must contain exactly the 'tag' field}}
  %0 = fabric.boundary [s2t] %d {sw_configs = []}
       : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// Generic syntax must reject a malformed present inherent attribute instead
// of erasing it into the unconfigured form.
fabric.module @boundary_generic_sw_configs_wrong_type(%a : !fabric.bits<32>) {
  // expected-error @+2 {{Invalid attribute `sw_configs` in property conversion}}
  // expected-error @+1 {{[s2t] present 'sw_configs' must contain exactly the 'tag' field}}
  %0 = "fabric.boundary"(%a) <{
    direction = 0 : i32,
    inner_input_types = []
  }> {sw_configs = []}
      : (!fabric.bits<32>) -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [s2t]: a present empty projection is not the canonical
// unconfigured form.
fabric.module @s2t_empty_sw_configs(%d : !fabric.bits<32>) {
  // expected-error @+1 {{[s2t] present 'sw_configs' must contain exactly the 'tag' field}}
  %0 = fabric.boundary [s2t] %d {sw_configs = {}}
       : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [s2t]: the configured projection has exactly one field.
fabric.module @s2t_extra_sw_config_field(%d : !fabric.bits<32>) {
  // expected-error @+1 {{[s2t] present 'sw_configs' must contain exactly the 'tag' field}}
  %0 = fabric.boundary [s2t] %d
       {sw_configs = {tag = 3 : i4, unexpected = 0 : i1}}
       : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [s2t]: sw_configs.tag must be an IntegerAttr.
fabric.module @s2t_tag_wrong_type(%d : !fabric.bits<32>) {
  // expected-error @+1 {{[s2t] 'sw_configs.tag' must be an IntegerAttr}}
  %0 = fabric.boundary [s2t] %d {sw_configs = {tag = "three"}}
       : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [s2t]: 1-operand form sw_configs.tag width != result TW.
fabric.module @s2t_const_tag_width_bad(%d : !fabric.bits<32>) {
  // expected-error @+1 {{[s2t] 'sw_configs.tag' integer attribute width 8 must equal result tag-width 4}}
  %0 = fabric.boundary [s2t] %d {sw_configs = {tag = 3 : i8}}
       : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [s2t]: negative tag literal rejected.
// (The signed integer type `si4` makes the negative literal observable;
// signless `i4` is normalized to a bit-pattern at parse time and so
// cannot syntactically distinguish `-1 : i4` from `15 : i4`.)
fabric.module @s2t_negative_tag(%d : !fabric.bits<32>) {
  // expected-error @+1 {{'sw_configs.tag' must be a non-negative integer literal}}
  %0 = fabric.boundary [s2t] %d {sw_configs = {tag = -1 : si4}}
       : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [s2t]: hw_params present (must be absent).
fabric.module @s2t_with_hw_params(%d : !fabric.bits<32>, %t : !fabric.bits<4>) {
  // expected-error @+1 {{[s2t] must not carry 'hw_params'}}
  %0 = fabric.boundary [s2t] %d, %t {hw_params = [{junk = 1 : i32}]}
       : (!fabric.bits<32>, !fabric.bits<4>) -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: BW1 != BW2.
fabric.module @t2t_bw_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] operand data-width 32 must equal result data-width 16}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [{src_tag = 0 : i4, dst_tag = 0 : i4}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<16, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: missing hw_params.
fabric.module @t2t_missing_hw_params(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] requires 'hw_params' attribute carrying 'lut_size'}}
  %0 = fabric.boundary [t2t] %a : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: a present empty projection is not the canonical
// unconfigured form.
fabric.module @t2t_empty_sw_configs(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] present 'sw_configs' must contain exactly the 'lookup_table' field}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}], sw_configs = {}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: a present lookup table must be nonempty.
fabric.module @t2t_empty_lookup_table(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] a present 'lookup_table' must be nonempty}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = []}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: the configured projection has exactly one field.
fabric.module @t2t_extra_sw_config_field(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] present 'sw_configs' must contain exactly the 'lookup_table' field}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [{src_tag = 0 : i4, dst_tag = 0 : i4}],
                      unexpected = 0 : i1}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: each configured entry has exactly two fields.
fabric.module @t2t_extra_lookup_entry_field(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] 'lookup_table' entry #0 must contain exactly 'src_tag' and 'dst_tag'}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [{src_tag = 0 : i4, dst_tag = 0 : i4,
                                       unexpected = 0 : i1}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: entry tags must be IntegerAttrs.
fabric.module @t2t_src_tag_wrong_type(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] 'lookup_table' entry #0 'src_tag'/'dst_tag' must be IntegerAttr}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [{src_tag = "zero", dst_tag = 0 : i4}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: src_tag width mismatch.
fabric.module @t2t_src_tag_width_bad(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] 'lookup_table' entry #0 'src_tag' integer width 8 must equal operand tag-width 4}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [{src_tag = 0 : i8, dst_tag = 0 : i4}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: dst_tag width mismatch.
fabric.module @t2t_dst_tag_width_bad(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] 'lookup_table' entry #0 'dst_tag' integer width 4 must equal result tag-width 8}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [{src_tag = 0 : i4, dst_tag = 0 : i4}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 8>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: duplicate src_tag values wider than 64 bits.
fabric.module @t2t_wide_duplicate_keys(%a : !fabric.bits_tag<32, 80>) {
  // expected-error @+1 {{[t2t] duplicate src_tag value}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [
          {src_tag = 18446744073709551617 : i80, dst_tag = 0 : i80},
          {src_tag = 18446744073709551617 : i80, dst_tag = 2 : i80}
        ]}}
       : !fabric.bits_tag<32, 80> -> !fabric.bits_tag<32, 80>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: hw_params is not length-1.
fabric.module @t2t_hw_params_bad_length(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] 'hw_params' must be a length-1 array wrapping a dictionary}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [],
        sw_configs = {lookup_table = [{src_tag = 0 : i4, dst_tag = 0 : i4}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: lookup_table.size() > lut_size.
fabric.module @t2t_lut_overflow(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2t] 'lookup_table' has more LUT entries than declared lut_size: 3 > 2}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 2 : i32}],
        sw_configs = {lookup_table = [{src_tag = 0 : i4, dst_tag = 0 : i4},
                                       {src_tag = 1 : i4, dst_tag = 1 : i4},
                                       {src_tag = 2 : i4, dst_tag = 2 : i4}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: negative src_tag literal (signed type so the sign
// is observable post-parse).
fabric.module @t2t_negative_src_tag(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'lookup_table' entry #0 has negative src_tag literal}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [{src_tag = -1 : si4, dst_tag = 0 : i4}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2t]: negative dst_tag literal (signed type so the sign
// is observable post-parse).
fabric.module @t2t_negative_dst_tag(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'lookup_table' entry #0 has negative dst_tag literal}}
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [{src_tag = 0 : i4, dst_tag = -1 : si4}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----
// fabric.boundary [t2s]: 2-result form with data width mismatch.
fabric.module @t2s_split_data_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2s] result #0 bits-width 16 must equal operand data-width 32}}
  %d, %t = fabric.boundary [t2s] %a : !fabric.bits_tag<32, 4> -> (!fabric.bits<16>, !fabric.bits<4>)
  fabric.yield
}

// -----
// fabric.boundary [t2s]: 2-result form with tag width mismatch.
fabric.module @t2s_split_tag_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2s] result #1 bits-width 8 must equal operand tag-width 4}}
  %d, %t = fabric.boundary [t2s] %a : !fabric.bits_tag<32, 4> -> (!fabric.bits<32>, !fabric.bits<8>)
  fabric.yield
}

// -----
// fabric.boundary [t2s]: drop-tag form data width mismatch.
fabric.module @t2s_drop_data_mismatch(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2s] result #0 bits-width 16 must equal operand data-width 32}}
  %d = fabric.boundary [t2s] %a : !fabric.bits_tag<32, 4> -> !fabric.bits<16>
  fabric.yield
}

// -----
// fabric.boundary [t2s]: hw_params present (must be absent).
fabric.module @t2s_with_hw_params(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2s] must not carry 'hw_params'}}
  %d = fabric.boundary [t2s] %a {hw_params = [{junk = 1 : i32}]}
       : !fabric.bits_tag<32, 4> -> !fabric.bits<32>
  fabric.yield
}

// -----
// fabric.boundary [t2s]: sw_configs present (must be absent).
fabric.module @t2s_with_sw_configs(%a : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{[t2s] must not carry 'sw_configs'}}
  %d = fabric.boundary [t2s] %a {sw_configs = {junk = 1 : i32}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits<32>
  fabric.yield
}

// -----
// fabric.boundary inside fabric.fu body: rejected by fu body whitelist.
// (s2t in constant-tag form so its operand has the FU's bits<W> type.)
fabric.module @boundary_in_fu(%d : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      // expected-error @+1 {{is not allowed inside fabric.fu}}
      %0 = fabric.boundary [s2t] %fa {sw_configs = {tag = 0 : i4}}
           : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// fabric.boundary inside fabric.pe body (anonymous): rejected by pe body whitelist.
fabric.module @boundary_in_pe(%d : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    // expected-error @+1 {{'fabric.pe' op body may only contain fabric.fu and fabric.instantiate}}
    %0 = fabric.boundary [s2t] %pa {sw_configs = {tag = 0 : i4}}
         : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}
