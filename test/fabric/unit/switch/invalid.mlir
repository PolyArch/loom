// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Bad schedule keyword.
fabric.module @sw_bad_schedule(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{expected fabric switch schedule keyword 'spatial' or 'temporal', got 'wibble'}}
  %o:2 = fabric.switch [wibble] %a, %b
         [{connectivity_table = ["11", "11"]}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// Schedule + port type-kind mismatch (spatial schedule with bits_tag ports).
fabric.module @sw_spatial_with_bits_tag(%a : !fabric.bits_tag<32, 4>,
                                        %b : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{schedule mismatch with port kind: spatial fabric.switch requires '!fabric.bits<W>' ports}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11", "11"]}]
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----
// Schedule + port type-kind mismatch (temporal schedule with bits ports).
fabric.module @sw_temporal_with_bits(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{schedule mismatch with port kind: temporal fabric.switch requires '!fabric.bits_tag<W, T>' ports}}
  %o:2 = fabric.switch [temporal] %a, %b
         [{connectivity_table = ["11", "11"], route_table_size = 1 : i32}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// Non-uniform widths on spatial ports.
fabric.module @sw_spatial_nonuniform(%a : !fabric.bits<32>, %b : !fabric.bits<16>) {
  // expected-error @+1 {{requires uniform 'bits<W>' on all switch ports}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11", "11"]}]
         : (!fabric.bits<32>, !fabric.bits<16>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// Non-uniform tag widths on temporal ports.
fabric.module @sw_temporal_nonuniform_tag(%a : !fabric.bits_tag<32, 4>,
                                          %b : !fabric.bits_tag<32, 3>) {
  // expected-error @+1 {{requires uniform 'bits_tag<W, T>' on all switch ports}}
  %o:2 = fabric.switch [temporal] %a, %b
         [{connectivity_table = ["11", "11"], route_table_size = 1 : i32}]
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 3>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----
// Missing connectivity_table.
fabric.module @sw_missing_conn(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{requires 'hw_params' with 'connectivity_table'}}
  %o:2 = fabric.switch [spatial] %a, %b
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// connectivity_table row length doesn't match K.
fabric.module @sw_conn_row_len(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{'connectivity_table' row #0 length 3 must equal K (2)}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["111", "11"]}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// connectivity_table length doesn't match L.
fabric.module @sw_conn_table_len(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{'connectivity_table' length 1 must equal L (2)}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11"]}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// connectivity_table illegal character.
fabric.module @sw_conn_bad_char(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{'connectivity_table' row #0 contains non-'0'/'1' character}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["1x", "11"]}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// connectivity_table row missing '1'.
fabric.module @sw_conn_row_zero(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{'connectivity_table' row #0 must have at least one '1' (each output needs at least one physical input source)}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["00", "11"]}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// connectivity_table column missing '1'.
fabric.module @sw_conn_col_zero(%a : !fabric.bits<32>, %b : !fabric.bits<32>,
                                %c : !fabric.bits<32>) {
  // expected-error @+1 {{'connectivity_table' column #1 must have at least one '1' (each input needs at least one physical destination)}}
  %o:2 = fabric.switch [spatial] %a, %b, %c
         [{connectivity_table = ["101", "001"]}]
         : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// Spatial route_table per-row '1' count > 1.
fabric.module @sw_spatial_fanin(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{spatial route_table row has '1' count > 1}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11", "11"]}]
         {route_table = ["11", "10"], switch_enable = true}
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// route_table row length mismatch (must equal '1' count of conn row).
fabric.module @sw_route_row_len(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{'route_table' row #0 length 3 must equal '1'-count of connectivity_table row #0 (2)}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11", "11"]}]
         {route_table = ["100", "01"], switch_enable = true}
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// All-or-nothing: route_table present but switch_enable absent.
fabric.module @sw_aon_violation(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{all-or-nothing violation: 'route_table' is present but 'switch_enable' is missing}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11", "11"]}]
         {route_table = ["10", "01"]}
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// All-or-nothing: switch_enable present but route_table absent.
fabric.module @sw_aon_violation2(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{all-or-nothing violation: 'switch_enable' is present but 'route_table' is missing}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11", "11"]}]
         {switch_enable = true}
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// Spatial PE may not carry temporal-only attribute (route_table_size).
fabric.module @sw_spatial_with_temporal_attr(%a : !fabric.bits<32>,
                                             %b : !fabric.bits<32>) {
  // expected-error @+1 {{spatial fabric.switch must not carry temporal-only attribute 'route_table_size'}}
  %o:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11", "11"], route_table_size = 4 : i32}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----
// Temporal: missing route_table_size.
fabric.module @sw_temporal_no_rts(%a : !fabric.bits_tag<32, 4>,
                                  %b : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{temporal fabric.switch requires 'route_table_size' attribute}}
  %o:2 = fabric.switch [temporal] %a, %b
         [{connectivity_table = ["11", "11"]}]
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----
// Temporal: route_table_size = 0.
fabric.module @sw_temporal_rts_zero(%a : !fabric.bits_tag<32, 4>,
                                    %b : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'route_table_size' must be >= 1}}
  %o:2 = fabric.switch [temporal] %a, %b
         [{connectivity_table = ["11", "11"], route_table_size = 0 : i32}]
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----
// Temporal: route_table length != route_table_size.
fabric.module @sw_temporal_rt_len(%a : !fabric.bits_tag<32, 4>,
                                  %b : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'route_table' length 1 must equal 'route_table_size' (2)}}
  %o:2 = fabric.switch [temporal] %a, %b
         [{connectivity_table = ["11", "11"], route_table_size = 2 : i32}]
         {
           route_table = [
             {route_sel = ["10", "01"], tag = 0 : i4, valid = true}
           ],
           switch_enable = true
         }
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----
// Temporal: duplicate valid tags.
fabric.module @sw_temporal_dup_tag(%a : !fabric.bits_tag<32, 4>,
                                   %b : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{temporal duplicate valid tag value 5}}
  %o:2 = fabric.switch [temporal] %a, %b
         [{connectivity_table = ["11", "11"], route_table_size = 2 : i32}]
         {
           route_table = [
             {route_sel = ["10", "01"], tag = 5 : i4, valid = true},
             {route_sel = ["01", "10"], tag = 5 : i4, valid = true}
           ],
           switch_enable = true
         }
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----
// Temporal: tag width != T.
fabric.module @sw_temporal_tag_width(%a : !fabric.bits_tag<32, 4>,
                                     %b : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{'tag' integer width 8 must equal port tag-width 4}}
  %o:2 = fabric.switch [temporal] %a, %b
         [{connectivity_table = ["11", "11"], route_table_size = 1 : i32}]
         {
           route_table = [
             {route_sel = ["10", "01"], tag = 5 : i8, valid = true}
           ],
           switch_enable = true
         }
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----
// Temporal: route_sel per-row '1' count > 1.
fabric.module @sw_temporal_routesel_fanin(%a : !fabric.bits_tag<32, 4>,
                                          %b : !fabric.bits_tag<32, 4>) {
  // expected-error @+1 {{spatial route_table row has '1' count > 1}}
  %o:2 = fabric.switch [temporal] %a, %b
         [{connectivity_table = ["11", "11"], route_table_size = 1 : i32}]
         {
           route_table = [
             {route_sel = ["11", "01"], tag = 0 : i4, valid = true}
           ],
           switch_enable = true
         }
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----
// Anonymous form with sym_name attribute is rejected by named/anonymous dichotomy.
// (Parser writes sym_name when '@' is seen, so we test mismatched empty-FT named form.)
fabric.module @sw_named_no_ft() {
  // expected-error @+1 {{requires 'hw_params' with 'connectivity_table'}}
  fabric.switch @NoConn [spatial] (!fabric.bits<32>) -> (!fabric.bits<32>)
  fabric.yield
}

// -----
// K = 0 (no inputs).
fabric.module @sw_no_inputs() {
  // expected-error @+1 {{requires at least 1 input port (K >= 1)}}
  fabric.switch @ZeroIn [spatial] () -> (!fabric.bits<32>)
         [{connectivity_table = ["1"]}]
  fabric.yield
}

// -----
// L = 0 (no outputs).
fabric.module @sw_no_outputs() {
  // expected-error @+1 {{requires at least 1 output port (L >= 1)}}
  fabric.switch @ZeroOut [spatial] (!fabric.bits<32>) -> ()
         [{connectivity_table = []}]
  fabric.yield
}

// -----
// Named form must not have SSA operands. The parser switches into the
// named branch on '@', then expects '(' for the function-type signature;
// an SSA operand at that point is rejected at parse time.
fabric.module @sw_named_bad_syntax(%a : !fabric.bits<32>) {
  // expected-error @+1 {{expected '('}}
  %o = fabric.switch @BadSw [spatial] %a [{connectivity_table = ["1"]}] : (!fabric.bits<32>) -> !fabric.bits<32>
  fabric.yield
}
