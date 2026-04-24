// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// op_list cannot be empty.
func.func @op_empty_list(%a: !fabric.bits<32>) -> !fabric.bits<32> {
  // expected-error @+1 {{'op_list' must be non-empty}}
  %0 = fabric.op [] (%a) : (!fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// Unknown op symbol.
func.func @op_unknown_symbol(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  // expected-error @+1 {{is not a fabric.op-supported software op}}
  %0 = fabric.op [@arith.no_such_op] (%a, %b)
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// arith.constant is explicitly disallowed (constants must come from
// fabric.op[@dataflow.constant]).
func.func @op_rejects_arith_constant(%a: !fabric.bits<32>) -> !fabric.bits<32> {
  // expected-error @+1 {{is not a fabric.op-supported software op}}
  %0 = fabric.op [@arith.constant] (%a)
       : (!fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// Two singleton ops cannot share a fabric.op.
func.func @op_two_singletons(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  // expected-error @+1 {{is not in any multi-member hardware-share group}}
  %0 = fabric.op [@arith.muli, @arith.addf] (%a, %b)
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// Two ops from different groups cannot share a fabric.op.
func.func @op_different_groups(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  // expected-error @+1 {{do not belong to the same hardware-share group}}
  %0 = fabric.op [@arith.addi, @arith.divsi] (%a, %b)
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// Multi-op programmed without op_sel.
func.func @op_missing_op_sel(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  // expected-error @+1 {{'sw_configs' must contain key 'op_sel'}}
  %0 = fabric.op [@arith.addi, @arith.subi] (%a, %b)
       {sw_configs = {something = "else"}}
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// op_sel value not in op_list.
func.func @op_bad_op_sel(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  // expected-error @+1 {{'sw_configs.op_sel' value "arith.muli" is not one of the symbols listed in 'op_list'}}
  %0 = fabric.op [@arith.addi, @arith.subi] (%a, %b)
       {sw_configs = {op_sel = "arith.muli"}}
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// Mismatched port count.
func.func @op_bad_port_count(%a: !fabric.bits<32>) -> !fabric.bits<32> {
  // expected-error @+1 {{port count (1->1) does not match the supported software ops (2->1)}}
  %0 = fabric.op [@arith.addi] (%a)
       : (!fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// Wrong fixed-port width: dataflow.stream's rwc port must be bits<1> not bits<0>.
func.func @op_stream_bad_rwc(%lb: !fabric.bits<32>, %ub: !fabric.bits<32>, %step: !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<0>) {
  // expected-error @+1 {{output port #1 has width 0 but software op(s) require width 1}}
  %i, %r = fabric.op [@dataflow.stream] (%lb, %ub, %step)
           : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<0>)
  return %i, %r : !fabric.bits<32>, !fabric.bits<0>
}

// -----
// dataflow.constant must have bits<0> input (none-typed ctrl).
func.func @op_constant_bad_ctrl(%ctrl: !fabric.bits<1>) -> !fabric.bits<32> {
  // expected-error @+1 {{input port #0 has width 1 but software op(s) require width 0}}
  %0 = fabric.op [@dataflow.constant] (%ctrl)
       {sw_configs = {const_hex_value = "0x1"}}
       : (!fabric.bits<1>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// hw_params must be a length-1 array.
func.func @op_bad_hw_params(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  // expected-error @+1 {{'hw_params' must be a length-1 array}}
  %0 = fabric.op [@arith.muli] (%a, %b)
       {hw_params = [{}, {}]}
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// Inconsistent type-parameter widths: dataflow.stream's three input ports must
// all share one width.
func.func @op_stream_inconsistent_t(%lb: !fabric.bits<32>, %ub: !fabric.bits<64>, %step: !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<1>) {
  // expected-error @+1 {{requires the same width on all ports tied to its type parameter}}
  %i, %r = fabric.op [@dataflow.stream] (%lb, %ub, %step)
           : (!fabric.bits<32>, !fabric.bits<64>, !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<1>)
  return %i, %r : !fabric.bits<32>, !fabric.bits<1>
}

// -----
// dataflow.sync: in/out counts must match.
func.func @op_sync_unequal_counts(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
    -> !fabric.bits<32> {
  // expected-error @+1 {{@dataflow.sync requires equal input/output counts}}
  %0 = fabric.op [@dataflow.sync] (%a, %b)
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----
// dataflow.sync: bitmask length must equal port count.
func.func @op_sync_bad_bitmask_len(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                                    %c: !fabric.bits<32>, %d: !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
  // expected-error @+1 {{'sw_configs.bitmask' length (3) must equal port count (4)}}
  %w, %x, %y, %z = fabric.op [@dataflow.sync] (%a, %b, %c, %d)
                   {sw_configs = {bitmask = "101"}}
                   : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                     -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  return %w, %x, %y, %z : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
}

// -----
// dataflow.sync: bitmask must contain only '0' / '1'.
func.func @op_sync_bad_bitmask_chars(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>) {
  // expected-error @+1 {{'sw_configs.bitmask' must contain only '0' and '1'}}
  %x, %y = fabric.op [@dataflow.sync] (%a, %b)
           {sw_configs = {bitmask = "1x"}}
           : (!fabric.bits<32>, !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<32>)
  return %x, %y : !fabric.bits<32>, !fabric.bits<32>
}

// -----
// dataflow.mux with 2 data inputs requires bits<1> sel.
func.func @op_mux2_bad_sel(%sel: !fabric.bits<32>, %a: !fabric.bits<16>, %b: !fabric.bits<16>)
    -> !fabric.bits<16> {
  // expected-error @+1 {{sel port (input #0) width 32 must be 1}}
  %0 = fabric.op [@dataflow.mux] (%sel, %a, %b)
       : (!fabric.bits<32>, !fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
  return %0 : !fabric.bits<16>
}

// -----
// dataflow.mux with >2 data inputs requires sel width = index width (default 32).
func.func @op_mux3_bad_sel(%sel: !fabric.bits<1>,
                            %a: !fabric.bits<16>, %b: !fabric.bits<16>, %c: !fabric.bits<16>)
    -> !fabric.bits<16> {
  // expected-error @+1 {{sel port (input #0) width 1 must be 32}}
  %0 = fabric.op [@dataflow.mux] (%sel, %a, %b, %c)
       : (!fabric.bits<1>, !fabric.bits<16>, !fabric.bits<16>, !fabric.bits<16>)
         -> !fabric.bits<16>
  return %0 : !fabric.bits<16>
}

// -----
// dataflow.demux with 2 outs requires bits<1> sel.
func.func @op_demux2_bad_sel(%sel: !fabric.bits<32>, %in: !fabric.bits<8>)
    -> (!fabric.bits<8>, !fabric.bits<8>) {
  // expected-error @+1 {{sel port (input #0) width 32 must be 1}}
  %a, %b = fabric.op [@dataflow.demux] (%sel, %in)
           : (!fabric.bits<32>, !fabric.bits<8>)
             -> (!fabric.bits<8>, !fabric.bits<8>)
  return %a, %b : !fabric.bits<8>, !fabric.bits<8>
}

// -----
// dataflow.mux: data inputs must match output width.
func.func @op_mux_data_mismatch(%sel: !fabric.bits<1>, %a: !fabric.bits<16>, %b: !fabric.bits<32>)
    -> !fabric.bits<16> {
  // expected-error @+1 {{@dataflow.mux input #2 width 32 must match output width 16}}
  %0 = fabric.op [@dataflow.mux] (%sel, %a, %b)
       : (!fabric.bits<1>, !fabric.bits<16>, !fabric.bits<32>) -> !fabric.bits<16>
  return %0 : !fabric.bits<16>
}

// -----
// hw_params allowed-set check: sw_configs value not in hw_params allowed array.
func.func @op_sw_value_not_in_hw_set(%lb: !fabric.bits<32>, %ub: !fabric.bits<32>,
                                      %step: !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<1>) {
  // expected-error @+1 {{'sw_configs["step_op"]' value "%=" is not in the 'hw_params["step_op"]' allowed set}}
  %i, %r = fabric.op [@dataflow.stream] (%lb, %ub, %step)
           {hw_params = [{step_op = ["+=", "/="], cont_cond = ["<", ">"]}],
            sw_configs = {step_op = "%=", cont_cond = "<"}}
           : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<1>)
  return %i, %r : !fabric.bits<32>, !fabric.bits<1>
}

// -----
// hw_params allowed-set check: hw value for shared key must be ArrayAttr.
func.func @op_hw_value_not_array(%lb: !fabric.bits<32>, %ub: !fabric.bits<32>,
                                  %step: !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<1>) {
  // expected-error @+1 {{'hw_params["step_op"]' must be an array of allowed values}}
  %i, %r = fabric.op [@dataflow.stream] (%lb, %ub, %step)
           {hw_params = [{step_op = "+="}],
            sw_configs = {step_op = "+=", cont_cond = "<"}}
           : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<1>)
  return %i, %r : !fabric.bits<32>, !fabric.bits<1>
}
