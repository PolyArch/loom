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
