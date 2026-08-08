// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// fabric.module requires a sym_name; the parser rejects an anonymous module.
// expected-error @+1 {{expected valid '@'-identifier for symbol name}}
fabric.module {
}

// -----
// fabric.module body must be a single block.
// expected-error @below {{0 or 1 blocks}}
fabric.module @m_multi_block() {
  cf.br ^bb1
^bb1:
  fabric.yield
}

// -----
// Yield value count must match the declared module-result count.
fabric.module @m_yield_count_mismatch_extra(%a : !fabric.bits<32>) {
  // expected-error @+1 {{yield value count (1) must match parent fabric.module result count (0)}}
  fabric.yield %a : !fabric.bits<32>
}

// -----
// Yield value count too few: declared two results but yields none.
fabric.module @m_yield_count_mismatch_missing()
    -> (!fabric.bits<32>, !fabric.bits<32>) {
  // expected-error @+1 {{yield value count (0) must match parent fabric.module result count (2)}}
  fabric.yield
}

// -----
// A registered declared_types property must describe every yielded value.
fabric.module @m_yield_declared_types_count(%a : !fabric.bits<32>)
    -> (!fabric.bits<32>) {
  // expected-error @+1 {{'declared_types' count (0) must match yield value count (1)}}
  "fabric.yield"(%a) <{declared_types = []}>
      : (!fabric.bits<32>) -> ()
}

// -----
// Every member of declared_types must be a TypeAttr.
fabric.module @m_yield_declared_types_member(%a : !fabric.bits<32>)
    -> (!fabric.bits<32>) {
  // expected-error @+1 {{'declared_types' member #0 must be a type attribute}}
  "fabric.yield"(%a) <{declared_types = [32 : i32]}>
      : (!fabric.bits<32>) -> ()
}

// -----
// Yield type-kind mismatch: yielding a `bits_tag` value for a `bits` result.
fabric.module @m_yield_kind_mismatch(%a : !fabric.bits_tag<8, 2>)
    -> (!fabric.bits<32>) {
  // expected-error @+1 {{has a different fabric kind than the module result type}}
  fabric.yield %a : !fabric.bits_tag<8, 2>
}

// -----
// A manager capability imported through the module signature cannot become a
// subordinate capability merely because the exported memref type is equal.
fabric.module @m_manager_to_subordinate_passthrough(%mgr : memref<8xi32>)
    -> (memref<8xi32>) {
  // expected-error @+1 {{memref module result #0 must originate from a subordinate provider result}}
  fabric.yield %mgr : memref<8xi32>
}

// -----
// memref width/shape mismatch on yield: a subordinate source memref must equal
// the module result memref exactly.
fabric.module @m_yield_memref_mismatch(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>)
    -> (memref<?x!fabric.bits<4>>) {
  %sub, %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = [[0 : i32]]
        }}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (memref<?x!fabric.bits<8>>, !fabric.bits<32>, !fabric.bits<0>)
  // expected-error @+1 {{declared destination type 'memref<?x!fabric.bits<8>>' does not match the module's result type 'memref<?x!fabric.bits<4>>'}}
  fabric.yield %sub : memref<?x!fabric.bits<8>>
}

// -----
// `to <inner-type>` cross-kind clause on fabric.fifo operand: a `bits`
// source cannot relax to a `bits_tag` FIFO inner type.
fabric.module @m_fifo_to_clause_kind_mismatch(%a : !fabric.bits<32>) {
  // expected-error @+1 {{must share the same fabric kind}}
  %0 = fabric.fifo %a [max_depth = 4, bypassable = false]
       : !fabric.bits<32> to !fabric.bits_tag<8, 3>
  fabric.yield
}

// -----
// A module transport input is point-to-point. Broadcast must be expressed by
// an explicit routing resource rather than by reusing an SSA input value.
// expected-error @+1 {{transport source is used by more than one consumer}}
fabric.module @m_input_fanout_requires_switch(%a : !fabric.bits<32>) {
  %left = fabric.fifo %a [max_depth = 2, bypassable = false]
          : !fabric.bits<32>
  %right = fabric.fifo %a [max_depth = 2, bypassable = false]
           : !fabric.bits<32>
  fabric.yield
}

// -----
// A direct operation result is also point-to-point. Yielding a value after it
// has already fed another resource is a second consuming use.
fabric.module @m_result_fanout_requires_switch(%a : !fabric.bits<32>)
    -> (!fabric.bits<32>) {
  // expected-error @+1 {{transport source is used by more than one consumer}}
  %source = fabric.fifo %a [max_depth = 2, bypassable = false]
            : !fabric.bits<32>
  %sink = fabric.fifo %source [max_depth = 2, bypassable = false]
          : !fabric.bits<32>
  fabric.yield %source : !fabric.bits<32>
}

// -----
// Disallowed input port type: i32 is not a valid fabric.module port type.
// expected-error @+1 {{is not an allowed fabric.module port type}}
fabric.module @m_bad_input_type(%a : i32) {
  fabric.yield
}

// -----
// Disallowed output port type: tensor is not a valid fabric.module port type.
// expected-error @+1 {{is not an allowed fabric.module port type}}
fabric.module @m_bad_output_type() -> (tensor<4xi32>) {
  fabric.yield
}

// -----
// Strict body whitelist: builtin.unrealized_conversion_cast is no longer
// permitted directly in a fabric.module body.
fabric.module @m_cast_rejected() {
  // expected-error @+1 {{is not allowed inside fabric.module}}
  %0 = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.yield
}

// -----
// Strict body whitelist: only fabric.pe, fabric.fifo, and the
// implicit fabric.yield terminator are permitted in fabric.module's body.
// A raw fabric.fu directly in the module body is rejected.
fabric.module @m_raw_fu_rejected(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{is not allowed inside fabric.module}}
  fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>) -> () {
    %k = fabric.op [@arith.muli] (%x, %y)
         {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield
  }
  fabric.yield
}

// -----
// Strict body whitelist: a raw fabric.op directly in the module body is
// rejected.
fabric.module @m_raw_op_rejected(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // expected-error @+1 {{is not allowed inside fabric.module}}
  %k = fabric.op [@arith.addi] (%a, %b)
       {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  fabric.yield
}

// -----
// fabric.fu placed at the top of builtin.module is rejected by the FU
// verifier (parent must be fabric.pe).
%a_top = builtin.unrealized_conversion_cast to !fabric.bits<32>
%b_top = builtin.unrealized_conversion_cast to !fabric.bits<32>
// expected-error @+1 {{must be inside a fabric.pe (parent must be fabric.pe)}}
fabric.fu(%x = %a_top : !fabric.bits<32>, %y = %b_top : !fabric.bits<32>) -> () {
  %k = fabric.op [@arith.muli] (%x, %y)
       {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  fabric.yield
}
