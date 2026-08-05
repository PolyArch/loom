// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Undefined symbol: instantiate references a name that is not defined in
// any reachable SymbolTable.
%a = builtin.unrealized_conversion_cast to !fabric.bits<32>
// expected-error @+1 {{references undefined symbol '@missing'}}
%r = fabric.instantiate @missing(%a : !fabric.bits<32>) -> (!fabric.bits<32>)

// -----
// Wrong-kind target: instantiating a fabric.module symbol from inside a
// fabric.pe body is illegal (pe-body sites may target only fabric.fu).
fabric.module @leaf_mod(%x : !fabric.bits<32>) -> (!fabric.bits<32>) {
  fabric.yield %x : !fabric.bits<32>
}
fabric.module @host_wrong_kind(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    // expected-error @+1 {{inside a fabric.pe body may only target 'fabric.fu'}}
    %g = fabric.instantiate @leaf_mod(%pa : !fabric.bits<32>)
         -> (!fabric.bits<32>)
  }
  fabric.yield
}

// -----
// Self-reference: a fabric.module body cannot instantiate its own
// enclosing fabric.module symbol (recursion is forbidden).
fabric.module @recursive_self(%a : !fabric.bits<32>) {
  // expected-error @+1 {{cannot instantiate the symbol that encloses it (self-reference of '@recursive_self')}}
  %r = fabric.instantiate @recursive_self(%a : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield
}

// -----
// Forward reference: instantiate appears textually before the named pe
// definition in the same fabric.module body.
fabric.module @host_forward(%a : !fabric.bits<32>) {
  // expected-error @+1 {{forward reference to symbol '@LATER'}}
  %s = fabric.instantiate @LATER(%a : !fabric.bits<32>) -> (!fabric.bits<32>)
  fabric.pe @LATER [spatial] (!fabric.bits<32>) -> (!fabric.bits<32>) {
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

// -----
// Out-of-scope reference: a top-level fabric.instantiate cannot reach a
// pe symbol that is nested inside another fabric.module's body.
fabric.module @scope_leak_host(%a : !fabric.bits<32>) {
  fabric.pe @INNER [spatial] (!fabric.bits<32>) -> (!fabric.bits<32>) {
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
%t = builtin.unrealized_conversion_cast to !fabric.bits<32>
// expected-error @+1 {{references undefined symbol '@INNER'}}
%u = fabric.instantiate @INNER(%t : !fabric.bits<32>) -> (!fabric.bits<32>)

// -----
// Operand count mismatch.
fabric.module @leaf_two_in(%x : !fabric.bits<32>, %y : !fabric.bits<32>)
    -> (!fabric.bits<32>) {
  fabric.yield %x : !fabric.bits<32>
}
fabric.module @host_count_mismatch(%a : !fabric.bits<32>) {
  // expected-error @+1 {{operand count (1) does not match callee '@leaf_two_in' input port count (2)}}
  %r = fabric.instantiate @leaf_two_in(%a : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield
}

// -----
// Output type mismatch: result type does not equal callee's declared
// output port type. Output direction is strict in this iteration.
fabric.module @leaf_out16(%x : !fabric.bits<32>) -> (!fabric.bits<16>) {
  %r = fabric.fifo %x [max_depth = 1, bypassable = false]
       : !fabric.bits<32> to !fabric.bits<16>
  fabric.yield %r : !fabric.bits<16>
}
fabric.module @host_out_mismatch(%a : !fabric.bits<32>) {
  // expected-error @+1 {{result #0 type '!fabric.bits<32>' must equal callee '@leaf_out16' output port type '!fabric.bits<16>'}}
  %r = fabric.instantiate @leaf_out16(%a : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield
}

// -----
// memref operands cannot use the 'to <inner-type>' clause: memref types
// must match exactly (no width relaxation on memref).
fabric.module @leaf_mem(%m : memref<8xi32>) {
  fabric.yield
}
fabric.module @host_mem_relax(%m : memref<8xi32>) {
  // expected-error @+1 {{memref operands cannot use the 'to <inner-type>' clause}}
  fabric.instantiate @leaf_mem(%m : memref<8xi32> to memref<4xi32>) -> ()
  fabric.yield
}

// -----
// Named fabric.pe template attempting to declare SSA results: once @sym is
// present the parser switches to the template signature form. Anonymous-
// style operand binding `(%pa = %a : ...)` is therefore rejected.
fabric.module @host_named_with_results(%a : !fabric.bits<32>) {
  // expected-error @+1 {{expected non-function type}}
  %r = fabric.pe @ALU [spatial] (%pa = %a : !fabric.bits<32>)
                                   -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Named fabric.pe template whose terminator carries a value: `function_type`
// is the sole owner of the PE result ports, so the body terminator is a
// pure zero-operand signature terminator.
fabric.module @host_named_value_bearing_yield(%a : !fabric.bits<32>) {
  // expected-error @+1 {{named fabric.pe body must terminate with a zero-operand fabric.yield}}
  fabric.pe @ALU [spatial] (!fabric.bits<32>) -> (!fabric.bits<32>) {
  ^bb0(%pa: !fabric.bits<32>):
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    fabric.yield %pa : !fabric.bits<32>
  }
  fabric.yield
}

// -----
// Named fabric.pe template whose zero-operand terminator still carries a
// 'declared_types' attribute: that would restate the result port types and
// compete with `function_type` for ownership of them.
fabric.module @host_named_yield_declared_types(%a : !fabric.bits<32>) {
  // expected-error @+1 {{must not carry a 'declared_types' attribute}}
  fabric.pe @ALU [spatial] (!fabric.bits<32>) -> (!fabric.bits<32>) {
  ^bb0(%pa: !fabric.bits<32>):
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    fabric.yield {declared_types = [!fabric.bits<32>]}
  }
  fabric.yield
}

// -----
// Anonymous fabric.fu attempting to attach a sym_name on an in-pe-body
// instance (the legacy "named instance" shape). With the template-only
// dichotomy the named form is template syntax (zero operands) and the
// anonymous-style `(%fa = ...)` binding is rejected by the parser.
fabric.module @host_named_fu_anon_shape(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    // expected-error @+1 {{expected non-function type}}
    fabric.fu @F (%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----
// Instantiate site references an anonymous (unnamed) fabric.pe: the
// symbol lookup fails because the op carries no @sym attribute.
fabric.module @host_target_anon(%a : !fabric.bits<32>) {
  %a_to_pe, %a_to_instantiate = fabric.switch [spatial] %a
       [{connectivity_table = ["1", "1"]}]
       : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
  %r = fabric.pe [spatial] (%pa = %a_to_pe : !fabric.bits<32>)
       -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  // expected-error @+1 {{references undefined symbol '@anon_pe'}}
  %s = fabric.instantiate @anon_pe(%a_to_instantiate : !fabric.bits<32>)
       -> (!fabric.bits<32>)
  fabric.yield
}

// -----
// Generic assembly must reject a non-array inner_input_types container.
fabric.module @inner_types_target(%x : !fabric.bits<32>)
    -> (!fabric.bits<32>) {
  fabric.yield %x : !fabric.bits<32>
}
fabric.module @inner_types_host(%a : !fabric.bits<32>) {
  // expected-error @+1 {{for `inner_input_types`: expected array attribute}}
  %r = "fabric.instantiate"(%a) <{
    callee = @inner_types_target,
    inner_input_types = "not-an-array"
  }> : (!fabric.bits<32>) -> !fabric.bits<32>
  fabric.yield
}

// -----
// A non-empty inner_input_types property must encode a real endpoint change.
fabric.module @redundant_inner_types_target(%x : !fabric.bits<32>)
    -> (!fabric.bits<32>) {
  fabric.yield %x : !fabric.bits<32>
}
fabric.module @redundant_inner_types_host(%a : !fabric.bits<32>) {
  // expected-error @+1 {{must be empty when every destination input type equals its operand type}}
  %r = "fabric.instantiate"(%a) <{
    callee = @redundant_inner_types_target,
    inner_input_types = [!fabric.bits<32>]
  }> : (!fabric.bits<32>) -> !fabric.bits<32>
  fabric.yield
}

// -----
// A same-name discardable attribute must not shadow a valid inherent property.
fabric.module @inner_types_collision_target(%x : !fabric.bits<16>)
    -> (!fabric.bits<16>) {
  fabric.yield %x : !fabric.bits<16>
}
fabric.module @inner_types_collision_host(%a : !fabric.bits<32>) {
  // expected-error @+1 {{discardable attribute 'inner_input_types' conflicts with the inherent property of the same name}}
  %r = "fabric.instantiate"(%a) <{
    callee = @inner_types_collision_target,
    inner_input_types = [!fabric.bits<16>]
  }> {inner_input_types = "not-an-array"}
      : (!fabric.bits<32>) -> !fabric.bits<16>
  fabric.yield
}

// -----
// The flat binding property must contain complete typed triples.
fabric.module @malformed_domain_binding_target() {
}
fabric.module @malformed_domain_binding_host() {
  // expected-error @+1 {{has malformed domain-slot bindings}}
  fabric.instantiate @malformed_domain_binding_target() -> ()
      {domain_slot_bindings = array<i64: 0, 0>}
}

// -----
// A zero-slot Module target requires the exact empty relation.
fabric.module @zero_slot_domain_target() {
}
fabric.module @zero_slot_domain_host() {
  // expected-error @+1 {{binding count does not equal the child slot count}}
  fabric.instantiate @zero_slot_domain_target() -> ()
      {domain_slot_bindings = array<i64: 0, 0, 0>}
}

// -----
// A non-Module target cannot carry a Module domain relation.
fabric.module @non_module_domain_binding(%arg : !fabric.bits<8>) {
  fabric.switch @SW [spatial]
      (!fabric.bits<8>) -> (!fabric.bits<8>)
      [{connectivity_table = ["1"]}]
  // expected-error @+1 {{a non-Module target cannot have domain-slot bindings}}
  %result = fabric.instantiate @SW(
      %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
      {domain_slot_bindings = array<i64: 0, 0, 0>}
  fabric.yield
}
