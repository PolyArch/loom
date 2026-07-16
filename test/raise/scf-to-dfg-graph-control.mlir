// RUN: loom-raise-opt --loom-lower-graph-control --verify-diagnostics %s | FileCheck %s

// Positive (mux case): a graph.func body with an `scf.if %c -> (f32)`
// whose then-region trivially yields %a and whose else-region trivially
// yields %b is rewritten into a single dataflow.mux. The lane order
// follows the spec part-2 rule: lane 0 is the false (else) yield, lane
// 1 is the true (then) yield.

// CHECK-LABEL: dataflow.graph.func private @g_mux_two_arith
// CHECK-NOT: scf.if
// CHECK: %[[MUX:.*]] = dataflow.mux %arg1, %arg3, %arg2 : (i1, f32, f32) -> f32
// CHECK: dataflow.graph.return %arg0, %[[MUX]] : none, f32
dataflow.graph.func private @g_mux_two_arith(%arg0: none, %arg1: i1,
                                             %arg2: f32, %arg3: f32)
    -> (none, f32) {
  %0 = scf.if %arg1 -> (f32) {
    scf.yield %arg2 : f32
  } else {
    scf.yield %arg3 : f32
  }
  dataflow.graph.return %arg0, %0 : none, f32
}

// Positive (gate case): a graph.func body with an `scf.if %c { %x =
// arith.addf %a, %b ; scf.yield }` -- no results, then-region only,
// pure body -- is lifted into a dataflow.gate over %x. The arith.addf
// is hoisted into the parent block and downstream uses (none here, so
// no use rewrite is required) are routed through the gate's
// `after_value`.

// CHECK-LABEL: dataflow.graph.func private @g_gate_pure_then
// CHECK-NOT: scf.if
// CHECK: %[[ADD:.*]] = arith.addf %arg2, %arg3 : f32
// CHECK: %{{.*}}, %{{.*}} = dataflow.gate %arg1, %[[ADD]] : f32
// CHECK: dataflow.graph.return %arg0, %arg2 : none, f32
dataflow.graph.func private @g_gate_pure_then(%arg0: none, %arg1: i1,
                                              %arg2: f32, %arg3: f32)
    -> (none, f32) {
  scf.if %arg1 {
    %0 = arith.addf %arg2, %arg3 : f32
    scf.yield
  }
  dataflow.graph.return %arg0, %arg2 : none, f32
}

// Negative-bail (effectful gate-shaped scf.if): a graph.func body with
// an `scf.if %c { llvm.store ... }` -- no results, then-region only,
// effectful body -- is left alone. No dataflow.gate is emitted because
// the store cannot be lifted out unconditionally.

// CHECK-LABEL: dataflow.graph.func private @g_bail_effectful_gate
// CHECK: scf.if %arg1
// CHECK: llvm.store
// CHECK-NOT: dataflow.gate
dataflow.graph.func private @g_bail_effectful_gate(%arg0: none, %arg1: i1,
                                                   %arg2: f32,
                                                   %arg3: !llvm.ptr)
    -> (none) {
  // expected-remark@+1 {{loom-lower-graph-control: scf.if shape not lifted}}
  scf.if %arg1 {
    llvm.store %arg2, %arg3 : f32, !llvm.ptr
    scf.yield
  }
  dataflow.graph.return %arg0 : none
}

// Negative-bail (effectful resultful scf.if): a graph.func body with an
// `scf.if %c -> (i32)` whose then-region issues a llvm.store cannot be
// wrapped with dataflow.gate. The else-lane result is a real value, and gate
// would drop it on a false condition before downstream consumers see it.

// CHECK-LABEL: dataflow.graph.func private @g_side_effect_gate_result
// CHECK: %[[IF:.*]] = scf.if %arg1 -> (i32)
// CHECK: llvm.store
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield
// CHECK: }
// CHECK-NOT: dataflow.gate %arg1, %[[IF]] : i32
// CHECK: dataflow.graph.return %arg0, %[[IF]] : none, i32
dataflow.graph.func private @g_side_effect_gate_result(%arg0: none, %arg1: i1,
                                                       %arg2: i32,
                                                       %arg3: i32,
                                                       %arg4: !llvm.ptr)
    -> (none, i32) {
  // expected-remark@+1 {{loom-lower-graph-control: scf.if shape not lifted}}
  %0 = scf.if %arg1 -> (i32) {
    llvm.store %arg2, %arg4 : i32, !llvm.ptr
    scf.yield %arg2 : i32
  } else {
    scf.yield %arg3 : i32
  }
  dataflow.graph.return %arg0, %0 : none, i32
}

// Negative-bail (uncommon two-sided no-result): a graph.func body with
// an `scf.if %c { ... } else { ... }` where neither region yields a
// value is left alone. The dataflow.gate / .mux primitives do not
// naturally express two-sided side-effecting alternatives.

// CHECK-LABEL: dataflow.graph.func private @g_bail_two_sided_no_result
// CHECK: scf.if %arg1
// CHECK-NEXT: llvm.store
// CHECK: } else {
// CHECK-NEXT: llvm.store
// CHECK-NOT: dataflow.gate
// CHECK-NOT: dataflow.mux
dataflow.graph.func private @g_bail_two_sided_no_result(%arg0: none,
                                                        %arg1: i1,
                                                        %arg2: f32,
                                                        %arg3: f32,
                                                        %arg4: !llvm.ptr)
    -> (none) {
  // expected-remark@+1 {{loom-lower-graph-control: scf.if shape not lifted}}
  scf.if %arg1 {
    llvm.store %arg2, %arg4 : f32, !llvm.ptr
    scf.yield
  } else {
    llvm.store %arg3, %arg4 : f32, !llvm.ptr
    scf.yield
  }
  dataflow.graph.return %arg0 : none
}

// Negative-bail: an i8 ordinal is not the stream's canonical i16 ordinal and
// can wrap independently of the pointer carry. It cannot prove that the load
// and conditional store address the same element.

// CHECK-LABEL: dataflow.graph.func private @g_bail_wrapping_store_ordinal
// CHECK: scf.if
// CHECK-NOT: arith.select
// CHECK: dataflow.graph.return
dataflow.graph.func private @g_bail_wrapping_store_ordinal(
    %arg0: none, %arg1: i16, %arg2: i16, %arg3: i16, %arg4: i8,
    %arg5: !llvm.ptr) -> none {
  %mem = builtin.unrealized_conversion_cast %arg5
      : !llvm.ptr to memref<?xi8>
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = ">", step_op = "+="} : i16
  %replacement = dataflow.invariant %rwc, %arg4 : i8
  %pointer = dataflow.carry %rwc, %arg5, %next_pointer : !llvm.ptr
  %after_cond, %current_pointer = dataflow.gate %rwc, %pointer : !llvm.ptr
  %c0 = dataflow.constant %arg0 {const_value = 0 : i8} : i8
  %c1 = dataflow.constant %arg0 {const_value = 1 : i8} : i8
  %one = dataflow.invariant %rwc, %c1 : i8
  %ordinal = dataflow.carry %rwc, %c0, %next_ordinal : i8
  %next_ordinal = arith.addi %ordinal, %one : i8
  %addr = arith.index_cast %ordinal : i8 to index
  %data, %done = dataflow.load %mem[%addr] %arg0 : memref<?xi8>
  %replace = arith.cmpi slt, %data, %replacement : i8
  // expected-remark@+1 {{loom-lower-graph-control: scf.if shape not lifted}}
  scf.if %replace {
    %c0_store = arith.constant 0 : index
    %store_mem = builtin.unrealized_conversion_cast %current_pointer
        : !llvm.ptr to memref<?xi8>
    %store = dataflow.store %store_mem[%c0_store] %replacement %arg0
        : memref<?xi8>
  }
  %next_pointer = llvm.getelementptr inbounds|nuw %current_pointer[1]
      : (!llvm.ptr) -> !llvm.ptr, i8
  dataflow.graph.return %arg0 : none
}

// Positive (index-domain induction): loop-carried i64 induction values that
// only feed memory addresses may be narrowed to Loom's index domain before PnR.
// The graph result keeps the original i64 shape by casting the narrowed cursor
// back at the return boundary.

// CHECK-LABEL: dataflow.graph.func private @g_index_domain_i64_carry
// CHECK: %[[INDEX:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK-NOT: dataflow.invariant {{.*}} : i64
// CHECK: %[[INIT:.*]] = arith.index_cast %arg5 : i64 to index
// CHECK: %[[STEP:.*]] = arith.index_cast %arg3 : i64 to index
// CHECK: %[[STEP_INV:.*]] = dataflow.invariant %[[RWC]], %[[STEP]] : index
// CHECK: %[[CURSOR:.*]] = dataflow.carry %[[RWC]], %[[INIT]], %[[NEXT:.*]] : index
// CHECK: %[[NEXT]] = arith.addi %[[CURSOR]], %[[STEP_INV]] : index
// CHECK: dataflow.load %arg4[%[[CURSOR]]]
// CHECK: %[[RETURN_CURSOR:.*]] = arith.index_cast %[[CURSOR]] : index to i64
// CHECK: dataflow.graph.return %{{.*}}, %[[RETURN_CURSOR]], %{{.*}} : none, i64, f32
dataflow.graph.func private @g_index_domain_i64_carry(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: memref<?xf32>, %arg5: i64, %arg6: f32) -> (none, i64, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %step = dataflow.invariant %rwc, %arg3 : i64
  %cursor = dataflow.carry %rwc, %arg5, %next : i64
  %addr = arith.index_cast %cursor : i64 to index
  %data, %done = dataflow.load %arg4[%addr] %arg0 : memref<?xf32>
  %next = arith.addi %cursor, %step : i64
  dataflow.graph.return %done, %cursor, %data : none, i64, f32
}

// Positive (index-domain address mask): correlation-style address arithmetic
// computes a wrapped address using i64 loop-invariant offset/mask values. The
// address-only add/and chain must lower to index-width ops before PnR.

// CHECK-LABEL: dataflow.graph.func private @g_index_domain_address_mask
// CHECK: %[[INDEX2:.*]], %[[RWC2:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK-NOT: dataflow.invariant {{.*}} : i64
// CHECK: %[[INDEX_AS_INDEX:.*]] = arith.index_cast %[[INDEX2]] : i64 to index
// CHECK: %[[OFFSET_ARG:.*]] = arith.index_cast %arg5 : i64 to index
// CHECK: %[[OFFSET:.*]] = dataflow.invariant %[[RWC2]], %[[OFFSET_ARG]] : index
// CHECK: %[[ADD:.*]] = arith.addi %[[INDEX_AS_INDEX]], %[[OFFSET]] : index
// CHECK: %[[MASK_ARG:.*]] = arith.index_cast %arg6 : i64 to index
// CHECK: %[[MASK:.*]] = dataflow.invariant %[[RWC2]], %[[MASK_ARG]] : index
// CHECK: %[[ADDR:.*]] = arith.andi %[[ADD]], %[[MASK]] : index
// CHECK: dataflow.load %arg4[%[[ADDR]]]
dataflow.graph.func private @g_index_domain_address_mask(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: memref<?xf32>, %arg5: i64, %arg6: i64) -> (none, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %offset = dataflow.invariant %rwc, %arg5 : i64
  %mask = dataflow.invariant %rwc, %arg6 : i64
  %biased = arith.addi %index, %offset : i64
  %wrapped = arith.andi %biased, %mask : i64
  %addr = arith.index_cast %wrapped : i64 to index
  %data, %done = dataflow.load %arg4[%addr] %arg0 : memref<?xf32>
  dataflow.graph.return %done, %data : none, f32
}

// Positive (index-domain guarded address): FIR-style guarded loads compute an
// i64 offset, compare it against an invariant bound, and select between the
// computed address and zero. The whole address/control chain is index-domain
// and must not require 64-bit fabric invariant resources.

// CHECK-LABEL: dataflow.graph.func private @g_index_domain_guarded_address
// CHECK: %[[INDEX3:.*]], %[[RWC3:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK-NOT: dataflow.invariant {{.*}} : i64
// CHECK: %[[UB_ARG:.*]] = arith.index_cast %arg6 : i64 to index
// CHECK: %[[UB:.*]] = dataflow.invariant %[[RWC3]], %[[UB_ARG]] : index
// CHECK: %[[INDEX_AS_INDEX3:.*]] = arith.index_cast %[[INDEX3]] : i64 to index
// CHECK: %[[DELTA:.*]] = arith.subi %[[UB]], %[[INDEX_AS_INDEX3]] : index
// CHECK: %[[LB_ARG:.*]] = arith.index_cast %arg5 : i64 to index
// CHECK: %[[LB:.*]] = dataflow.invariant %[[RWC3]], %[[LB_ARG]] : index
// CHECK: %[[PRED:.*]] = arith.cmpi sgt, %[[DELTA]], %[[LB]] : index
// CHECK: %[[SAFE_ADDR:.*]] = arith.select %[[PRED]], %[[DELTA]], %{{.*}} : index
// CHECK: dataflow.load %arg4[%[[SAFE_ADDR]]]
dataflow.graph.func private @g_index_domain_guarded_address(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: memref<?xf32>, %arg5: i64, %arg6: i64) -> (none, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %lb = dataflow.invariant %rwc, %arg5 : i64
  %ub = dataflow.invariant %rwc, %arg6 : i64
  %delta = arith.subi %ub, %index : i64
  %pred = arith.cmpi sgt, %delta, %lb : i64
  %addr = arith.index_cast %delta : i64 to index
  %zero = dataflow.constant %arg0 {const_value = 0 : index} : index
  %safe = arith.select %pred, %addr, %zero : index
  %data, %done = dataflow.load %arg4[%safe] %arg0 : memref<?xf32>
  dataflow.graph.return %done, %data : none, f32
}

// Positive (index-domain zext address): an i32 address expression widened with
// llvm.zext before index_cast is still an address-domain value and must not
// require a 64-bit fabric op.

// CHECK-LABEL: dataflow.graph.func private @g_index_domain_zext_address
// CHECK: %[[LHS:.*]] = arith.index_cast %arg5 : i32 to index
// CHECK: %[[RHS:.*]] = arith.index_cast %arg6 : i32 to index
// CHECK: %[[ADDR:.*]] = arith.addi %[[LHS]], %[[RHS]] : index
// CHECK-NOT: llvm.zext
// CHECK: dataflow.load %arg4[%[[ADDR]]]
dataflow.graph.func private @g_index_domain_zext_address(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: memref<?xf32>, %arg5: i32, %arg6: i32) -> (none, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %sum = arith.addi %arg5, %arg6 : i32
  %wide = llvm.zext nneg %sum : i32 to i64
  %addr = arith.index_cast %wide : i64 to index
  %data, %done = dataflow.load %arg4[%addr] %arg0 : memref<?xf32>
  dataflow.graph.return %done, %data : none, f32
}
