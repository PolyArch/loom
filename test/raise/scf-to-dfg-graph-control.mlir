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
