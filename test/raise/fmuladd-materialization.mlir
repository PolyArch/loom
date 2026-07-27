// RUN: split-file %s %t
// RUN: not loom-raise-opt --loom-materialize-fmuladd %t/choice.mlir 2>&1 | FileCheck %s --check-prefix=UNSELECTED
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=fused %t/choice.mlir | FileCheck %s --check-prefix=FUSED
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=split %t/choice.mlir | FileCheck %s --check-prefix=SPLIT
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=fused %t/choice.mlir | mlir-opt --convert-math-to-llvm --convert-arith-to-llvm | FileCheck %s --check-prefix=FUSED-LLVM --implicit-check-not=constrained
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=split %t/choice.mlir | mlir-opt --convert-math-to-llvm --convert-arith-to-llvm | FileCheck %s --check-prefix=SPLIT-LLVM --implicit-check-not=constrained
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=split %t/choice.mlir | mlir-opt --math-uplift-to-fma | FileCheck %s --check-prefix=SPLIT-KEPT --implicit-check-not=math.fma
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=fused %t/unrepresentable.mlir | FileCheck %s --check-prefix=SCOPED
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=fused %t/nested.mlir | FileCheck %s --check-prefix=NESTED --implicit-check-not=llvm.intr.fmuladd
// RUN: loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading %t/selected-fused.mlir | FileCheck %s --check-prefix=SELECTED-FUSED --implicit-check-not=loom.spatial_region

// `llvm.intr.fmuladd` states a choice, not a computation: the target may fuse
// it into one rounding or evaluate a separate multiply and add. Materializing
// that choice is a typed decision with no default, so the shape is required
// and is never inferred from the intrinsic spelling.
// UNSELECTED: loom-materialize-fmuladd requires an explicit 'shape' option

// Fusing is what the fused shape decided, so the exact operand and result
// types and the complete imported fast-math contract -- `contract` included --
// all carry onto the one `math.fma`.
// FUSED-LABEL: llvm.func @chosen
// FUSED: math.fma %arg0, %arg1, %arg2 fastmath<nnan,contract> : f32
// FUSED-NOT: llvm.intr.fmuladd
// FUSED-LABEL: llvm.func @vector_chosen
// FUSED: math.fma %arg0, %arg1, %arg2 : vector<4xf32>
// FUSED-NOT: llvm.intr.fmuladd

// The split shape is an ordinary multiply then an ordinary add, each rounding
// on its own. `contract` is the source's permission to fuse them back into one
// rounding, so selecting Split consumes it: neither operation restates it.
// Every other imported flag describes the computation rather than the fusion
// and carries onto both operations unchanged.
// SPLIT-LABEL: llvm.func @chosen
// SPLIT: %[[PROD:.*]] = arith.mulf %arg0, %arg1 fastmath<nnan> : f32
// SPLIT: arith.addf %[[PROD]], %arg2 fastmath<nnan> : f32
// SPLIT-NOT: llvm.intr.fmuladd
// SPLIT-LABEL: llvm.func @vector_chosen
// SPLIT: %[[VPROD:.*]] = arith.mulf %arg0, %arg1 : vector<4xf32>
// SPLIT: arith.addf %[[VPROD]], %arg2 : vector<4xf32>
// SPLIT-NOT: llvm.intr.fmuladd

// Consuming the permission is what enforces the decision. Upstream's own
// arith-to-`math.fma` uplift contracts a multiply and add only when both still
// permit it, so the selected split survives it unchanged instead of being
// silently re-fused.
// SPLIT-KEPT-LABEL: llvm.func @chosen
// SPLIT-KEPT: %[[KPROD:.*]] = arith.mulf %arg0, %arg1 fastmath<nnan> : f32
// SPLIT-KEPT: arith.addf %[[KPROD]], %arg2 fastmath<nnan> : f32

// Neither shape states a rounding mode, because a standard operation that
// states one is a constrained operation: standard lowering would turn it into
// `llvm.intr.experimental.constrained.*` and drop the fast-math flags.
// llvm.intr.fmuladd is an ordinary non-constrained intrinsic, so both shapes
// lower back to ordinary LLVM operations carrying exactly the flags the shape
// kept.
// FUSED-LLVM: llvm.intr.fma(%arg0, %arg1, %arg2) {fastmathFlags = #llvm.fastmath<nnan, contract>}
// SPLIT-LLVM: %[[LPROD:.*]] = llvm.fmul %arg0, %arg1 {fastmathFlags = #llvm.fastmath<nnan>} : f32
// SPLIT-LLVM: llvm.fadd %[[LPROD]], %arg2 {fastmathFlags = #llvm.fastmath<nnan>} : f32

// A callable stating a floating-point policy no standard operation restates
// cannot receive either shape. That refusal is local to the intrinsic: a
// representable sibling still receives the selected typed shape, while the
// unrepresentable intrinsic remains explicit for candidate finalization to
// reject if its region is selected for SpatialCore.
// SCOPED-LABEL: llvm.func @representable
// SCOPED: math.fma %arg0, %arg1, %arg2 : f32
// SCOPED-NOT: llvm.intr.fmuladd
// SCOPED-LABEL: llvm.func @estimated
// SCOPED: llvm.intr.fmuladd
// SCOPED-NOT: math.fma

// A native func.func owns a nested imported llvm.func held under a
// builtin.module. Each callable region processes only the operations it owns:
// the enclosing func.func's walk prunes the nested llvm.func body, so the one
// llvm.intr.fmuladd inside it is collected exactly once and materialized once.
// Before the ownership walk pruned nested callables, the same intrinsic was
// collected from both regions and the second materialization crashed on the
// already-replaced operation.
// NESTED-LABEL: func.func @native_owner
// NESTED-LABEL: llvm.func @inner
// NESTED: math.fma %{{.*}}, %{{.*}}, %{{.*}} : f32

// The Fused execution shape materializes as the registered `math.fma` actor.
// A selected candidate carrying that exact actor must cross atomic graph
// publication without a frontend name rule or a second actor registry.
// SELECTED-FUSED-LABEL: dataflow.thread private @selected_fused domain(#dataflow.thread_domain<dense>)
// SELECTED-FUSED: dataflow.graph.launch @selected_fused_graph
// SELECTED-FUSED-LABEL: dataflow.graph private @selected_fused_graph
// SELECTED-FUSED: math.fma %{{.*}}, %{{.*}}, %{{.*}} fastmath<nnan,contract> : f32

//--- choice.mlir
llvm.func @chosen(%x: f32, %y: f32, %z: f32) -> f32 {
  %r = llvm.intr.fmuladd(%x, %y, %z)
      {fastmathFlags = #llvm.fastmath<nnan, contract>} : (f32, f32, f32) -> f32
  llvm.return %r : f32
}

llvm.func @vector_chosen(%x: vector<4xf32>, %y: vector<4xf32>,
                         %z: vector<4xf32>) -> vector<4xf32> {
  %r = llvm.intr.fmuladd(%x, %y, %z)
      : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %r : vector<4xf32>
}

//--- unrepresentable.mlir
llvm.func @representable(%x: f32, %y: f32, %z: f32) -> f32 {
  %r = llvm.intr.fmuladd(%x, %y, %z) : (f32, f32, f32) -> f32
  llvm.return %r : f32
}

llvm.func @estimated(%x: f32, %y: f32, %z: f32) -> f32
    attributes {reciprocal_estimates = "all"} {
  %r = llvm.intr.fmuladd(%x, %y, %z) : (f32, f32, f32) -> f32
  llvm.return %r : f32
}

//--- nested.mlir
func.func @native_owner(%a: f32, %b: f32, %c: f32) -> f32 {
  builtin.module {
    llvm.func @inner(%x: f32, %y: f32, %z: f32) -> f32 {
      %r = llvm.intr.fmuladd(%x, %y, %z) : (f32, f32, f32) -> f32
      llvm.return %r : f32
    }
  }
  return %a : f32
}

//--- selected-fused.mlir
dataflow.thread private @selected_fused domain(#dataflow.thread_domain<dense>)(
    %lhs: f32, %rhs: f32, %acc: f32) ctrl (%start: none) {
  %result = "loom.spatial_region"(%lhs, %rhs, %acc)
      <{operandSegmentSizes = array<i32: 3, 0, 0, 0>,
        resultSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%a: f32, %b: f32, %c: f32):
      %fused = math.fma %a, %b, %c fastmath<nnan,contract> : f32
      "loom.spatial_yield"(%fused)
          <{operandSegmentSizes = array<i32: 1, 0>}> : (f32) -> ()
  }) {graph_name = "selected_fused_graph", source_maps = []} :
      (f32, f32, f32) -> f32
  dataflow.thread.yield
}
