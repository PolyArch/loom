// RUN: split-file %s %t
// RUN: not loom-raise-opt --loom-materialize-fmuladd %t/choice.mlir 2>&1 | FileCheck %s --check-prefix=UNSELECTED
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=fused %t/choice.mlir | FileCheck %s --check-prefix=FUSED
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=split %t/choice.mlir | FileCheck %s --check-prefix=SPLIT
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=fused %t/choice.mlir | mlir-opt --convert-math-to-llvm --convert-arith-to-llvm | FileCheck %s --check-prefix=FUSED-LLVM --implicit-check-not=constrained
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=split %t/choice.mlir | mlir-opt --convert-math-to-llvm --convert-arith-to-llvm | FileCheck %s --check-prefix=SPLIT-LLVM --implicit-check-not=constrained
// RUN: loom-raise-opt --loom-materialize-fmuladd=shape=split %t/choice.mlir | mlir-opt --math-uplift-to-fma | FileCheck %s --check-prefix=SPLIT-KEPT --implicit-check-not=math.fma
// RUN: not loom-raise-opt --loom-materialize-fmuladd=shape=fused --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t/unrepresentable.mlir 2>&1 | FileCheck %s --check-prefix=REFUSE

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
// cannot receive either shape. The transform is atomic, so refusing it leaves
// every fmuladd in the module untouched, including the one that on its own
// would have been representable.
// REFUSE: error: 'llvm.intr.fmuladd' op cannot be materialized
// REFUSE-LABEL: llvm.func @representable
// REFUSE: llvm.intr.fmuladd
// REFUSE-LABEL: llvm.func @estimated
// REFUSE: llvm.intr.fmuladd
// REFUSE-NOT: math.fma
// REFUSE-NOT: arith.mulf

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
