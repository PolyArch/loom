#ifndef LOOM_FRONTEND_RAISING_PASSES_H
#define LOOM_FRONTEND_RAISING_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class PassManager;
} // namespace mlir

namespace loom {
namespace raising {

// Convert llvm.br / llvm.cond_br / llvm.switch terminators inside callable
// regions into the matching cf.br / cf.cond_br / cf.switch ops, carrying
// imported branch weights and loop annotations onto the replacing branch.
// Pattern-rewrite based, with no incidental region simplification.
// llvm.return is intentionally preserved: it is the return of an imported
// LLVM callable and the CFG-to-SCF transformation treats it as an ordinary
// return-like exit. A weighted llvm.switch keeps its LLVM form, because
// cf.switch has no branch-weight carrier and preserving the operation that
// owns the imported profile is exact where respelling it is not.
std::unique_ptr<::mlir::Pass> createLLVMCfToCfPass();

// Structure the cf-shaped control flow of each callable region with the
// upstream region-level mlir::transformCFGToSCF utility. An imported
// llvm.func is structured in place and keeps its complete ABI envelope;
// undefined values are spelled llvm.mlir.undef and unreachable continuations
// llvm.unreachable there, and each imported loop annotation moves to the
// structured loop that owns its cycle.
//
// A region that cannot be structured exactly -- weighted control no structured
// operation can own, a terminator whose transfer would not be restated
// exactly, a switch carrier that would drop high bits, a value type LLVM
// cannot spell, or a loop annotation whose loop is not exactly identifiable --
// is preserved with its complete original semantics. Unstructured cf control
// is legal S0, so this never fails the module; rejection belongs to a
// candidate that selects a loom.spatial_region and needs structured control
// there. Preservation is per region: sibling callables are still structured.
std::unique_ptr<::mlir::Pass> createLiftCFToSCFPass();

// Rewrite each llvm computation whose complete semantics an arith or math
// operation restates exactly into that standard operation, scoped to callable
// regions. The catalog covers integer and floating arithmetic; integer and
// floating minimum and maximum; negation and absolute value; the exact fused
// multiply-add llvm.intr.fma; comparisons, selection and numeric constants;
// and the width and domain casts trunc / zext / sext, signed and unsigned
// integer-to-float, float-to-signed and float-to-unsigned integer, fpext and
// fptrunc.
//
// An operation is left in llvm form when an operand or result type has no
// exact standard counterpart -- pointers, structs, scalable vectors -- when it
// states a semantic fact the standard operation cannot carry, or when it is a
// floating computation inside a callable stating a floating-point environment
// arith and math cannot restate. llvm.intr.fmuladd states a choice no single
// standard operation restates and is left for typed materialization. Pointer
// arithmetic via llvm.getelementptr stays llvm by design.
std::unique_ptr<::mlir::Pass> createLLVMArithToArithPass();

// Normalize the exact poison-safe loop-exit scaffold emitted by CFG-to-SCF
// structuring so counted-loop uplift can read its comparison directly.
// Scaffolds with live loop results or any unexpected structure are unchanged.
std::unique_ptr<::mlir::Pass> createNormalizeLiftedSCFExitPass();

// Uplift counted scf.while loops inside callable regions into scf.for, keeping
// each loop's imported annotation on the uplifted loop. This combines the
// upstream counted-loop utility with Loom's conservative do-while counted-loop
// fallback. Each existing operation is offered the transform once; loops
// outside callables and unsupported shapes are left as scf.while.
std::unique_ptr<::mlir::Pass> createSCFWhileToForPass();

// The closed set of execution shapes an `llvm.intr.fmuladd` can be
// materialized as. The intrinsic states a target choice rather than a
// computation, and the two shapes round differently, so the shape is a typed
// decision with no default and is never inferred from the intrinsic spelling.
enum class FMulAddExecutionShape {
  // One math.fma: the multiply and the add share a single rounding, and the
  // complete source fast-math contract carries onto it.
  Fused,
  // An arith.mulf followed by an arith.addf, each rounding separately. This
  // shape consumes the source's `contract` permission, so neither operation
  // may be contracted back into one rounding by a later pass or by target
  // code generation. Every other source fast-math flag is preserved.
  Split,
};

// Materialize the selected execution shape for each exactly representable
// llvm.intr.fmuladd in callable regions, preserving exact types and locations
// and carrying the source fast-math flags the selected shape still permits.
// An intrinsic whose exact types or enclosing floating-point policy the
// standard operations cannot restate remains explicit without rejecting
// representable siblings.
// Mechanical S0 raising never runs this pass: the shape is candidate lineage,
// not a mechanical disposition.
std::unique_ptr<::mlir::Pass>
createMaterializeFMulAddPass(FMulAddExecutionShape shape);

// Lift trivially parallel scf.for loops inside callable regions (no iter_args,
// iv-dependent stores only, no nested calls; lane-local structured scf.while
// is allowed) into scf.forall. Each existing operation is offered the
// transform once; loops outside callables and loops that do not match the
// conservative parallel criterion are left as scf.for.
// This development-only transformation is registered for explicit pass
// runners. The standard raising pipeline does not invoke it.
std::unique_ptr<::mlir::Pass> createSCFForToForallPass();

// Register all raising passes with the global pass registry. Lets
// `mlir-opt` style drivers expose them via --loom-llvm-cf-to-cf,
// --loom-lift-cf-to-scf, --loom-llvm-arith-to-arith.
void registerRaisingPasses();

// Append the standard Loom raising pipeline to the given pass manager:
//   loom-llvm-cf-to-cf
//   loom-lift-cf-to-scf
//   loom-llvm-arith-to-arith
//   loom-normalize-lifted-scf-exit
//   loom-scf-while-to-for
// Selected SCF optimization decisions are outside this pipeline.
void buildRaisingPipeline(::mlir::PassManager &pm);

} // namespace raising
} // namespace loom

#endif // LOOM_FRONTEND_RAISING_PASSES_H
