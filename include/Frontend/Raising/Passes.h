#ifndef LOOM_FRONTEND_RAISING_PASSES_H
#define LOOM_FRONTEND_RAISING_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class PassManager;
} // namespace mlir

namespace loom {
namespace raising {

// Convert llvm.br / llvm.cond_br terminators inside llvm.func bodies into
// the matching cf.br / cf.cond_br ops. Pattern-rewrite based, runs only
// inside llvm.func regions and intentionally does not replace llvm.return
// (the function-body terminator stays). llvm.switch is not converted in
// this design and is reported as a no-op for that op so the rest of the
// pipeline can still proceed; the function will simply remain in llvm form.
std::unique_ptr<::mlir::Pass> createLLVMCfToCfPass();

// Convert llvm.* arithmetic, comparison and integer/float constant ops into
// the matching arith dialect ops:
//   integer:    add, sub, mul, sdiv, udiv, srem, urem,
//               shl, lshr, ashr, and, or, xor.
//   float:      fadd, fsub, fmul, fdiv, frem.
//   compares:   icmp -> arith.cmpi, fcmp -> arith.cmpf (predicate kept).
//   constants:  llvm.mlir.constant -> arith.constant when the type is a
//               builtin integer or float.
// Any operation whose operands or results carry non-builtin types
// (pointers, structs, vectors with non-builtin element types, ...) is
// skipped and remains in llvm form. Pointer arithmetic via
// llvm.getelementptr stays as llvm by design.
std::unique_ptr<::mlir::Pass> createLLVMArithToArithPass();

// Uplift counted scf.while loops produced by --lift-cf-to-scf into
// scf.for. This is a thin wrapper around the upstream
// scf::populateUpliftWhileToForPatterns. Loops that do not match the
// counted shape are left as scf.while.
std::unique_ptr<::mlir::Pass> createSCFWhileToForPass();

// Lift trivially parallel scf.for loops (no iter_args, iv-dependent
// stores only, no nested calls or while) into scf.forall. Loops that do
// not match the conservative parallel criterion are left as scf.for.
// The produced scf.forall carries no shared_outs and no mapping
// attribute -- downstream device-mapping passes can attach one later.
std::unique_ptr<::mlir::Pass> createSCFForToForallPass();

// For each llvm.func whose signature is composed entirely of MLIR-native
// types (builtin integers, floats, index, !llvm.ptr), create a sibling
// func.func with the same name, move the body region over, replace the
// llvm.return terminator on each block ending with one with func.return,
// and erase the original llvm.func.
//
// A function is *skipped* (left as llvm.func) when:
//   * it is variadic;
//   * it has no body (declaration only);
//   * any argument type is a non-pointer aggregate type (struct, array,
//     non-builtin vector);
//   * the result type is a non-pointer aggregate type;
//   * a callee referenced by an llvm.call inside the body is itself going
//     to be skipped.
//
// This pass must run AFTER createLLVMCfToCfPass() so the body branch
// terminators are already cf.br / cf.cond_br; otherwise the upstream
// --lift-cf-to-scf pass that runs next will not see anything to lift.
std::unique_ptr<::mlir::Pass> createLLVMFuncToFuncPass();

// Register all raising passes with the global pass registry. Lets
// `mlir-opt` style drivers expose them via --loom-llvm-cf-to-cf,
// --loom-llvm-arith-to-arith, --loom-llvm-func-to-func.
void registerRaisingPasses();

// Append the standard Loom raising pipeline to the given pass manager:
//   loom-llvm-cf-to-cf
//   loom-llvm-func-to-func
//   --lift-cf-to-scf       (upstream)
//   loom-llvm-arith-to-arith
//   --canonicalize         (upstream)
//   loom-scf-while-to-for
//   --canonicalize         (upstream)
//   loom-scf-for-to-forall
//   --canonicalize         (upstream)
void buildRaisingPipeline(::mlir::PassManager &pm);

} // namespace raising
} // namespace loom

#endif // LOOM_FRONTEND_RAISING_PASSES_H
