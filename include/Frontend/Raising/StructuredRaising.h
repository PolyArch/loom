#ifndef LOOM_FRONTEND_RAISING_STRUCTUREDRAISING_H
#define LOOM_FRONTEND_RAISING_STRUCTUREDRAISING_H

#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

#include <memory>

namespace llvm {
class Module;
} // namespace llvm

namespace loom::raising {

/// Controls mechanical LLVM-to-S0 raising only. Candidate-generating SCF
/// transforms intentionally remain outside this boundary.
struct StructuredRaisingOptions {
  bool allowUnregisteredDialects = false;
  bool verifyEach = true;
  bool applyPassManagerCommandLineOptions = false;
};

/// Projects only exact, signature-preserving callback targets proven by the
/// pinned LLVM interprocedural analysis. All other proof-clone rewrites are
/// discarded, so this remains a mechanical LLVM normalization.
llvm::Error normalizeProvenConstantCallbacks(llvm::Module &module);

/// Clones a defined dispatcher for each distinct set of exact function
/// constants passed to callback formals that are used directly as callees.
/// Each clone replaces only those formal uses, and each original call site is
/// redirected to the matching clone. Unknown callback values remain indirect.
llvm::Error specializeExactConstantCallbackCallSites(llvm::Module &module);

/// Imports one verified LLVM module, performs the exact mechanical raising
/// pipeline, and publishes the resulting immutable Structured Program view.
llvm::Expected<frontend::StructuredProgramCandidate>
raiseLlvmModuleToStructuredProgram(std::unique_ptr<llvm::Module> module,
                                   StructuredRaisingOptions options = {});

/// Runs the same mechanical raising transaction while retaining the
/// invocation-local source provenance projected before finalization erases
/// locations from canonical bytes.
llvm::Expected<frontend::FinalizedStructuredProgramProjection>
raiseLlvmModuleToStructuredProgramWithProjection(
    std::unique_ptr<llvm::Module> module,
    StructuredRaisingOptions options = {});

} // namespace loom::raising

#endif // LOOM_FRONTEND_RAISING_STRUCTUREDRAISING_H
