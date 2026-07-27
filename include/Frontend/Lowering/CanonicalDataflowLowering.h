#ifndef LOOM_FRONTEND_LOWERING_CANONICALDATAFLOWLOWERING_H
#define LOOM_FRONTEND_LOWERING_CANONICALDATAFLOWLOWERING_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace loom::lowering {

/// Controls the mechanical Structured Program-to-Dataflow publication path.
/// Dataflow-only candidate transforms and Mapping remain outside this boundary.
struct CanonicalDataflowLoweringOptions {
  bool verifyEach = true;
  bool applyPassManagerCommandLineOptions = false;
};

/// Runs the standard mechanical SCF-to-Dataflow transaction in place. This
/// supports focused developer tools and leaves Artifact publication to callers
/// that hold a complete root-closed candidate.
llvm::Error
lowerStructuredModuleInPlace(mlir::ModuleOp module,
                             CanonicalDataflowLoweringOptions options = {});

/// Lowers a private clone of one complete Structured Program snapshot through
/// the standard mechanical SCF-to-Dataflow transaction and finalizes D0.
llvm::Expected<dataflow::CanonicalDataflowArtifact>
lowerStructuredModuleToCanonicalDataflow(
    mlir::ModuleOp module, CanonicalDataflowLoweringOptions options = {});

inline llvm::Expected<dataflow::CanonicalDataflowArtifact>
lowerStructuredProgramToCanonicalDataflow(
    const frontend::StructuredProgramCandidate &candidate,
    CanonicalDataflowLoweringOptions options = {}) {
  return lowerStructuredModuleToCanonicalDataflow(candidate.module(), options);
}

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_CANONICALDATAFLOWLOWERING_H
