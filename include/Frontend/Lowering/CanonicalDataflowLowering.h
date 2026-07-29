#ifndef LOOM_FRONTEND_LOWERING_CANONICALDATAFLOWLOWERING_H
#define LOOM_FRONTEND_LOWERING_CANONICALDATAFLOWLOWERING_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir

namespace loom::lowering {

/// Controls the mechanical Structured Program-to-Dataflow publication path.
/// Dataflow-only candidate transforms and Mapping remain outside this boundary.
struct CanonicalDataflowLoweringOptions {
  bool verifyEach = true;
  bool applyPassManagerCommandLineOptions = false;
};

/// Returns a lowering-owned reason when `scope` contains a structural
/// operation that neither graph-region lowering nor a mandatory typed
/// ownership decision can implement. This is a conservative preflight:
/// absence of a reason does not prove finalizability, while a returned reason
/// identifies an unsupported capability before candidate materialization.
std::optional<std::string>
explainGraphRegionStructuralRejection(mlir::Operation *scope);

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
