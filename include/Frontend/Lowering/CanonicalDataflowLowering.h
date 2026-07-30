#ifndef LOOM_FRONTEND_LOWERING_CANONICALDATAFLOWLOWERING_H
#define LOOM_FRONTEND_LOWERING_CANONICALDATAFLOWLOWERING_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <vector>

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

/// One lowering-owned correspondence between a Structured Spatial ownership
/// carrier and the exact static graph launch mechanically derived from it.
/// Both references retain their respective Artifact owners; this disposable
/// relation is not a persistent identity or schema field.
struct StructuredSpatialGraphProjection final {
  frontend::StructuredEntityRef spatialRegion;
  dataflow::StaticGraphLaunchRef staticGraphLaunch;
};

struct ProjectedCanonicalDataflow final {
  dataflow::CanonicalDataflowArtifact artifact;
  std::vector<StructuredSpatialGraphProjection> spatialGraphs;
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

llvm::Expected<dataflow::CanonicalDataflowArtifact>
lowerStructuredProgramToCanonicalDataflow(
    const frontend::StructuredProgramCandidate &candidate,
    CanonicalDataflowLoweringOptions options = {});

/// Lowers an exact finalized Structured Program and returns the total ordered
/// correspondence from each loom.spatial_region to its static graph launch.
llvm::Expected<ProjectedCanonicalDataflow>
lowerStructuredProgramToCanonicalDataflowWithProjection(
    const frontend::StructuredProgramCandidate &candidate,
    CanonicalDataflowLoweringOptions options = {});

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_CANONICALDATAFLOWLOWERING_H
