#ifndef LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H
#define LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Frontend/Raising/Passes.h"

#include "llvm/Support/Error.h"

#include <optional>

namespace loom::frontend {

/// One ordinary child Structured Program and its mechanically derived D0
/// projection. The pair is returned only after both artifact owners finalize
/// successfully and the exact Fabric admits every canonical actor.
struct MaterializedOwnershipCandidate final {
  StructuredProgramCandidate structuredProgram;
  dataflow::CanonicalDataflowArtifact canonicalDataflow;
};

/// Typed decisions that must be materialized in the selected Structured
/// Program before the mechanical Dataflow boundary. An absent decision never
/// selects a default; a selected region that still contains such a choice
/// fails canonical publication.
struct WholeCallableSpatialOwnershipOptions final {
  lowering::CanonicalDataflowLoweringOptions lowering;
  std::optional<raising::FMulAddExecutionShape> fmuladdExecutionShape;
};

/// Materializes the explicit whole-callable SpatialCore ownership choice for
/// one exact void, single-block LLVM callable. Its original body moves into an
/// ordered thread while the same LLVM callable remains the ABI authority and
/// launches that thread. Fabric is used only for hard-negative actor-capability
/// pruning; this function performs no Mapping or QoR choice.
llvm::Expected<MaterializedOwnershipCandidate>
materializeWholeCallableSpatialOwnership(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &callable,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const WholeCallableSpatialOwnershipOptions &options = {});

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H
