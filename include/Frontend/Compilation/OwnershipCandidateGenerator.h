#ifndef LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H
#define LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Frontend/Raising/Passes.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

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

/// Typed decisions selected for one operation-owned Spatial candidate. An
/// explicit canonical index width is materialized into the child Structured
/// Program. Wider LLVM GEP indices are narrowed only when their complete
/// value domain is proven to fit that signed width; absence never selects an
/// ambient or Fabric-derived default.
struct OperationSpatialOwnershipOptions final {
  lowering::CanonicalDataflowLoweringOptions lowering;
  std::optional<unsigned> canonicalIndexWidth;
};

/// Enumerates ownership-free structured scopes in the exact parent
/// candidate's canonical operation order. These are search coordinates, not
/// accepted or ranked candidates: applying a concrete typed decision may
/// still fail legality, materialization, or exact-Fabric admission.
llvm::Expected<std::vector<StructuredEntityRef>>
enumerateOperationSpatialOwnershipScopes(
    const StructuredProgramCandidate &parent);

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

/// Materializes the explicit SpatialCore ownership choice for one exact
/// structured operation inside an ordinary LLVM callable. The operation must
/// sit outside any dataflow.thread, dataflow.graph, or loom.spatial_region,
/// own at least one region, and have no SSA result used outside itself. Every
/// external SSA live-in becomes an explicit input of one new private rank-zero
/// dataflow.thread holding one loom.spatial_region; the original LLVM callable
/// remains the ABI authority and launches that thread at the operation's exact
/// position. Fabric is used only for hard-negative actor-capability pruning;
/// this function performs no Mapping or QoR choice.
llvm::Expected<MaterializedOwnershipCandidate>
materializeOperationSpatialOwnership(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &operation,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const OperationSpatialOwnershipOptions &options = {});

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H
