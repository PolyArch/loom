#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSANALYSIS_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSANALYSIS_H

#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingProgressProjection.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {
class FabricSystemRootView;
}

namespace loom::mapping {

struct SystemMappingClosureProjection;

/// Immutable event-causality index derived from one Canonical Dataflow. A
/// search problem freezes the complete possible activation-event inventory
/// once; strict Mapping verification freezes the selected inventory. Neither
/// path retains the Dataflow MLIR view or invents another event relation.
class FrozenMappingProgressModel final {
public:
  const ArtifactIdentity &dataflowIdentity() const { return dataflowIdentity_; }

private:
  FrozenMappingProgressModel(
      ArtifactIdentity dataflowIdentity,
      std::map<std::string, std::uint32_t> eventOrdinals,
      std::vector<std::vector<std::uint32_t>> reverseEdges)
      : dataflowIdentity_(std::move(dataflowIdentity)),
        eventOrdinals_(std::move(eventOrdinals)),
        reverseEdges_(std::move(reverseEdges)) {}

  ArtifactIdentity dataflowIdentity_;
  std::map<std::string, std::uint32_t> eventOrdinals_;
  std::vector<std::vector<std::uint32_t>> reverseEdges_;

  friend llvm::Expected<FrozenMappingProgressModel>
  freezeMappingProgressModel(const ::dataflow::CanonicalDataflowProgramView &,
                             llvm::ArrayRef<::dataflow::EventFamilyKey>);
  friend llvm::Expected<MappingProgressClosure>
  deriveMappingProgressClosure(const FrozenMappingProgressModel &,
                               const MappingProgressProjection &);
  friend llvm::Expected<bool>
  mappingEventPrecedes(const FrozenMappingProgressModel &,
                       const ::dataflow::EventFamilyKey &,
                       const ::dataflow::EventFamilyKey &);
};

/// Freezes Dataflow causality after interning every possible trigger and
/// release event supplied by the caller. The inventory may be a conservative
/// superset, but every selected event must be present.
llvm::Expected<FrozenMappingProgressModel> freezeMappingProgressModel(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::EventFamilyKey> activationEvents);

/// Queries the one frozen Dataflow causality relation. Equality is reflexive;
/// a missing event is rejected rather than treated as independent. This is a
/// derived scheduling fact, not a Mapping progress proof.
llvm::Expected<bool>
mappingEventPrecedes(const FrozenMappingProgressModel &model,
                     const ::dataflow::EventFamilyKey &predecessor,
                     const ::dataflow::EventFamilyKey &dependent);

/// Derives the progress-only projection of the strict System closure. This is
/// the adapter used by final verification; System PnR constructs the same
/// projection directly from its frozen selection domains.
llvm::Expected<MappingProgressProjection> projectSystemMappingProgress(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemMappingClosureProjection &closure);

llvm::Expected<MappingProgressClosure>
deriveMappingProgressClosure(const FrozenMappingProgressModel &model,
                             const MappingProgressProjection &projection);

llvm::StringRef
mappingProgressClosureReasonSpelling(MappingProgressClosureReason reason);

/// Derives only the reusable Dataflow basis for exactly the supplied covered
/// graphs. It deliberately cannot return a Mapping progress proof: selected
/// routes, finite resources, service plans, arbitration, and Fabric progress
/// mechanisms must still be analyzed by the Mapping-level owner.
llvm::Expected<MappingDataflowProgressBasis> deriveMappingDataflowProgressBasis(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs);

/// Emits the already-derived basis and canonical residual cycle without
/// rerunning graph analysis. This presentation cannot change Mapping state.
void emitMappingDataflowProgressBasisDiagnostic(
    const MappingDataflowProgressBasis &basis,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    mapping_debug::Stage stage);

/// Completes the System progress proof from the one shared physical closure
/// projection. Resource uses that share an activation trigger are acquired as
/// one atomic group. Causal release dependencies and capacity blocking are
/// composed as an active-holder/pending-activation wait-for graph. A possible
/// cycle, fixed-priority starvation, or relation that cannot be reconstructed
/// returns ProofNotEstablished; an over-approximate cycle is never reported as
/// a proven deadlock.
llvm::Expected<MappingProgressClosure> deriveSystemMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemMappingClosureProjection &closure);

/// Qualifies one independently imported Mapping for resource-time endpoint
/// publication. The ordinary Mapping verifier already proved route/resource
/// closure. The qualifier invokes the same exact Dataflow/Fabric capacity,
/// arbitration, and causal-release kernel used by System verification; any
/// missing token/occupancy witness remains typed ProofNotEstablished rather
/// than being inferred from finite replay.
llvm::Expected<MappingProgressClosure>
qualifySystemMappingResourceTimeProgress(
    const FinalizedSystemMapping &mapping,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric);

struct SpatialRouteExternalSinkPrerequisite final {
  std::uint64_t sinkOrdinal = 0;
};

struct SpatialRouteInternalMemoryConnectionPrerequisite final {
  std::uint64_t memoryRealizationOrdinal = 0;
  std::uint64_t internalEdgeOrdinal = 0;
};

struct SpatialRouteInitializedFeedbackPrerequisite final {};

/// A route prerequisite is either another externally routed sink or one exact
/// realization-internal memory connection. Initialized feedback has no second
/// sink: it requires a durable disposition on the dependent edge itself. The
/// variants deliberately do not share an ordinal domain.
using SpatialRouteProgressPrerequisite =
    std::variant<SpatialRouteExternalSinkPrerequisite,
                 SpatialRouteInternalMemoryConnectionPrerequisite,
                 SpatialRouteInitializedFeedbackPrerequisite>;

/// One route-level wait dependency within a residual multicast net. Ordinals
/// address the canonical TechMapping inventories; this projection has no
/// persistent identity.
struct SpatialRouteProgressDependency final {
  std::uint64_t logicalNetOrdinal = 0;
  SpatialRouteProgressPrerequisite prerequisite =
      SpatialRouteExternalSinkPrerequisite{};
  std::uint64_t dependentSinkOrdinal = 0;
};

llvm::Expected<std::vector<SpatialRouteProgressDependency>>
deriveSpatialRouteProgressDependencies(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping);

llvm::Expected<MappingProgressProjection> projectSpatialMappingProgress(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<::dataflow::GraphRef> selectedGraphs);

struct SystemTransferRouteProgressDependency final {
  CanonicalServiceLegKey leg;
  ::dataflow::StructuralOrdinal prerequisiteSinkOrdinal = 0;
  ::dataflow::StructuralOrdinal dependentSinkOrdinal = 0;
};

llvm::Expected<std::vector<SystemTransferRouteProgressDependency>>
deriveSystemTransferRouteProgressDependencies(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<CanonicalServiceLegKey> transferLegs);

llvm::Expected<std::vector<MappingRouteProgressObligationProjection>>
projectSystemTransferRouteProgress(
    llvm::ArrayRef<SystemTransferLegView> transferLegs,
    llvm::ArrayRef<SystemTransferRouteProgressDependency> dependencies);

llvm::Expected<std::vector<MappingRouteProgressObligationProjection>>
projectSystemTransferRouteProgress(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<SystemTransferLegView> transferLegs);

/// Completes the reusable Dataflow basis against exact selected route trees.
/// A dependent multicast branch must cross a Buffered FIFO after it diverges
/// from every prerequisite branch; a FIFO on the shared prefix cannot release
/// the atomic fork.
llvm::Expected<MappingProgressClosure> deriveSpatialMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> operandQueueGroups =
        {});

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSANALYSIS_H
