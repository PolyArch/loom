#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGCLOSUREPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGCLOSUREPROJECTION_H

#include "Common/ExecutionControl.h"
#include "Mapping/Artifact/MappingProgressProjection.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include "Fabric/IR/UsePatternValue.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::mapping {

struct SystemCapacityCellProjection final {
  ::loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOwner;
  ::fabric::StateKey state{0};
  ::fabric::CapacityDimensionKey dimension{0};
  std::uint64_t capacity = 0;
  std::uint64_t baselineOccupancy = 0;
};

struct SystemCapacityClaimProjection final {
  std::uint64_t capacityCellOrdinal = 0;
  std::uint64_t amount = 0;
};

struct SystemCausalReleasePointProjection final {
  std::vector<::dataflow::EventFamilyKey> alternatives;
  std::optional<std::vector<std::uint8_t>> guaranteedOffset;
};

/// One exact ResourceUse projected into a reachable execution context. The
/// relation domain uses the same logical signature as its event projection.
/// Several trigger alternatives denote one acquisition opportunity.
struct SystemResourceActivationProjection final {
  ExecutionContextKey context;
  std::vector<SystemPresburgerCell> relationDomain;
  std::vector<::dataflow::EventFamilyKey> triggerAlternatives;
  ::loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOwner;
  ::loom::fabric::FabricOrdinal usePatternOrdinal = 0;
  std::vector<::fabric::UsePatternValue> parameters;
  std::vector<::fabric::UsePatternValue> sharingAssignments;
  std::vector<SystemCapacityClaimProjection> capacityClaims;
  std::vector<SystemCausalReleasePointProjection> causalRelease;
};

/// The one removable projection shared by SystemMapping verification,
/// Deployment, Runtime, and simulator bridges. All dense ordinals are local
/// to this value and are derived from canonical physical keys.
struct SystemMappingClosureProjection final {
  SystemExecutionContextProjection executionContexts;
  std::vector<SystemServiceRealizationView> serviceRealizations;
  MappingDataflowProgressBasis progressBasis;
  std::vector<MappingRouteProgressObligationProjection> routeObligations;
  std::vector<SystemCapacityCellProjection> capacityCells;
  std::vector<SystemResourceActivationProjection> resourceActivations;
};

/// Rebase one SpatialMapping-relative activation event into a selected rooted
/// graph execution. These helpers are the sole event projection used by both
/// System PnR and strict SystemMapping closure reconstruction.
llvm::Expected<std::vector<::dataflow::EventFamilyKey>>
projectSystemSpatialActivityEvent(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef graph,
    const SpatialActivityEventRef &event);

llvm::Expected<::dataflow::GraphRef> resolveSystemSpatialActivityEventGraph(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const SpatialActivityEventRef &event);

llvm::Expected<std::vector<SystemCausalReleasePointProjection>>
projectSystemSpatialCausalRelease(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<SpatialEventPointView> release);

llvm::Expected<SystemMappingClosureProjection> projectSystemMappingClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemMappingView &mapping, const ArtifactStore &store,
    const SpatialMappingImportContext *spatialMappings = nullptr,
    ExecutionControlView executionControl = {});

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGCLOSUREPROJECTION_H
