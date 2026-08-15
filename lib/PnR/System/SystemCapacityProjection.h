#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMCAPACITYPROJECTION_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMCAPACITYPROJECTION_H

#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/ResourceCapacityVerification.h"
#include "PnR/System/SystemCandidateState.h"
#include "SystemPnrSearchDomainInternal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace loom::pnr::detail {

struct SystemCandidateCapacityProjectionView final {
  llvm::ArrayRef<PnrIndex> threadChoices;
  llvm::ArrayRef<PnrIndex> graphChoices;
  llvm::ArrayRef<SystemServiceRouteSelection> serviceRoutes;
  llvm::ArrayRef<SystemServiceRouteNodeSelection> serviceRouteNodes;
  llvm::ArrayRef<SystemServiceRouteSinkSelection> serviceRouteSinks;
  llvm::ArrayRef<SystemInstructionResourceUseSelection> instructionResourceUses;
  llvm::ArrayRef<SystemServiceResourceUseSelection> serviceResourceUses;
};

struct SystemCandidatePhysicalDemandProjection final {
  ::loom::mapping::detail::ResourceCapacityOveruseProjection capacity;
  ::loom::mapping::MappingProgressClosure progress;
  ::loom::mapping::detail::ResourcePhysicalTimingProjection timing;
};

class SystemCandidateProjectionCache;

struct SystemCandidateProjectionResult final {
  SystemCandidatePhysicalDemandProjection demand;
  std::shared_ptr<const SystemCandidateProjectionCache> cache;
};

struct SystemImportedRouteProjection final {
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals;
  std::uint32_t payloadWidthBits = 0;
};

class SystemCapacityModel final {
public:
  llvm::Expected<SystemCandidatePhysicalDemandProjection>
  project(const FrozenSystemPnrProblem &problem,
          SystemCandidateCapacityProjectionView candidate) const;

  llvm::Expected<SystemCandidateProjectionResult>
  projectWithCache(const FrozenSystemPnrProblem &problem,
                   SystemCandidateCapacityProjectionView candidate) const;

  llvm::Expected<SystemCandidateProjectionResult>
  projectRouteDelta(const FrozenSystemPnrProblem &problem,
                    SystemCandidateCapacityProjectionView candidate,
                    const SystemCandidateProjectionCache &previous) const;

  llvm::Expected<SystemCandidateProjectionResult>
  projectResourceDelta(const FrozenSystemPnrProblem &problem,
                       SystemCandidateCapacityProjectionView candidate,
                       const SystemCandidateProjectionCache &previous) const;

private:
  struct ImportedProgressUseProjection final {
    ::loom::fabric::FabricUsePatternRef pattern;
    std::string activationKey;
    std::vector<::dataflow::EventFamilyKey> triggerAlternatives;
    std::vector<::loom::mapping::MappingProgressCausalReleaseProjection>
        causalRelease;
  };

  struct ImportedGraphProgressProjection final {
    std::string graphKey;
    std::vector<::loom::mapping::MappingRouteProgressObligationProjection>
        routeObligations;
    std::vector<ImportedProgressUseProjection> uses;
  };

  struct ImportedProjection final {
    ArtifactIdentity mappingIdentity;
    std::vector<SystemImportedRouteProjection> routes;
    std::vector<ImportedGraphProgressProjection> graphProgress;
  };

  ::loom::mapping::detail::FrozenResourceCapacityIndex resources_;
  std::optional<::loom::mapping::FrozenMappingProgressModel> progressModel_;
  std::vector<std::string> patternPhysicalOwnerKeys_;
  std::vector<std::size_t> rootTraversalOrdinals_;
  std::vector<ImportedProjection> importedProjections_;
  std::vector<PnrIndex> coreTargetClasses_;
  std::vector<PnrIndex> mappingTargetClasses_;
  std::vector<std::string> graphKeys_;
  std::vector<::loom::mapping::SystemTransferRouteProgressDependency>
      serviceRouteProgressDependencies_;

  friend llvm::Expected<std::unique_ptr<SystemCapacityModel>>
  buildSystemCapacityModel(
      const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::fabric::FabricSystemRootView &,
      llvm::ArrayRef<::loom::fabric::AccCoreOccurrenceRef>,
      llvm::ArrayRef<PnrIndex>, llvm::ArrayRef<PnrIndex>,
      llvm::ArrayRef<SpatialCatalogEntry>,
      llvm::ArrayRef<FrozenSystemGraphExecutionDecision>,
      llvm::ArrayRef<FrozenSystemInstructionUsePatternDomain>,
      llvm::ArrayRef<FrozenSystemMemoryServiceBinding>,
      llvm::ArrayRef<FrozenSystemConsistencyUsePatternDomain>,
      llvm::ArrayRef<FrozenSystemServiceContext>,
      llvm::ArrayRef<FrozenSystemServiceLeg>,
      const FrozenEndpointRoutingTopology &);
};

llvm::Expected<std::unique_ptr<SystemCapacityModel>> buildSystemCapacityModel(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::loom::fabric::AccCoreOccurrenceRef> cores,
    llvm::ArrayRef<PnrIndex> coreTargetClasses,
    llvm::ArrayRef<PnrIndex> mappingTargetClasses,
    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog,
    llvm::ArrayRef<FrozenSystemGraphExecutionDecision> graphDecisions,
    llvm::ArrayRef<FrozenSystemInstructionUsePatternDomain>
        instructionUsePatterns,
    llvm::ArrayRef<FrozenSystemMemoryServiceBinding> memoryBindings,
    llvm::ArrayRef<FrozenSystemConsistencyUsePatternDomain>
        consistencyUsePatterns,
    llvm::ArrayRef<FrozenSystemServiceContext> serviceContexts,
    llvm::ArrayRef<FrozenSystemServiceLeg> serviceLegs,
    const FrozenEndpointRoutingTopology &routingTopology);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMCAPACITYPROJECTION_H
