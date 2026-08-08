#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMCAPACITYPROJECTION_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMCAPACITYPROJECTION_H

#include "Mapping/Artifact/ResourceCapacityVerification.h"
#include "PnR/System/SystemCandidateState.h"
#include "SystemPnrSearchDomainInternal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace loom::pnr::detail {

struct SystemCandidateCapacityProjectionView final {
  llvm::ArrayRef<PnrIndex> threadChoices;
  llvm::ArrayRef<PnrIndex> graphChoices;
  llvm::ArrayRef<SystemServiceRouteSelection> serviceRoutes;
  llvm::ArrayRef<SystemServiceRouteNodeSelection> serviceRouteNodes;
  llvm::ArrayRef<SystemInstructionResourceUseSelection> instructionResourceUses;
  llvm::ArrayRef<SystemServiceResourceUseSelection> serviceResourceUses;
};

class SystemCapacityModel final {
public:
  llvm::Expected<::loom::mapping::detail::ResourceCapacityOveruseProjection>
  project(const FrozenSystemPnrProblem &problem,
          SystemCandidateCapacityProjectionView candidate) const;

private:
  struct ImportedUseProjection final {
    ::loom::fabric::FabricUsePatternRef pattern;
    std::string activationKey;
  };

  struct ImportedProjection final {
    ArtifactIdentity mappingIdentity;
    std::vector<ImportedUseProjection> uses;
    std::vector<std::vector<::loom::fabric::FabricPhysicalTraversalRef>> routes;
  };

  ::loom::mapping::detail::FrozenResourceCapacityIndex resources_;
  std::vector<std::size_t> rootTraversalOrdinals_;
  std::vector<ImportedProjection> importedProjections_;
  std::vector<PnrIndex> coreTargetClasses_;
  std::vector<PnrIndex> mappingTargetClasses_;
  std::vector<std::string> graphKeys_;

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
    const FrozenEndpointRoutingTopology &routingTopology);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMCAPACITYPROJECTION_H
