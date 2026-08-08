#ifndef LOOM_PNR_SYSTEM_SYSTEMPNRPROBLEM_H
#define LOOM_PNR_SYSTEM_SYSTEMPNRPROBLEM_H

#include "Fabric/Artifact/FabricMemoryServiceClosure.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "PnR/EndpointRoutingTopology.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/PnrIndex.h"
#include "PnR/System/SystemPnrSearchDomain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::pnr {

namespace detail {
class InitializerRelationModel;
}

class FrozenSystemPnrProblem;
class SystemCandidateState;
struct InitializedSystemCandidate;
using SystemCandidateStateHandle = std::shared_ptr<SystemCandidateState>;
using FrozenSystemPnrProblemHandle =
    std::shared_ptr<const FrozenSystemPnrProblem>;

enum class SystemPnrFreezeFailureKind : std::uint32_t {
  Invalid,
  ProvenInfeasible,
};

class SystemPnrFreezeFailure final
    : public llvm::ErrorInfo<SystemPnrFreezeFailure> {
public:
  static char ID;

  SystemPnrFreezeFailure(SystemPnrFreezeFailureKind kind, std::string message)
      : kind_(kind), message_(std::move(message)) {}

  SystemPnrFreezeFailureKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SystemPnrFreezeFailureKind kind_;
  std::string message_;
};

struct FrozenSystemSpatialTargetClass final {
  ArtifactIdentity moduleIdentity;
  ::loom::fabric::FabricModuleTemplateRef moduleTemplate;
};

struct FrozenSystemThreadExecutionDecision final {
  ::dataflow::RootThreadLaunchRef root;
  ::loom::mapping::SystemPresburgerCell cell;
  PnrIndex choiceOffset = 0;
  PnrIndex choiceCount = 0;
  PnrIndex relationDecision = 0;
};

struct FrozenSystemGraphExecutionDecision final {
  ::dataflow::RootedGraphLaunchRef launch;
  ::loom::mapping::SystemPresburgerCell cell;
  PnrIndex choiceOffset = 0;
  PnrIndex choiceCount = 0;
  PnrIndex relationDecision = 0;
};

struct FrozenSystemTransferTerminal final {
  ::loom::mapping::SystemTransferTerminalKey key;
  bool fixedHostOwner = false;
  PnrIndex ownerThreadDecision = getInvalidPnrIndex();
  PnrIndex ownerDomainOffset = 0;
  PnrIndex ownerDomainCount = 0;
};

using FrozenSystemTransferTerminalOwner =
    std::variant<::loom::fabric::HostCoreOccurrenceRef,
                 ::loom::fabric::AccCoreOccurrenceRef>;

struct FrozenSystemTransferTerminalOwnerDomain final {
  FrozenSystemTransferTerminalOwner owner;
  PnrIndex endpointChoiceOffset = 0;
  PnrIndex endpointChoiceCount = 0;
};

struct FrozenSystemApplicableMessageSink final {
  ::dataflow::StructuralOrdinal sinkOrdinal = 0;
  PnrIndex ownerThreadDecision = getInvalidPnrIndex();

  friend bool operator==(const FrozenSystemApplicableMessageSink &lhs,
                         const FrozenSystemApplicableMessageSink &rhs) {
    return lhs.sinkOrdinal == rhs.sinkOrdinal &&
           lhs.ownerThreadDecision == rhs.ownerThreadDecision;
  }
};

struct FrozenSystemServiceContext final {
  PnrIndex service = 0;
  PnrIndex graphDecision = getInvalidPnrIndex();
  PnrIndex threadDecision = getInvalidPnrIndex();
  std::vector<::loom::mapping::SystemPresburgerCell> cells;
  std::vector<SystemServiceTargetSubject> subjects;
  std::vector<FrozenSystemApplicableMessageSink> applicableMessageSinks;
};

struct FrozenSystemMemoryServiceBinding final {
  ::loom::mapping::SystemServiceObligationKey obligation;
  SystemServiceTargetSubject subject;
  ArtifactRootReference spatialMapping;
  ::loom::fabric::AccCoreOccurrenceRef accCore;
  ::loom::fabric::SystemServiceEndpointRef systemEndpoint;
  ::loom::fabric::FabricMemoryEndpointRef occurrenceEndpoint;
  std::vector<::loom::fabric::FabricMemoryServiceTargetPlan> targetPlans;
  struct UsePatternDomain final {
    ::loom::fabric::FabricMemoryServiceRegionRef region;
    std::vector<::loom::fabric::FabricUsePatternRef> patterns;
  };
  std::vector<UsePatternDomain> usePatternDomains;
  std::optional<::loom::mapping::SpatialMemoryIntervalView> interval;
  std::optional<::loom::fabric::SubordinateEndpointRef> exposureTerminal;
};

struct FrozenSystemInstructionUsePatternDomain final {
  ::loom::fabric::InstructionCoreContextRef context;
  std::vector<::loom::fabric::FabricUsePatternRef> patterns;
};

struct FrozenSystemConsistencyUsePatternDomain final {
  ::loom::fabric::MemoryConsistencyDomainRef domain;
  std::vector<::loom::fabric::FabricUsePatternRef> patterns;
};

struct FrozenSystemServiceLeg final {
  ::loom::mapping::CanonicalServiceLegKey key;
  PnrIndex serviceContext = getInvalidPnrIndex();
  PnrIndex sourceTerminal = 0;
  PnrIndex sinkOffset = 0;
  PnrIndex sinkCount = 0;
  std::uint32_t requiredPayloadWidthBits = 0;
};

class FrozenSystemPnrProblem final {
public:
  FrozenSystemPnrProblem(const FrozenSystemPnrProblem &) = delete;
  FrozenSystemPnrProblem(FrozenSystemPnrProblem &&) = delete;
  FrozenSystemPnrProblem &operator=(const FrozenSystemPnrProblem &) = delete;
  FrozenSystemPnrProblem &operator=(FrozenSystemPnrProblem &&) = delete;
  ~FrozenSystemPnrProblem();

  const ArtifactIdentity &dataflowIdentity() const { return dataflowIdentity_; }
  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  const ArtifactIdentity &constraintIdentity() const {
    return constraintIdentity_;
  }
  const SystemPnrSearchDomainDigest &searchDomainDigest() const {
    return searchDomainDigest_;
  }
  const ResolvedPnrConfigView &config() const { return config_; }
  const MappingObjectiveProgram &objectiveProgram() const {
    return objectiveProgram_;
  }
  llvm::ArrayRef<DeterministicWorkBudgetEntry> workBudget() const {
    return workBudget_;
  }
  const ::loom::mapping::MappingProgressClosure &progressClosure() const {
    return progressClosure_;
  }
  llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches() const {
    return rootThreadLaunches_;
  }
  llvm::ArrayRef<FrozenSystemSpatialTargetClass> targetClasses() const {
    return targetClasses_;
  }
  llvm::ArrayRef<::loom::fabric::AccCoreOccurrenceRef> accCores() const {
    return accCores_;
  }
  llvm::ArrayRef<ArtifactRootReference> spatialMappings() const {
    return spatialMappings_;
  }
  llvm::ArrayRef<FrozenSystemThreadExecutionDecision> threadDecisions() const {
    return threadDecisions_;
  }
  llvm::ArrayRef<FrozenSystemGraphExecutionDecision> graphDecisions() const {
    return graphDecisions_;
  }
  const FrozenEndpointRoutingTopology &routingTopology() const {
    return routingTopology_;
  }
  llvm::ArrayRef<FrozenSystemTransferTerminal> serviceTerminals() const {
    return serviceTerminals_;
  }
  llvm::ArrayRef<FrozenSystemTransferTerminalOwnerDomain>
  serviceTerminalOwnerDomains(PnrIndex terminal) const;
  llvm::ArrayRef<PnrIndex> serviceTerminalOwnerEndpointChoices(
      const FrozenSystemTransferTerminalOwnerDomain &domain) const;
  llvm::ArrayRef<SystemSearchServiceDomain> serviceDomains() const {
    return serviceDomains_;
  }
  llvm::ArrayRef<FrozenSystemServiceContext> serviceContexts() const {
    return serviceContexts_;
  }
  llvm::ArrayRef<FrozenSystemMemoryServiceBinding>
  memoryServiceBindings() const {
    return memoryServiceBindings_;
  }
  llvm::ArrayRef<FrozenSystemInstructionUsePatternDomain>
  instructionUsePatternDomains() const {
    return instructionUsePatternDomains_;
  }
  llvm::ArrayRef<FrozenSystemConsistencyUsePatternDomain>
  consistencyUsePatternDomains() const {
    return consistencyUsePatternDomains_;
  }
  llvm::ArrayRef<FrozenSystemServiceLeg> serviceLegs() const {
    return serviceLegs_;
  }
  llvm::ArrayRef<PnrIndex> threadChoiceCatalogOrdinals(PnrIndex decision) const;
  llvm::ArrayRef<PnrIndex> graphChoiceCatalogOrdinals(PnrIndex decision) const;
  llvm::ArrayRef<PnrIndex> serviceLegSinkTerminals(PnrIndex leg) const;
  PnrIndex accCoreTargetClass(PnrIndex core) const;
  PnrIndex spatialMappingTargetClass(PnrIndex mapping) const;
  const detail::InitializerRelationModel &initializerRelations() const {
    return *initializerRelations_;
  }

private:
  FrozenSystemPnrProblem(
      ArtifactIdentity dataflowIdentity, ArtifactIdentity fabricIdentity,
      ArtifactIdentity constraintIdentity,
      SystemPnrSearchDomainDigest searchDomainDigest,
      ResolvedPnrConfigView config, MappingObjectiveProgram objectiveProgram,
      std::vector<DeterministicWorkBudgetEntry> workBudget,
      ::loom::mapping::MappingProgressClosure progressClosure,
      std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
      std::vector<FrozenSystemSpatialTargetClass> targetClasses,
      std::vector<::loom::fabric::AccCoreOccurrenceRef> accCores,
      std::vector<PnrIndex> accCoreTargetClasses,
      std::vector<ArtifactRootReference> spatialMappings,
      std::vector<PnrIndex> spatialMappingTargetClasses,
      std::vector<FrozenSystemThreadExecutionDecision> threadDecisions,
      std::vector<PnrIndex> threadChoiceCatalogOrdinals,
      std::vector<FrozenSystemGraphExecutionDecision> graphDecisions,
      std::vector<PnrIndex> graphChoiceCatalogOrdinals,
      std::vector<PnrIndex> graphThreadOverlapOffsets,
      std::vector<PnrIndex> graphThreadOverlaps,
      FrozenEndpointRoutingTopology routingTopology,
      std::vector<FrozenSystemTransferTerminal> serviceTerminals,
      std::vector<FrozenSystemTransferTerminalOwnerDomain>
          serviceTerminalOwnerDomains,
      std::vector<PnrIndex> serviceTerminalEndpointChoices,
      std::vector<SystemSearchServiceDomain> serviceDomains,
      std::vector<FrozenSystemServiceContext> serviceContexts,
      std::vector<FrozenSystemMemoryServiceBinding> memoryServiceBindings,
      std::vector<FrozenSystemInstructionUsePatternDomain>
          instructionUsePatternDomains,
      std::vector<FrozenSystemConsistencyUsePatternDomain>
          consistencyUsePatternDomains,
      std::vector<FrozenSystemServiceLeg> serviceLegs,
      std::vector<PnrIndex> serviceLegSinkTerminals,
      std::unique_ptr<detail::InitializerRelationModel> initializerRelations);

  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity fabricIdentity_;
  ArtifactIdentity constraintIdentity_;
  SystemPnrSearchDomainDigest searchDomainDigest_;
  ResolvedPnrConfigView config_;
  MappingObjectiveProgram objectiveProgram_;
  std::vector<DeterministicWorkBudgetEntry> workBudget_;
  ::loom::mapping::MappingProgressClosure progressClosure_;
  std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches_;
  std::vector<FrozenSystemSpatialTargetClass> targetClasses_;
  std::vector<::loom::fabric::AccCoreOccurrenceRef> accCores_;
  std::vector<PnrIndex> accCoreTargetClasses_;
  std::vector<ArtifactRootReference> spatialMappings_;
  std::vector<PnrIndex> spatialMappingTargetClasses_;
  std::vector<FrozenSystemThreadExecutionDecision> threadDecisions_;
  std::vector<PnrIndex> threadChoiceCatalogOrdinals_;
  std::vector<FrozenSystemGraphExecutionDecision> graphDecisions_;
  std::vector<PnrIndex> graphChoiceCatalogOrdinals_;
  std::vector<PnrIndex> graphThreadOverlapOffsets_;
  std::vector<PnrIndex> graphThreadOverlaps_;
  FrozenEndpointRoutingTopology routingTopology_;
  std::vector<FrozenSystemTransferTerminal> serviceTerminals_;
  std::vector<FrozenSystemTransferTerminalOwnerDomain>
      serviceTerminalOwnerDomains_;
  std::vector<PnrIndex> serviceTerminalEndpointChoices_;
  std::vector<SystemSearchServiceDomain> serviceDomains_;
  std::vector<FrozenSystemServiceContext> serviceContexts_;
  std::vector<FrozenSystemMemoryServiceBinding> memoryServiceBindings_;
  std::vector<FrozenSystemInstructionUsePatternDomain>
      instructionUsePatternDomains_;
  std::vector<FrozenSystemConsistencyUsePatternDomain>
      consistencyUsePatternDomains_;
  std::vector<FrozenSystemServiceLeg> serviceLegs_;
  std::vector<PnrIndex> serviceLegSinkTerminals_;
  std::unique_ptr<detail::InitializerRelationModel> initializerRelations_;

  friend class SystemCandidateState;
  friend llvm::Expected<InitializedSystemCandidate>
      initializeCanonicalSystemCandidate(FrozenSystemPnrProblemHandle);
  friend llvm::Expected<InitializedSystemCandidate>
  initializeSystemCandidateAttempt(FrozenSystemPnrProblemHandle, std::uint32_t);
  friend llvm::Expected<InitializedSystemCandidate>
      initializeSystemCandidateWithFixedChoices(FrozenSystemPnrProblemHandle,
                                                llvm::ArrayRef<PnrIndex>);
  friend llvm::Expected<SystemCandidateStateHandle>
  initializeSystemCandidate(FrozenSystemPnrProblemHandle,
                            llvm::ArrayRef<PnrIndex>, llvm::ArrayRef<PnrIndex>,
                            std::uint64_t *, std::uint64_t *);
  friend llvm::Expected<std::shared_ptr<const FrozenSystemPnrProblem>>
  freezeSystemPnrProblem(
      const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::fabric::FabricSystemRootView &,
      const SystemPnrSearchDomainView &, const ResolvedPnrConfigView &,
      const ::loom::mapping::FinalizedSystemMappingConstraintSet &,
      const ArtifactStore &);
};

llvm::Expected<FrozenSystemPnrProblemHandle> freezeSystemPnrProblem(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemPnrSearchDomainView &searchDomain,
    const ResolvedPnrConfigView &config,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const ArtifactStore &store);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMPNRPROBLEM_H
