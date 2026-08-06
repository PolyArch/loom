#ifndef LOOM_PNR_SYSTEM_SYSTEMPNRPROBLEM_H
#define LOOM_PNR_SYSTEM_SYSTEMPNRPROBLEM_H

#include "Mapping/Artifact/SystemMappingConstraintSet.h"
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
  SystemPresburgerCell cell;
  PnrIndex choiceOffset = 0;
  PnrIndex choiceCount = 0;
  PnrIndex relationDecision = 0;
};

struct FrozenSystemGraphExecutionDecision final {
  ::dataflow::RootedGraphLaunchRef launch;
  SystemPresburgerCell cell;
  PnrIndex choiceOffset = 0;
  PnrIndex choiceCount = 0;
  PnrIndex relationDecision = 0;
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
  llvm::ArrayRef<PnrIndex> threadChoiceCatalogOrdinals(PnrIndex decision) const;
  llvm::ArrayRef<PnrIndex> graphChoiceCatalogOrdinals(PnrIndex decision) const;
  PnrIndex accCoreTargetClass(PnrIndex core) const;
  PnrIndex spatialMappingTargetClass(PnrIndex mapping) const;

private:
  FrozenSystemPnrProblem(
      ArtifactIdentity dataflowIdentity, ArtifactIdentity fabricIdentity,
      ArtifactIdentity constraintIdentity,
      SystemPnrSearchDomainDigest searchDomainDigest,
      ResolvedPnrConfigView config,
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
      std::unique_ptr<detail::InitializerRelationModel> initializerRelations);

  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity fabricIdentity_;
  ArtifactIdentity constraintIdentity_;
  SystemPnrSearchDomainDigest searchDomainDigest_;
  ResolvedPnrConfigView config_;
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
  std::unique_ptr<detail::InitializerRelationModel> initializerRelations_;

  friend class SystemCandidateState;
  friend llvm::Expected<InitializedSystemCandidate>
      initializeCanonicalSystemCandidate(FrozenSystemPnrProblemHandle);
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
