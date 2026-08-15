#include "PnR/System/SystemPnrProblem.h"

#include "PnR/InitializerRelationSolver.h"
#include "SystemCapacityProjection.h"

#include <cassert>
#include <utility>

namespace loom::pnr {
namespace {

llvm::ArrayRef<PnrIndex> choiceSlice(llvm::ArrayRef<PnrIndex> choices,
                                     PnrIndex offset, PnrIndex count) {
  assert(offset <= choices.size());
  assert(count <= choices.size() - offset);
  return choices.slice(offset, count);
}

} // namespace

FrozenSystemPnrProblem::FrozenSystemPnrProblem(
    ArtifactIdentity dataflowIdentity, ArtifactIdentity fabricIdentity,
    ArtifactIdentity constraintIdentity,
    SystemPnrSearchDomainDigest searchDomainDigest,
    ResolvedPnrConfigView config, MappingObjectiveProgram objectiveProgram,
    std::vector<DeterministicWorkBudgetEntry> workBudget,
    ::loom::mapping::MappingDataflowProgressBasis progressBasis,
    std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
    std::vector<FrozenSystemSpatialTargetClass> targetClasses,
    std::vector<::loom::fabric::AccCoreOccurrenceRef> accCores,
    std::vector<PnrIndex> accCoreTargetClasses,
    std::vector<ArtifactRootReference> spatialMappings,
    std::vector<PnrIndex> spatialMappingTargetClasses,
    std::vector<std::uint64_t> spatialMappingWorstRouteArrivalDelayQuanta,
    std::vector<std::uint64_t> spatialMappingTotalRouteNegativeSlackQuanta,
    std::vector<ComponentViewDigest::Storage>
        spatialMappingPhysicalTimingProfileDigests,
    std::vector<::loom::fabric::FabricPhysicalTimingProfileKind>
        spatialMappingPhysicalTimingProfileKinds,
    std::vector<FrozenSystemThreadExecutionDecision> threadDecisions,
    std::vector<PnrIndex> threadChoiceCatalogOrdinals,
    std::vector<FrozenSystemGraphExecutionDecision> graphDecisions,
    std::vector<PnrIndex> graphChoiceCatalogOrdinals,
    std::vector<std::uint64_t> graphChoiceStaticSchedulePressures,
    std::vector<SpatialRecurrenceTimingProjection> graphChoiceRecurrenceTimings,
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
    std::unique_ptr<detail::SystemCapacityModel> capacityModel,
    std::unique_ptr<detail::InitializerRelationModel> initializerRelations)
    : dataflowIdentity_(std::move(dataflowIdentity)),
      fabricIdentity_(std::move(fabricIdentity)),
      constraintIdentity_(std::move(constraintIdentity)),
      searchDomainDigest_(std::move(searchDomainDigest)),
      config_(std::move(config)),
      objectiveProgram_(std::move(objectiveProgram)),
      workBudget_(std::move(workBudget)), progressBasis_(progressBasis),
      rootThreadLaunches_(std::move(rootThreadLaunches)),
      targetClasses_(std::move(targetClasses)), accCores_(std::move(accCores)),
      accCoreTargetClasses_(std::move(accCoreTargetClasses)),
      spatialMappings_(std::move(spatialMappings)),
      spatialMappingTargetClasses_(std::move(spatialMappingTargetClasses)),
      spatialMappingWorstRouteArrivalDelayQuanta_(
          std::move(spatialMappingWorstRouteArrivalDelayQuanta)),
      spatialMappingTotalRouteNegativeSlackQuanta_(
          std::move(spatialMappingTotalRouteNegativeSlackQuanta)),
      spatialMappingPhysicalTimingProfileDigests_(
          std::move(spatialMappingPhysicalTimingProfileDigests)),
      spatialMappingPhysicalTimingProfileKinds_(
          std::move(spatialMappingPhysicalTimingProfileKinds)),
      threadDecisions_(std::move(threadDecisions)),
      threadChoiceCatalogOrdinals_(std::move(threadChoiceCatalogOrdinals)),
      graphDecisions_(std::move(graphDecisions)),
      graphChoiceCatalogOrdinals_(std::move(graphChoiceCatalogOrdinals)),
      graphChoiceStaticSchedulePressures_(
          std::move(graphChoiceStaticSchedulePressures)),
      graphChoiceRecurrenceTimings_(std::move(graphChoiceRecurrenceTimings)),
      graphThreadOverlapOffsets_(std::move(graphThreadOverlapOffsets)),
      graphThreadOverlaps_(std::move(graphThreadOverlaps)),
      routingTopology_(std::move(routingTopology)),
      serviceTerminals_(std::move(serviceTerminals)),
      serviceTerminalOwnerDomains_(std::move(serviceTerminalOwnerDomains)),
      serviceTerminalEndpointChoices_(
          std::move(serviceTerminalEndpointChoices)),
      serviceDomains_(std::move(serviceDomains)),
      serviceContexts_(std::move(serviceContexts)),
      memoryServiceBindings_(std::move(memoryServiceBindings)),
      instructionUsePatternDomains_(std::move(instructionUsePatternDomains)),
      consistencyUsePatternDomains_(std::move(consistencyUsePatternDomains)),
      serviceLegs_(std::move(serviceLegs)),
      serviceLegSinkTerminals_(std::move(serviceLegSinkTerminals)),
      capacityModel_(std::move(capacityModel)),
      initializerRelations_(std::move(initializerRelations)) {}

FrozenSystemPnrProblem::~FrozenSystemPnrProblem() = default;

llvm::ArrayRef<PnrIndex>
FrozenSystemPnrProblem::threadChoiceCatalogOrdinals(PnrIndex decision) const {
  assert(decision < threadDecisions_.size());
  const auto &record = threadDecisions_[decision];
  return choiceSlice(threadChoiceCatalogOrdinals_, record.choiceOffset,
                     record.choiceCount);
}

llvm::ArrayRef<PnrIndex>
FrozenSystemPnrProblem::graphChoiceCatalogOrdinals(PnrIndex decision) const {
  assert(decision < graphDecisions_.size());
  const auto &record = graphDecisions_[decision];
  return choiceSlice(graphChoiceCatalogOrdinals_, record.choiceOffset,
                     record.choiceCount);
}

llvm::ArrayRef<std::uint64_t>
FrozenSystemPnrProblem::graphChoiceStaticSchedulePressures(
    PnrIndex decision) const {
  assert(decision < graphDecisions_.size());
  const auto &record = graphDecisions_[decision];
  return llvm::ArrayRef(graphChoiceStaticSchedulePressures_)
      .slice(record.choiceOffset, record.choiceCount);
}

llvm::ArrayRef<SpatialRecurrenceTimingProjection>
FrozenSystemPnrProblem::graphChoiceRecurrenceTimings(PnrIndex decision) const {
  assert(decision < graphDecisions_.size());
  const auto &record = graphDecisions_[decision];
  return llvm::ArrayRef(graphChoiceRecurrenceTimings_)
      .slice(record.choiceOffset, record.choiceCount);
}

llvm::ArrayRef<PnrIndex>
FrozenSystemPnrProblem::graphThreadOverlaps(PnrIndex decision) const {
  assert(decision < graphDecisions_.size());
  assert(decision + 1 < graphThreadOverlapOffsets_.size());
  const PnrIndex begin = graphThreadOverlapOffsets_[decision];
  const PnrIndex end = graphThreadOverlapOffsets_[decision + 1];
  assert(begin <= end && end <= graphThreadOverlaps_.size());
  return llvm::ArrayRef(graphThreadOverlaps_).slice(begin, end - begin);
}

const detail::SystemCapacityModel &
FrozenSystemPnrProblem::capacityModel() const {
  return *capacityModel_;
}

llvm::ArrayRef<FrozenSystemTransferTerminalOwnerDomain>
FrozenSystemPnrProblem::serviceTerminalOwnerDomains(PnrIndex terminal) const {
  assert(terminal < serviceTerminals_.size());
  const auto &record = serviceTerminals_[terminal];
  return llvm::ArrayRef(serviceTerminalOwnerDomains_)
      .slice(record.ownerDomainOffset, record.ownerDomainCount);
}

llvm::ArrayRef<PnrIndex>
FrozenSystemPnrProblem::serviceTerminalOwnerEndpointChoices(
    const FrozenSystemTransferTerminalOwnerDomain &domain) const {
  return choiceSlice(serviceTerminalEndpointChoices_,
                     domain.endpointChoiceOffset, domain.endpointChoiceCount);
}

llvm::ArrayRef<PnrIndex>
FrozenSystemPnrProblem::serviceLegSinkTerminals(PnrIndex leg) const {
  assert(leg < serviceLegs_.size());
  const auto &record = serviceLegs_[leg];
  return choiceSlice(serviceLegSinkTerminals_, record.sinkOffset,
                     record.sinkCount);
}

PnrIndex FrozenSystemPnrProblem::accCoreTargetClass(PnrIndex core) const {
  assert(core < accCoreTargetClasses_.size());
  return accCoreTargetClasses_[core];
}

PnrIndex
FrozenSystemPnrProblem::spatialMappingTargetClass(PnrIndex mapping) const {
  assert(mapping < spatialMappingTargetClasses_.size());
  return spatialMappingTargetClasses_[mapping];
}

} // namespace loom::pnr
