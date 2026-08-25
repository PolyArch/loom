#include "DSE/GroundTruthPlan.h"

#include "DSE/HardwareImplementationEvaluationAcquisition.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Models/OpenRoadStaticFpa.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

constexpr std::array<evaluation::MetricKind, 4> kFpaMetrics = {
    evaluation::MetricKind::LimitingClockFrequency,
    evaluation::MetricKind::TotalArea, evaluation::MetricKind::DynamicPower,
    evaluation::MetricKind::LeakagePower};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fpa_ground_truth_plan_invalid: " + message);
}

llvm::Expected<std::vector<ArtifactRootReference>> canonicalCandidates(
    std::vector<ArtifactRootReference> candidates, llvm::StringRef partition,
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    const hardware::ExternalImplementationContractCatalog &externalContracts) {
  if (candidates.empty())
    return invalid(partition + " partition is empty");
  llvm::sort(candidates, artifactRootReferenceLess);
  if (std::adjacent_find(candidates.begin(), candidates.end()) !=
      candidates.end())
    return invalid(partition + " partition contains a duplicate HImpl");
  for (const ArtifactRootReference &candidate : candidates) {
    auto implementation = hardware::importHardwareImplementation(
        candidate, externalContracts, artifactStore, blobStore);
    if (!implementation)
      return implementation.takeError();
    const hardware::ImplementationRepresentationRoot &root =
        implementation->implementation().representationRoot();
    if (root.variant != hardware::RepresentationRootVariant::AsicPhysical ||
        root.stage != hardware::RepresentationPhysicalStage::Routed)
      return invalid(partition + " partition contains a non-routed ASIC HImpl");
    if (!implementation->implementation().implementationPlatform())
      return invalid(partition + " partition contains an HImpl without an "
                                 "ImplementationPlatform");
  }
  return candidates;
}

llvm::Error validatePartitionRequests(
    llvm::ArrayRef<ArtifactRootReference> candidates,
    const EvidenceObligationTemplate &obligation,
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    const hardware::ExternalImplementationContractCatalog &externalContracts) {
  for (const ArtifactRootReference &candidate : candidates) {
    auto resolution =
        evaluation::models::resolveHardwareImplementationPhysicalCase(
            candidate, externalContracts, artifactStore, blobStore);
    if (!resolution)
      return resolution.takeError();
    auto request = instantiateEvidenceObligation(
        obligation, candidate, {}, 0, *resolution, artifactStore, blobStore);
    if (!request)
      return request.takeError();
    auto complete =
        evaluation::models::projectCompleteOpenRoadStaticFpaConfiguration(
            *request, *resolution, artifactStore, blobStore);
    if (!complete)
      return complete.takeError();
  }
  return llvm::Error::success();
}

llvm::Error
validateCharacterizationLeaf(const fabric::FabricModuleDomainMemberRef &leaf,
                             const fabric::FabricArtifactView &module) {
  switch (leaf.kind()) {
  case fabric::FabricModuleDomainMemberKind::Boundary: {
    const auto &boundary =
        std::get<fabric::FabricModuleBoundaryEndpointRef>(leaf.payload);
    if (llvm::Error error = fabric::validateFabricRef(module, boundary))
      return error;
    if (module.moduleBoundaryEndpointPlane(boundary) !=
        fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport)
      return invalid("characterization boundary is not a transport leaf");
    return llvm::Error::success();
  }
  case fabric::FabricModuleDomainMemberKind::Internal: {
    const auto &owner =
        std::get<fabric::FabricModulePhysicalOwnerRef>(leaf.payload);
    switch (owner.kind()) {
    case fabric::FabricModulePhysicalOwnerKind::PeOccurrence:
    case fabric::FabricModulePhysicalOwnerKind::FuOccurrence:
    case fabric::FabricModulePhysicalOwnerKind::MemoryOccurrence:
    case fabric::FabricModulePhysicalOwnerKind::SwitchOccurrence:
      break;
    default:
      return invalid("characterization owner is not a PE, FU, memory, or "
                     "switch leaf");
    }
    return std::visit(
        [&](const auto &value) {
          return fabric::validateFabricRef(module, value);
        },
        owner.payload());
  }
  }
  llvm_unreachable("unknown Module domain member kind");
}

} // namespace

llvm::Expected<FpaCharacterizationUnavailable>
assessFpaLeafCharacterizationTarget(const FpaLeafCharacterizationTarget &target,
                                    const ArtifactStore &artifactStore,
                                    const BlobStore &blobStore) {
  auto contracts = eda::makeKnownAsicStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  auto imported = hardware::importHardwareImplementation(
      target.hardwareImplementation, *contracts, artifactStore, blobStore);
  if (!imported)
    return imported.takeError();
  const hardware::HardwareImplementation &implementation =
      imported->implementation();
  auto fabricRoot =
      fabric::importEntireFabricRoot(implementation.fabric(), artifactStore);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto system = fabric::requireSystemRoot(fabricRoot->view());
  if (!system)
    return system.takeError();
  const auto selected =
      system->spatialCoreTarget(implementation.subject().core);
  if (!selected || selected->dependencyOrdinal >=
                       system->artifact().importedModules().size())
    return invalid("implementation subject has no imported SpatialCore Module");
  const fabric::FabricArtifactView &module =
      system->artifact().importedModules()[selected->dependencyOrdinal];
  if (llvm::Error error = validateCharacterizationLeaf(target.leaf, module))
    return std::move(error);

  const hardware::ImplementationRepresentationRoot &representation =
      implementation.representationRoot();
  if (representation.variant !=
          hardware::RepresentationRootVariant::AsicPhysical ||
      representation.stage != hardware::RepresentationPhysicalStage::Routed ||
      !implementation.implementationPlatform())
    return FpaCharacterizationUnavailable{
        target, FpaCharacterizationUnavailableReason::
                    RoutedAsicImplementationUnavailable};
  return FpaCharacterizationUnavailable{
      target,
      FpaCharacterizationUnavailableReason::IndependentlyRoutedLeafUnavailable};
}

llvm::Expected<FpaGroundTruthCollectionPlan> buildFpaGroundTruthCollectionPlan(
    FpaGroundTruthPlanInputs inputs, const ResolvedConfig &baseConfig,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (llvm::Error error =
          registerHardwareImplementationEvaluationPromotionAcquisition())
    return std::move(error);
  auto externalContracts = eda::makeKnownAsicStandardCellContractCatalog();
  if (!externalContracts)
    return externalContracts.takeError();
  auto training = canonicalCandidates(
      std::move(inputs.training.hardwareImplementations), "Training",
      artifactStore, blobStore, *externalContracts);
  if (!training)
    return training.takeError();
  auto validation = canonicalCandidates(
      std::move(inputs.validation.hardwareImplementations), "Validation",
      artifactStore, blobStore, *externalContracts);
  if (!validation)
    return validation.takeError();
  auto heldOut = canonicalCandidates(
      std::move(inputs.heldOut.hardwareImplementations), "HeldOut",
      artifactStore, blobStore, *externalContracts);
  if (!heldOut)
    return heldOut.takeError();

  std::map<std::vector<std::uint8_t>, CalibrationPartitionRole> groupOwners;
  const auto admitGroups =
      [&](llvm::ArrayRef<ArtifactRootReference> candidates,
          CalibrationPartitionRole partition) -> llvm::Error {
    for (const ArtifactRootReference &candidate : candidates) {
      auto key = evaluation::models::deriveFpaSampleGroupKey(
          candidate, artifactStore, blobStore);
      if (!key)
        return key.takeError();
      auto [found, inserted] = groupOwners.emplace(*key, partition);
      if (!inserted && found->second != partition)
        return invalid("one FPA sample group occurs in multiple partitions");
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          admitGroups(*training, CalibrationPartitionRole::Training))
    return std::move(error);
  if (llvm::Error error =
          admitGroups(*validation, CalibrationPartitionRole::Validation))
    return std::move(error);
  if (llvm::Error error =
          admitGroups(*heldOut, CalibrationPartitionRole::HeldOut))
    return std::move(error);

  const ArtifactRootReference &prototype = training->front();
  auto obligation = prepareOpenRoadStaticFpaEvidenceObligationTemplate(
      prototype, inputs.operatingConditions, kFpaMetrics, std::nullopt,
      baseConfig, artifactStore, blobStore);
  if (!obligation)
    return obligation.takeError();

  if (llvm::Error error = validatePartitionRequests(
          *training, *obligation, artifactStore, blobStore,
          *externalContracts))
    return std::move(error);
  if (llvm::Error error = validatePartitionRequests(
          *validation, *obligation, artifactStore, blobStore,
          *externalContracts))
    return std::move(error);
  if (llvm::Error error = validatePartitionRequests(
          *heldOut, *obligation, artifactStore, blobStore,
          *externalContracts))
    return std::move(error);

  auto collectionConfig = projectResolvedEvidenceObligationSetConfigView(
      {EvidenceObligationTemplateRef(0)});
  if (!collectionConfig)
    return collectionConfig.takeError();
  auto gate = QualityGatePolicy::get({});
  if (!gate)
    return gate.takeError();

  ResolvedConfig planConfig = baseConfig;
  planConfig.dse.modelAuthorizations = {
      {evaluation::models::openRoadStaticFpaModelDescriptorRef()}};
  planConfig.dse.evidenceObligationTemplates = {std::move(*obligation)};
  planConfig.dse.objectiveCatalogs = {};
  planConfig.dse.qualityGatePolicies = {*gate};
  planConfig.dse.planNodes = {
      PromotePlanNodeDefinition{
          hardwareImplementationEvaluationPromotionAcquisitionDescriptor()
              .reference(),
          {ExactPlanArtifacts{std::move(*training)}},
          collectionConfig->canonicalViewBytes().vec(),
          collectionConfig->digest(),
          QualityGatePolicyRef(0), AllPassingSelection{},
          PromotePurpose::CandidateSelection},
      PromotePlanNodeDefinition{
          hardwareImplementationEvaluationPromotionAcquisitionDescriptor()
              .reference(),
          {ExactPlanArtifacts{std::move(*validation)}},
          collectionConfig->canonicalViewBytes().vec(),
          collectionConfig->digest(), QualityGatePolicyRef(0),
          AllPassingSelection{}, PromotePurpose::CandidateSelection},
      PromotePlanNodeDefinition{
          hardwareImplementationEvaluationPromotionAcquisitionDescriptor()
              .reference(),
          {ExactPlanArtifacts{std::move(*heldOut)}},
          collectionConfig->canonicalViewBytes().vec(),
          collectionConfig->digest(),
          QualityGatePolicyRef(0), AllPassingSelection{},
          PromotePurpose::CandidateSelection}};
  auto admitted = projectResolvedDseConfigView(planConfig);
  if (!admitted)
    return admitted.takeError();
  return FpaGroundTruthCollectionPlan{std::move(planConfig),
                                      PlanOutputRef{0, 1}, PlanOutputRef{1, 1},
                                      PlanOutputRef{2, 1}};
}

llvm::Expected<CampaignExecutionPolicy> makeFpaGroundTruthCampaignPolicy(
    std::uint64_t pilotDispatchCount,
    std::uint64_t minimumObservedPilotWorkUnits,
    std::uint64_t sampleActiveWallTimeLimitNanoseconds) {
  return CampaignExecutionPolicy::get(
      pilotDispatchCount, minimumObservedPilotWorkUnits,
      sampleActiveWallTimeLimitNanoseconds,
      CampaignExecutionPolicy::maximumCampaignActiveWallTimeNanoseconds);
}

llvm::Expected<CampaignExecutionResult>
runFpaGroundTruthCampaign(const ResolvedDseConfigView &view,
                          const DseRunClosure &closure,
                          const CampaignExecutionPolicy &campaignPolicy,
                          const PlanExecutionPolicy &executionPolicy,
                          SiteScheduler &scheduler, ExecutionJournal &journal,
                          const ArtifactStore &store, const BlobStore &blobs) {
  return runGroundTruthCampaign(view, closure, campaignPolicy, executionPolicy,
                                scheduler, journal, store, blobs);
}

} // namespace loom::dse
