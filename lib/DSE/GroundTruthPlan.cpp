#include "DSE/GroundTruthPlan.h"

#include "DSE/HardwareImplementationEvaluationAcquisition.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Models/OpenRoadStaticFpa.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
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

llvm::Expected<EvidenceObligationTemplateRef>
findTemplate(llvm::ArrayRef<EvidenceObligationTemplate> templates,
             llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  for (std::size_t index = 0; index < templates.size(); ++index)
    if (templates[index].canonicalBytes() == canonicalBytes)
      return EvidenceObligationTemplateRef(static_cast<std::uint32_t>(index));
  return invalid("canonical obligation disappeared during ordering");
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

} // namespace

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
  auto trainingObligation = prepareOpenRoadStaticFpaEvidenceObligationTemplate(
      prototype, inputs.operatingConditions, kFpaMetrics,
      CalibrationPartitionRole::Training, baseConfig, artifactStore, blobStore);
  if (!trainingObligation)
    return trainingObligation.takeError();
  auto validationObligation =
      prepareOpenRoadStaticFpaEvidenceObligationTemplate(
          prototype, inputs.operatingConditions, kFpaMetrics,
          CalibrationPartitionRole::Validation, baseConfig, artifactStore,
          blobStore);
  if (!validationObligation)
    return validationObligation.takeError();
  auto heldOutObligation = prepareOpenRoadStaticFpaEvidenceObligationTemplate(
      prototype, inputs.operatingConditions, kFpaMetrics,
      CalibrationPartitionRole::HeldOut, baseConfig, artifactStore, blobStore);
  if (!heldOutObligation)
    return heldOutObligation.takeError();

  const std::vector<std::uint8_t> trainingKey(
      trainingObligation->canonicalBytes().begin(),
      trainingObligation->canonicalBytes().end());
  const std::vector<std::uint8_t> validationKey(
      validationObligation->canonicalBytes().begin(),
      validationObligation->canonicalBytes().end());
  const std::vector<std::uint8_t> heldOutKey(
      heldOutObligation->canonicalBytes().begin(),
      heldOutObligation->canonicalBytes().end());
  std::vector<EvidenceObligationTemplate> obligations = {
      std::move(*trainingObligation), std::move(*validationObligation),
      std::move(*heldOutObligation)};
  llvm::sort(obligations, [](const EvidenceObligationTemplate &lhs,
                             const EvidenceObligationTemplate &rhs) {
    return std::lexicographical_compare(
        lhs.canonicalBytes().begin(), lhs.canonicalBytes().end(),
        rhs.canonicalBytes().begin(), rhs.canonicalBytes().end());
  });
  auto trainingRef = findTemplate(obligations, trainingKey);
  auto validationRef = findTemplate(obligations, validationKey);
  auto heldOutRef = findTemplate(obligations, heldOutKey);
  if (!trainingRef)
    return trainingRef.takeError();
  if (!validationRef)
    return validationRef.takeError();
  if (!heldOutRef)
    return heldOutRef.takeError();

  if (llvm::Error error = validatePartitionRequests(
          *training, obligations[trainingRef->ordinal()], artifactStore,
          blobStore, *externalContracts))
    return std::move(error);
  if (llvm::Error error = validatePartitionRequests(
          *validation, obligations[validationRef->ordinal()], artifactStore,
          blobStore, *externalContracts))
    return std::move(error);
  if (llvm::Error error = validatePartitionRequests(
          *heldOut, obligations[heldOutRef->ordinal()], artifactStore,
          blobStore, *externalContracts))
    return std::move(error);

  auto trainingConfig =
      projectResolvedEvidenceObligationSetConfigView({*trainingRef});
  auto validationConfig =
      projectResolvedEvidenceObligationSetConfigView({*validationRef});
  auto heldOutConfig =
      projectResolvedEvidenceObligationSetConfigView({*heldOutRef});
  if (!trainingConfig)
    return trainingConfig.takeError();
  if (!validationConfig)
    return validationConfig.takeError();
  if (!heldOutConfig)
    return heldOutConfig.takeError();
  auto gate = QualityGatePolicy::get({});
  if (!gate)
    return gate.takeError();

  ResolvedConfig planConfig = baseConfig;
  planConfig.dse.modelAuthorizations = {
      {evaluation::models::openRoadStaticFpaModelDescriptorRef()}};
  planConfig.dse.evidenceObligationTemplates = std::move(obligations);
  planConfig.dse.objectiveCatalogs = {};
  planConfig.dse.qualityGatePolicies = {*gate};
  planConfig.dse.planNodes = {
      PromotePlanNodeDefinition{
          hardwareImplementationEvaluationPromotionAcquisitionDescriptor()
              .reference(),
          {ExactPlanArtifacts{std::move(*training)}},
          trainingConfig->canonicalViewBytes().vec(),
          trainingConfig->digest(),
          QualityGatePolicyRef(0),
          AllPassingSelection{},
          PromotePurpose::CandidateSelection},
      PromotePlanNodeDefinition{
          hardwareImplementationEvaluationPromotionAcquisitionDescriptor()
              .reference(),
          {ExactPlanArtifacts{std::move(*validation)}},
          validationConfig->canonicalViewBytes().vec(),
          validationConfig->digest(),
          QualityGatePolicyRef(0),
          AllPassingSelection{},
          PromotePurpose::CandidateSelection},
      PromotePlanNodeDefinition{
          hardwareImplementationEvaluationPromotionAcquisitionDescriptor()
              .reference(),
          {ExactPlanArtifacts{std::move(*heldOut)}},
          heldOutConfig->canonicalViewBytes().vec(),
          heldOutConfig->digest(),
          QualityGatePolicyRef(0),
          AllPassingSelection{},
          PromotePurpose::CandidateSelection}};
  auto admitted = projectResolvedDseConfigView(planConfig);
  if (!admitted)
    return admitted.takeError();
  return FpaGroundTruthCollectionPlan{std::move(planConfig),
                                      PlanOutputRef{0, 1}, PlanOutputRef{1, 1},
                                      PlanOutputRef{2, 1}};
}

} // namespace loom::dse
