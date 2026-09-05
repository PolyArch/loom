#include "DSE/JointDesignExploration.h"
#include "ADG/Builtin.h"
#include "Application/Build.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/HardwareDecision.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/JointMappingMigration.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Deployment/Package.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/Models/CalibratedFpa.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/CgraClosedWait.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/FabricLowConfidence.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "JointDesignExplorationFixture.h"
#include "JointDesignMutationTest.h"
#include "JointHardwareReopenExecution.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "joint design exploration anchor failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

namespace joint_fixture = loom::dse::joint_test;

/// Selects which bounded-quality sections run. The four sections share one
/// fixture and one bounded quality policy but are otherwise independent, so a
/// selector lets them run as parallel processes. Empty disables them;
/// `allJointDesignTestSections` (the shared "*" selector) runs all four in
/// order.
void exerciseJointExploration(bool runFifoHardwareRepair,
                              bool runRuntimeWitnessBudget,
                              bool runOperandHardwareRepair,
                              bool runTransportRepair,
                              llvm::StringRef qualitySection,
                              llvm::StringRef mutationFamily) {
  const auto qualityRuns = [qualitySection](llvm::StringRef section) {
    return qualitySection == joint_fixture::allJointDesignTestSections ||
           qualitySection == section;
  };
  // One import session for the whole anchor, matching what the loom-dse and
  // loom-system-run entry points install; nested production sessions defer to
  // it through ReuseEnclosing.
  loom::fabric::FabricArtifactImportSession fabricImportSession;
  joint_fixture::TemporaryDirectory temporary;
  llvm::SmallString<128> blobPath(temporary.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  loom::ArtifactStore store(temporary.path());
  loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = joint_fixture::makeContext();

  auto first = joint_fixture::buildDataflow(context, 7);
  auto second = joint_fixture::buildDataflow(context, 11);
  take(dataflow::publishCanonicalDataflow(first, store));
  take(dataflow::publishCanonicalDataflow(second, store));
  const loom::ArtifactRootReference firstWorkload =
      joint_fixture::publishApplicationWorkload(first, store);
  const loom::ArtifactRootReference secondWorkload =
      joint_fixture::publishApplicationWorkload(second, store);
  auto small = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto alternate = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Coverage));
  if (small.roots().size() != 1 || alternate.roots().size() != 1)
    fail("builtin fixture did not publish one complete System");
  const loom::ArtifactRootReference system = small.roots().front().reference();
  const loom::ArtifactRootReference alternateSystem =
      alternate.roots().front().reference();
  auto systemArtifact =
      take(loom::fabric::importEntireFabricRoot(system, store));
  auto systemView =
      take(loom::fabric::requireSystemRoot(systemArtifact.view()));
  const std::uint64_t analyticClockPeriodPicoseconds =
      take(loom::evaluation::models::fabricLowConfidenceClockPeriodPicoseconds(
          systemArtifact));
  auto timingProfiles = take(
      loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(systemView));
  std::vector<loom::ArtifactRootReference> timingProfileRoots;
  for (const auto &profile : timingProfiles)
    timingProfileRoots.push_back(
        take(loom::fabric::publishFabricPhysicalTimingProfile(profile, store)));

  const loom::dse::JointDesignPolicy policy =
      take(loom::dse::JointDesignPolicy::get(2, 1, 1, 2, 32));
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  config.hardwareTarget = {loom::adg::builtinSmallTarget.templateIdentity.str(),
                           {loom::adg::builtinSmallTarget.schemaMajor,
                            loom::adg::builtinSmallTarget.schemaMinor},
                           loom::adg::builtinSmallTarget.scale};
  config.dse.techMapping.candidatePublicationLimit = 4;
  auto plan = take(loom::dse::buildJointDesignExplorationPlan(
      {{{firstWorkload}, {secondWorkload}}, {system}}, timingProfileRoots,
      policy, config, store));
  if (plan.frontier.eligiblePairCount != 2 || !plan.frontier.truncated ||
      plan.frontier.pairs.size() != 1 || plan.pairOutputs.size() != 1)
    fail("bounded pair frontier did not declare deterministic truncation");
  if (plan.frontier.analyticEvaluatedPairCount != 2 ||
      plan.frontier.analyticDeferredPairCount != 1 ||
      plan.frontier.pairProjections.size() != 1 ||
      plan.frontier.pairProjections.front().softwareActorCount == 0 ||
      plan.frontier.pairProjections.front().systemAccCoreCount == 0)
    fail("analytic pair funnel lost bounded ranking evidence");
  if (plan.pairOutputs.front().techMappings.empty() ||
      plan.pairOutputs.front().spatialMappings.empty())
    fail("joint Mapping plan lost an intermediate result projection");
  const auto &systemNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
      plan.resolvedConfig.dse.planNodes
          [plan.pairOutputs.front().systemMappings.producerNodeOrdinal]);
  const auto &join =
      std::get<loom::dse::BoundedPlanOutputJoin>(systemNode.inputBindings[1]);
  if (join.outputs.empty() || join.maximumArtifacts != 32)
    fail("joint Mapping plan lost its explicit SpatialMapping bound");
  for (const loom::dse::PlanOutputRef &spatialOutput : join.outputs) {
    const auto &spatialNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
        plan.resolvedConfig.dse.planNodes[spatialOutput.producerNodeOrdinal]);
    const auto &techJoin = std::get<loom::dse::BoundedPlanOutputJoin>(
        spatialNode.inputBindings.front());
    if (techJoin.outputs.size() != 1 || techJoin.maximumArtifacts != 2)
      fail("joint Mapping plan lost its TechMapping admission bound");
    const auto &techOutput = techJoin.outputs.front();
    const auto &techNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
        plan.resolvedConfig.dse.planNodes[techOutput.producerNodeOrdinal]);
    if (techNode.descriptor !=
        loom::dse::applicationGraphTechMappingCandidateGeneratorDescriptor()
            .reference())
      fail("joint Mapping plan used a whole-program TechMapping cover");
  }

  auto view =
      take(loom::dse::projectResolvedDseConfigView(plan.resolvedConfig));
  auto execution = take(loom::dse::executeDsePlan(view, store, blobs));
  const loom::dse::CompletedDsePlanExecution *completed =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&execution);
  if (!completed) {
    const auto &incomplete =
        std::get<loom::dse::IncompleteDsePlanExecution>(execution);
    const auto *reason =
        std::get_if<loom::dse::CandidateGeneratorIncompleteReason>(
            &incomplete.reason());
    if (!reason ||
        *reason != loom::dse::CandidateGeneratorIncompleteReason::
                       SemanticLimitReached ||
        incomplete.executionStopped())
      fail("joint Mapping plan changed retained frontier semantics: " +
           loom::dse::toString(incomplete.reason()));
    completed = &incomplete.availableExecution();
  }
  const std::vector<loom::ArtifactRootReference> mappings =
      completed->resolve(plan.pairOutputs.front().systemMappings).vec();
  if (mappings.empty())
    fail("joint Mapping plan produced no complete SystemMapping");
  for (const loom::ArtifactRootReference &reference : mappings) {
    auto mapping = take(loom::mapping::importSystemMapping(reference, store));
    if (mapping.view().dataflowIdentity() !=
            plan.frontier.pairs.front().software.dataflow.artifact ||
        mapping.view().fabricIdentity() != system.artifact)
      fail("joint Mapping output lost its exact pair owners");
  }

  std::optional<loom::dse::JointBoundedQualityPolicy> incompleteRepairQuality;
  std::optional<loom::dse::JointBoundedQualityPolicy> qualityPolicy;
  std::shared_ptr<std::uint64_t> qualityAcquisitionCount;
  if (!qualitySection.empty()) {
    if (llvm::Error error =
            loom::evaluation::registerProductionEvaluationRegistry())
      fail(llvm::toString(std::move(error)));
    loom::dse::CandidateMeasureObjectiveCatalogs objectiveCatalogs;
    objectiveCatalogs.dimensions = {
        {0, loom::ResolvedObjectiveDirection::Minimize, 0, 100}};
    objectiveCatalogs.weightedLevels = {{{{0, 1}}}};
    objectiveCatalogs.totalOrderings = {{{0}}};
    auto objectiveProgram = take(
        loom::dse::ObjectiveProgram::getCandidateMeasures(objectiveCatalogs));
    auto sharedObjectiveProgram =
        std::make_shared<const loom::dse::ObjectiveProgram>(
            std::move(objectiveProgram));
    qualityAcquisitionCount = std::make_shared<std::uint64_t>(0);

    loom::dse::JointBoundedQualityPolicy quality;
    quality.objectiveProgram = sharedObjectiveProgram;
    quality.objectiveDimensionLabels = {"mapping_quality"};
    quality.paretoDimensions = {0};
    quality.finalTotalOrdering = 0;
    quality.acquire = [sharedObjectiveProgram, system,
                       count = qualityAcquisitionCount](
                          const loom::dse::JointDesignExecution &result,
                          std::uint64_t planOrdinal)
        -> llvm::Expected<loom::dse::JointDesignQualityAcquisition> {
      ++*count;
      if (!result.summary.selectedMapping)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "quality fixture has no Mapping");
      loom::dse::ObjectiveVector objective =
          sharedObjectiveProgram->makeVector();
      const bool hardwareChild = llvm::any_of(
          result.mappedPairs, [&](const loom::dse::JointMappedPair &pair) {
            return pair.pair.system != system;
          });
      const std::array<std::uint64_t, 1> measures = {hardwareChild ? UINT64_C(0)
                                                     : planOrdinal == 1
                                                         ? UINT64_C(1)
                                                         : UINT64_C(2)};
      if (llvm::Error error = sharedObjectiveProgram->evaluateCandidateMeasures(
              measures, objective))
        return std::move(error);
      return loom::dse::JointDesignQualityAcquisition{
          std::vector<loom::dse::JointDesignQualityCandidate>{
              {{*result.summary.selectedMapping, std::move(objective)},
               std::nullopt,
               {{loom::resolvedObjectiveInteger(measures[0])}, {}, {}}}}};
    };
    quality.hardwarePromotion = loom::dse::JointHardwarePromotionQualityPolicy{
        sharedObjectiveProgram,
        {"predicted_mapping_quality"},
        0,
        [sharedObjectiveProgram](
            const loom::dse::JointDesignExplorationPlan &candidate,
            std::uint64_t planOrdinal)
            -> llvm::Expected<loom::dse::JointDesignQualityAcquisition> {
          if (candidate.frontier.systemFrontier.size() != 1)
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "promotion fixture has no exact System");
          loom::dse::ObjectiveVector objective =
              sharedObjectiveProgram->makeVector();
          const std::array<std::uint64_t, 1> measures = {
              planOrdinal == 1 ? UINT64_C(0) : UINT64_C(1)};
          if (llvm::Error error =
                  sharedObjectiveProgram->evaluateCandidateMeasures(measures,
                                                                    objective))
            return std::move(error);
          return loom::dse::JointDesignQualityAcquisition{
              std::vector<loom::dse::JointDesignQualityCandidate>{
                  {{candidate.frontier.systemFrontier.front(),
                    std::move(objective)},
                   std::nullopt,
                   {{loom::resolvedObjectiveInteger(measures[0])}, {}, {}}}}};
        }};
    quality.maximumHardwareSpectrumParents = 1;
    quality.maximumHardwareRepairProbes = 1;
    qualityPolicy = quality;
    incompleteRepairQuality = quality;
    incompleteRepairQuality->acquire =
        [](const loom::dse::JointDesignExecution &result, std::uint64_t)
        -> llvm::Expected<loom::dse::JointDesignQualityAcquisition> {
      if (!result.summary.selectedMapping)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "incomplete quality fixture has no "
                                       "Mapping");
      return loom::dse::JointDesignQualityAcquisition{
          loom::dse::IncompleteJointDesignQuality{
              loom::dse::JointDesignQualityIncompleteReason::Unsupported,
              result.summary.selectedMapping, std::nullopt}};
    };
  }
  if (qualityRuns("promotion")) {
    const loom::dse::JointDesignPolicy promotionPolicy =
        take(loom::dse::JointDesignPolicy::get(2, 2, 1, 2, 32));
    auto firstPlan = take(loom::dse::buildJointDesignExplorationPlan(
        {{{firstWorkload}}, {system}}, timingProfileRoots, promotionPolicy,
        config, store));
    auto secondPlan = take(loom::dse::buildJointDesignExplorationPlan(
        {{{secondWorkload}}, {system}}, timingProfileRoots, promotionPolicy,
        config, store));
    const std::array promotionPlans = {&firstPlan, &secondPlan};

    auto alternateRoot =
        take(loom::fabric::importEntireFabricRoot(alternateSystem, store));
    auto alternateView =
        take(loom::fabric::requireSystemRoot(alternateRoot.view()));
    auto alternateTiming =
        take(loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(
            alternateView));
    std::vector<loom::ArtifactRootReference> alternateTimingRoots;
    for (const auto &profile : alternateTiming)
      alternateTimingRoots.push_back(take(
          loom::fabric::publishFabricPhysicalTimingProfile(profile, store)));
    auto alternatePlan = take(loom::dse::buildJointDesignExplorationPlan(
        {{{secondWorkload}}, {alternateSystem}}, alternateTimingRoots,
        promotionPolicy, config, store));
    auto outside = take(loom::adg::buildBuiltinTarget(
        store, loom::adg::BuiltinTargetPreset::Large));
    if (outside.roots().size() != 1)
      fail("out-of-domain FPA fixture did not publish one System");
    const loom::ArtifactRootReference outsideSystem =
        outside.roots().front().reference();
    auto outsideRoot =
        take(loom::fabric::importEntireFabricRoot(outsideSystem, store));
    auto outsideView =
        take(loom::fabric::requireSystemRoot(outsideRoot.view()));
    auto outsideTiming =
        take(loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(
            outsideView));
    std::vector<loom::ArtifactRootReference> outsideTimingRoots;
    for (const auto &profile : outsideTiming)
      outsideTimingRoots.push_back(take(
          loom::fabric::publishFabricPhysicalTimingProfile(profile, store)));
    auto outsidePlan = take(loom::dse::buildJointDesignExplorationPlan(
        {{{secondWorkload}}, {outsideSystem}}, outsideTimingRoots,
        promotionPolicy, config, store));

    const auto firstFeatures = joint_fixture::projectFpaFeatures(
        firstPlan.frontier.softwareFrontier.front().dataflow, system, config,
        store, blobs);
    const auto alternateFeatures = joint_fixture::projectFpaFeatures(
        alternatePlan.frontier.softwareFrontier.front().dataflow,
        alternateSystem, config, store, blobs);
    const loom::evaluation::models::FpaMetricPredictionView firstObservation{
        take(loom::evaluation::DecimalValue::get(1, 8)),
        take(loom::evaluation::DecimalValue::get(1, -6)),
        take(loom::evaluation::DecimalValue::get(1, -3)),
        take(loom::evaluation::DecimalValue::get(1, -4))};
    const loom::evaluation::models::FpaMetricPredictionView
        alternateObservation{take(loom::evaluation::DecimalValue::get(5, 8)),
                             take(loom::evaluation::DecimalValue::get(2, -6)),
                             take(loom::evaluation::DecimalValue::get(2, -3)),
                             take(loom::evaluation::DecimalValue::get(2, -4))};
    auto parameters = take(loom::evaluation::models::trainFpaGbdtParameters(
        {{firstFeatures, firstObservation, {0x21}, {0x31}},
         {alternateFeatures, alternateObservation, {0x21}, {0x32}}},
        loom::evaluation::models::FpaGbdtTrainingConfig{13, 4, 2, 1, 1, 1}));
    auto bundle = take(loom::evaluation::finalizeModelParameterBundle(
        loom::evaluation::models::fpaModelParameterContractRef(),
        loom::evaluation::OwnerValue::get(std::move(parameters)), store,
        blobs));
    const std::array<std::uint8_t, 1> digestOwner = {0x41};
    const std::array<std::uint8_t, 1> digestValue = {0x42};
    const loom::ComponentViewDigest candidateDigest =
        take(loom::computeComponentViewDigest(digestOwner, digestValue));
    std::vector<loom::application::PreparedApplicationMappingAlternative>
        applicationAlternatives = {
            {0,
             0,
             candidateDigest,
             candidateDigest,
             {},
             firstPlan.frontier.softwareFrontier.front().dataflow,
             {},
             {},
             firstPlan},
            {1,
             1,
             candidateDigest,
             candidateDigest,
             {},
             alternatePlan.frontier.softwareFrontier.front().dataflow,
             {},
             {},
             alternatePlan}};
    loom::application::PreparedApplicationBuild preparedApplication{
        {},
        promotionPolicy,
        {},
        {},
        {},
        {},
        {},
        {},
        0,
        false,
        {},
        {},
        {},
        {},
        {},
        std::move(applicationAlternatives),
        {},
        {},
        loom::dse::StructuredOwnershipSelectionMode::SemanticConformance,
        loom::dse::StructuredOwnershipSelectionMode::SemanticConformance,
        {},
        std::nullopt,
        std::nullopt,
        firstPlan.frontier.softwareFrontier.front().dataflow,
        system,
        firstWorkload,
        firstWorkload,
        candidateDigest,
        systemView.artifact().accCoreOccurrences().size(),
        analyticClockPeriodPicoseconds,
        std::nullopt,
        std::nullopt,
        bundle.reference(),
        {}};
    const auto qualityPolicyExecution =
        take(loom::dse::PlanExecutionPolicy::get(
            32, take(loom::dse::SiteResourceClaim::get(1, 0, 0))));
    auto applicationQuality =
        take(loom::application::makeApplicationBoundedQualityPolicy(
            preparedApplication, qualityPolicyExecution, store, blobs));
    if (!applicationQuality.hardwarePromotion ||
        applicationQuality.objectiveDimensionLabels.size() != 7 ||
        applicationQuality.semanticInputs.size() != 3 ||
        !llvm::is_contained(applicationQuality.semanticInputs,
                            bundle.reference()))
      fail("application quality policy lost its frozen FPA closure");
    if (llvm::count_if(applicationQuality.semanticInputs, [](const auto &root) {
          return root.schemaIdentity ==
                     loom::evaluation::EvaluationRequest::artifactSchema
                         .identity &&
                 root.schemaVersion ==
                     loom::evaluation::EvaluationRequest::artifactSchema
                         .version;
        }) != 2)
      fail("application quality closure lost an exact FPA Request");
    auto firstPhysical =
        take(applicationQuality.hardwarePromotion->acquire(firstPlan, 0));
    auto alternatePhysical =
        take(applicationQuality.hardwarePromotion->acquire(alternatePlan, 1));
    auto *firstPhysicalObjectives =
        std::get_if<std::vector<loom::dse::JointDesignQualityCandidate>>(
            &firstPhysical);
    auto *alternatePhysicalObjectives =
        std::get_if<std::vector<loom::dse::JointDesignQualityCandidate>>(
            &alternatePhysical);
    if (!firstPhysicalObjectives || firstPhysicalObjectives->size() != 1 ||
        !alternatePhysicalObjectives ||
        alternatePhysicalObjectives->size() != 1 ||
        !firstPhysicalObjectives->front().evidence ||
        !alternatePhysicalObjectives->front().evidence)
      fail("application FPA promotion did not publish completed Evidence");
    const loom::ArtifactRootReference sharedQualityEvidence =
        *firstPhysicalObjectives->front().evidence;
    for (const auto *objectives :
         {firstPhysicalObjectives, alternatePhysicalObjectives}) {
      const loom::ArtifactRootReference &evidence =
          *objectives->front().evidence;
      if (evidence.schemaIdentity !=
              loom::evaluation::EvaluationEvidence::artifactSchema.identity ||
          evidence.schemaVersion !=
              loom::evaluation::EvaluationEvidence::artifactSchema.version)
        fail("application FPA promotion returned a foreign Evidence root");
      if (objectives->front().provenance.runtimeCompletion !=
              loom::dse::JointDesignQualityRuntimeCompletion::NotEstablished ||
          objectives->front().provenance.calibratedModelSupport !=
              loom::dse::JointDesignCalibratedModelSupport::InDomain)
        fail("application FPA promotion lost its typed model support");
      take(store.get(evidence));
    }
    auto outsidePhysical =
        take(applicationQuality.hardwarePromotion->acquire(outsidePlan, 2));
    const auto *outsideIncomplete =
        std::get_if<loom::dse::IncompleteJointDesignQuality>(&outsidePhysical);
    if (!outsideIncomplete ||
        outsideIncomplete->reason !=
            loom::dse::JointDesignQualityIncompleteReason::Unsupported ||
        !outsideIncomplete->evidence ||
        outsideIncomplete->provenance.runtimeCompletion !=
            loom::dse::JointDesignQualityRuntimeCompletion::NotEstablished ||
        outsideIncomplete->provenance.calibratedModelSupport !=
            loom::dse::JointDesignCalibratedModelSupport::OutOfDomain)
      fail("application FPA out-of-domain refusal lost its typed provenance");
    auto weight = take(loom::evaluation::models::importEdaPredictionModelWeight(
        bundle.reference(), store, blobs));
    auto outsideRequest =
        take(loom::evaluation::models::
                 prepareCanonicalDataflowFabricCalibratedFpaEvaluation(
                     outsidePlan.frontier.softwareFrontier.front().dataflow,
                     outsideSystem, weight, {}, outsidePlan.resolvedConfig,
                     store, blobs));
    auto strictOutsideEvidence =
        take(loom::evaluation::importEvaluationEvidence(
            *outsideIncomplete->evidence, outsideRequest.resolution, store,
            blobs));
    if (strictOutsideEvidence.requestRef() !=
            loom::evaluation::evaluationRequestReference(
                outsideRequest.request) ||
        strictOutsideEvidence.outcomeKind() !=
            loom::evaluation::EvidenceOutcomeKind::Unsupported)
      fail("application FPA out-of-domain Evidence failed strict import");
    auto physicalOrder =
        take(applicationQuality.hardwarePromotion->objectiveProgram
                 ->compareTotalOrdering(
                     alternatePhysicalObjectives->front().objective.objective,
                     loom::encodeArtifactRootReference(alternateSystem),
                     firstPhysicalObjectives->front().objective.objective,
                     loom::encodeArtifactRootReference(system), 0));
    if (physicalOrder >= 0)
      fail("frozen FPA predictions did not rank the better physical plan");

    const loom::dse::JointBoundedQualityPolicy repairQuality = *qualityPolicy;
    llvm::SmallString<128> promotionJournal(temporary.path());
    llvm::sys::path::append(promotionJournal, "hardware-quality-promotion");
    auto promoted = take(loom::dse::executeJointDesignWithHardwareReopen(
        promotionPlans, promotionPolicy,
        {take(loom::dse::DseProducerSemanticBuildIdentity::get(
             "loom.test.hardware_quality_promotion.v1")),
         promotionJournal.str().str(),
         {},
         loom::dse::JointDesignStoppingPolicy::BoundedQuality,
         std::move(*qualityPolicy),
         5,
         take(loom::dse::SiteCapacity::get(2, 0, 0)),
         take(loom::dse::PlanExecutionPolicy::get(
             32, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
        store, blobs));
    if (promoted.summary.hardwareParentPromotions != 1 ||
        promoted.summary.hardwareReopensDeferredByQuality == 0 ||
        promoted.summary.hardwarePromotionObservations.size() != 2)
      fail("bounded hardware promotion lost its exact work ledger");
    if (!promoted.invocationManifest() ||
        promoted.supportingInvocationManifests().empty())
      fail("bounded hardware promotion lost its production DSE occurrences");
    take(loom::dse::importJointDesignInvocationManifest(
        *promoted.invocationManifest(), store, blobs));
    auto contentOnly = take(loom::dse::InvocationManifestReference::get(
        promoted.invocationManifest()->resolvedConfig(),
        promoted.invocationManifest()->blob(),
        promoted.invocationManifest()->occurrence(), store, blobs));
    if (contentOnly.journalReceipt())
      fail("strict manifest import forged an ExecutionJournal receipt");
    llvm::Error foreignBinding =
        loom::dse::appendJointDesignSupportingInvocationManifest(
            promoted, std::move(contentOnly));
    if (!foreignBinding)
      fail("joint execution accepted a manifest without its journal receipt");
    const std::string foreignBindingMessage =
        llvm::toString(std::move(foreignBinding));
    if (foreignBindingMessage.find("no ExecutionJournal receipt") ==
        std::string::npos)
      fail("foreign manifest rejection lost its journal ownership reason");
    for (const auto &supporting : promoted.supportingInvocationManifests())
      take(loom::dse::importJointDesignInvocationManifest(supporting, store,
                                                          blobs));
    for (const auto &observation :
         promoted.summary.hardwarePromotionObservations) {
      const bool expectedPromotion = observation.planOrdinal == 1;
      if (observation.promotedToExactMapping != expectedPromotion ||
          observation.incompleteReason ||
          observation.objectiveCodes.size() != 1)
        fail("bounded hardware promotion ignored its objective order");
    }
    if (!promoted.summary.selectedMapping)
      fail("bounded hardware promotion produced no selected Mapping");
    const loom::ArtifactRootReference promotedMapping =
        *promoted.summary.selectedMapping;
    if (!promoted.summary.selectedPlanOrdinal)
      fail("bounded hardware promotion lost its selected plan");
    const auto promotedParent = llvm::find_if(
        promoted.summary.hardwarePromotionObservations,
        [](const loom::dse::JointHardwarePromotionObservation &observation) {
          return observation.promotedToExactMapping;
        });
    if (promotedParent == promoted.summary.hardwarePromotionObservations.end())
      fail("bounded hardware promotion lost its selected parent");
    const std::size_t selectedChildAttempts = llvm::count_if(
        promoted.summary.attempts,
        [&](const loom::dse::JointDesignAttemptRecord &attempt) {
          return attempt.planOrdinal == *promoted.summary.selectedPlanOrdinal &&
                 attempt.hardwarePromotionParentSystem ==
                     promotedParent->system &&
                 attempt.system != promotedParent->system &&
                 attempt.disposition ==
                     loom::dse::JointDesignAttemptDisposition::Verified &&
                 llvm::is_contained(attempt.systemMappings, promotedMapping);
        });
    if (selectedChildAttempts != 1)
      fail("selected Mapping lost its exact promoted-parent child lineage");
    llvm::SmallString<128> repairJournal(temporary.path());
    llvm::sys::path::append(repairJournal, "repair-quality-selection");
    const loom::dse::JointDesignExplorationPlan *firstRepairPlan = &firstPlan;
    auto firstRepair = take(loom::dse::executeJointDesignWithHardwareReopen(
        llvm::ArrayRef<const loom::dse::JointDesignExplorationPlan *>(
            &firstRepairPlan, 1),
        promotionPolicy,
        {take(loom::dse::DseProducerSemanticBuildIdentity::get(
             "loom.test.repair_quality_selection.v1")),
         repairJournal.str().str(),
         {},
         loom::dse::JointDesignStoppingPolicy::BoundedQuality,
         repairQuality,
         5,
         take(loom::dse::SiteCapacity::get(2, 0, 0)),
         take(loom::dse::PlanExecutionPolicy::get(
             32, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
        store, blobs));
    if (!firstRepair.summary.selectedMapping ||
        firstRepair.summary.qualityDisposition !=
            loom::dse::JointDesignQualityDisposition::Complete)
      fail("single-plan repair quality execution produced no exact Mapping");
    const loom::ArtifactRootReference firstRepairMapping =
        *firstRepair.summary.selectedMapping;
    std::vector<loom::dse::JointDesignExecution> repairExecutions;
    repairExecutions.push_back(std::move(firstRepair));
    repairExecutions.push_back(std::move(promoted));
    const std::uint64_t acquisitionsBeforeRepairSelection =
        *qualityAcquisitionCount;
    auto repairSelection = take(loom::dse::selectJointRepairMappingByQuality(
        repairExecutions, repairQuality, store));
    const auto *selectedRepair =
        std::get_if<loom::dse::JointRepairQualitySelection>(&repairSelection);
    if (!selectedRepair || selectedRepair->executionOrdinal != 1 ||
        selectedRepair->mapping != promotedMapping)
      fail("repair quality selection ignored the shared objective order");
    if (*qualityAcquisitionCount != acquisitionsBeforeRepairSelection)
      fail("repair quality selection repeated an acquired objective");
    bool foundFirstEvidenceDomain = false;
    bool foundSecondEvidenceDomain = false;
    std::vector<loom::dse::JointDesignQualityObservation *>
        sharedEvidenceObservations;
    for (loom::dse::JointDesignExecution &execution : repairExecutions)
      for (loom::dse::JointDesignQualityObservation &observation :
           execution.summary.qualityObservations) {
        if (observation.candidate == firstRepairMapping)
          foundFirstEvidenceDomain = true;
        else if (observation.candidate == promotedMapping)
          foundSecondEvidenceDomain = true;
        else
          continue;
        observation.evidence = sharedQualityEvidence;
        sharedEvidenceObservations.push_back(&observation);
      }
    if (!foundFirstEvidenceDomain || !foundSecondEvidenceDomain)
      fail("repair quality lost its selected Mapping observation");
    auto foreignEvidenceSelection =
        loom::dse::selectJointRepairMappingByQuality(repairExecutions,
                                                     repairQuality, store);
    if (foreignEvidenceSelection)
      fail("repair quality accepted Evidence across pair domains");
    const std::string foreignEvidenceError =
        llvm::toString(foreignEvidenceSelection.takeError());
    if (!llvm::StringRef(foreignEvidenceError).contains("Evidence crossed"))
      fail("repair quality Evidence-domain rejection lost its reason");
    for (loom::dse::JointDesignQualityObservation *observation :
         sharedEvidenceObservations)
      observation->evidence.reset();
    loom::dse::CandidateMeasureObjectiveCatalogs runtimeCatalogs;
    runtimeCatalogs.dimensions = {
        {0, loom::ResolvedObjectiveDirection::Minimize, 0, 100},
        {1, loom::ResolvedObjectiveDirection::Minimize, 0, 100},
        {2, loom::ResolvedObjectiveDirection::Minimize, 0, 100}};
    runtimeCatalogs.weightedLevels = {{{{0, 1}}}, {{{1, 1}}}, {{{2, 1}}}};
    runtimeCatalogs.totalOrderings = {{{0, 1, 2}}};
    auto runtimeProgram = std::make_shared<const loom::dse::ObjectiveProgram>(
        take(loom::dse::ObjectiveProgram::getCandidateMeasures(
            runtimeCatalogs)));
    loom::dse::JointBoundedQualityPolicy runtimeRepairQuality = repairQuality;
    runtimeRepairQuality.objectiveProgram = runtimeProgram;
    runtimeRepairQuality.objectiveDimensionLabels = {
        "dfg_cycles", "cgra_cycles", "acc_core_count"};
    runtimeRepairQuality.provenanceDomain =
        loom::dse::JointDesignQualityProvenanceDomain::ApplicationRuntime;
    runtimeRepairQuality.paretoDimensions = {0, 1, 2};
    std::vector<loom::dse::JointDesignExecutionSummary> savedRuntimeSummaries;
    for (const loom::dse::JointDesignExecution &execution : repairExecutions)
      savedRuntimeSummaries.push_back(execution.summary);
    for (auto indexed : llvm::enumerate(repairExecutions)) {
      indexed.value().summary.qualityObjectiveDimensionLabels =
          runtimeRepairQuality.objectiveDimensionLabels;
      for (loom::dse::JointDesignQualityObservation &observation :
           indexed.value().summary.qualityObservations) {
        observation.provenance.rawMeasures = {
            loom::resolvedObjectiveInteger(indexed.index() + 1),
            loom::resolvedObjectiveInteger(2),
            loom::resolvedObjectiveInteger(3)};
        observation.provenance.resourceCoreCost = 3;
        loom::dse::ObjectiveVector objective = runtimeProgram->makeVector();
        if (llvm::Error error = runtimeProgram->evaluateCandidateMeasures(
                observation.provenance.rawMeasures, objective))
          fail(llvm::toString(std::move(error)));
        observation.objectiveCodes.assign(objective.codes().begin(),
                                          objective.codes().end());
      }
    }
    repairExecutions[0].summary.qualityDisposition =
        loom::dse::JointDesignQualityDisposition::Unsupported;
    repairExecutions[0].summary.qualityIncompleteCandidate = firstRepairMapping;
    repairExecutions[0].summary.selectedMapping.reset();
    repairExecutions[0].summary.selectedPlanOrdinal.reset();
    for (loom::dse::JointDesignQualityObservation &observation :
         repairExecutions[0].summary.qualityObservations)
      if (observation.candidate == firstRepairMapping) {
        observation.objectiveCodes.clear();
        observation.incompleteReason =
            loom::dse::JointDesignQualityIncompleteReason::Unsupported;
        observation.provenance.rawMeasures.clear();
      }
    repairExecutions[1]
        .summary.qualityObservations.front()
        .provenance.resourceCoreCost.reset();
    auto missingLaterResource = loom::dse::selectJointRepairMappingByQuality(
        repairExecutions, runtimeRepairQuality, store);
    if (missingLaterResource)
      fail("repair quality returned incomplete before validating a later "
           "runtime provenance");
    const std::string missingLaterResourceError =
        llvm::toString(missingLaterResource.takeError());
    if (!llvm::StringRef(missingLaterResourceError)
             .contains("lost its exact resource count"))
      fail("repair quality runtime provenance rejection lost its reason");
    for (auto indexed : llvm::enumerate(repairExecutions))
      indexed.value().summary =
          std::move(savedRuntimeSummaries[indexed.index()]);
    auto savedForeignMappedPairs = repairExecutions[1].mappedPairs;
    auto savedForeignSelectedMapping =
        repairExecutions[1].summary.selectedMapping;
    auto savedForeignQualityObservations =
        repairExecutions[1].summary.qualityObservations;
    repairExecutions[1].mappedPairs = repairExecutions[0].mappedPairs;
    repairExecutions[1].summary.selectedMapping =
        repairExecutions[0].summary.selectedMapping;
    repairExecutions[1].summary.qualityObservations =
        repairExecutions[0].summary.qualityObservations;
    auto &foreignPair = repairExecutions[1].mappedPairs.front().pair;
    foreignPair.system =
        foreignPair.system == system ? alternateSystem : system;
    auto foreignDomainSelection = loom::dse::selectJointRepairMappingByQuality(
        repairExecutions, repairQuality, store);
    if (foreignDomainSelection)
      fail("repair quality accepted one Mapping across pair domains");
    const std::string foreignDomainError =
        llvm::toString(foreignDomainSelection.takeError());
    if (!llvm::StringRef(foreignDomainError).contains("crossed pair"))
      fail("repair quality pair-domain rejection lost its reason");
    repairExecutions[1].mappedPairs = std::move(savedForeignMappedPairs);
    repairExecutions[1].summary.selectedMapping = savedForeignSelectedMapping;
    repairExecutions[1].summary.qualityObservations =
        std::move(savedForeignQualityObservations);
    repairExecutions[0].summary.qualityDisposition =
        loom::dse::JointDesignQualityDisposition::Unsupported;
    repairExecutions[0].summary.qualityIncompleteCandidate = firstRepairMapping;
    repairExecutions[0].summary.selectedMapping.reset();
    repairExecutions[0].summary.selectedPlanOrdinal.reset();
    for (loom::dse::JointDesignQualityObservation &observation :
         repairExecutions[0].summary.qualityObservations) {
      if (observation.candidate != firstRepairMapping)
        continue;
      observation.objectiveCodes.clear();
      observation.incompleteReason =
          loom::dse::JointDesignQualityIncompleteReason::Unsupported;
    }
    for (loom::dse::JointDesignQualityObservation &observation :
         repairExecutions[1].summary.qualityObservations) {
      if (observation.candidate != firstRepairMapping)
        continue;
      observation.objectiveCodes.clear();
      observation.incompleteReason =
          loom::dse::JointDesignQualityIncompleteReason::Unsupported;
    }
    const auto later = llvm::find_if(
        repairExecutions[1].summary.qualityObservations,
        [&](const loom::dse::JointDesignQualityObservation &observation) {
          return observation.candidate != firstRepairMapping &&
                 !observation.incompleteReason;
        });
    if (later == repairExecutions[1].summary.qualityObservations.end())
      fail("repair quality fixture has no later complete observation");
    auto &laterObservation = *later;
    const auto laterRawMeasures = laterObservation.provenance.rawMeasures;
    laterObservation.provenance.rawMeasures = {
        loom::resolvedObjectiveInteger(99)};
    auto invalidLaterObservation = loom::dse::selectJointRepairMappingByQuality(
        repairExecutions, repairQuality, store);
    if (invalidLaterObservation)
      fail("repair quality returned incomplete before validating all "
           "executions");
    const std::string invalidLaterObservationError =
        llvm::toString(invalidLaterObservation.takeError());
    if (!llvm::StringRef(invalidLaterObservationError)
             .contains("raw measures disagree"))
      fail("repair quality later-provenance rejection lost its reason");
    laterObservation.provenance.rawMeasures = laterRawMeasures;
    auto incompleteRepair = take(loom::dse::selectJointRepairMappingByQuality(
        repairExecutions, repairQuality, store));
    const auto *incomplete =
        std::get_if<loom::dse::JointRepairQualityIncomplete>(&incompleteRepair);
    if (!incomplete || incomplete->executionOrdinal != 0 ||
        incomplete->incomplete.reason !=
            loom::dse::JointDesignQualityIncompleteReason::Unsupported ||
        incomplete->incomplete.candidate != firstRepairMapping)
      fail("repair quality selection ranked a typed incomplete candidate");
    if (*qualityAcquisitionCount != acquisitionsBeforeRepairSelection)
      fail("repair quality incomplete selection repeated acquisition");
    repairExecutions[0].summary.qualityObservations.clear();
    auto missingIncompleteObservation =
        loom::dse::selectJointRepairMappingByQuality(repairExecutions,
                                                     repairQuality, store);
    if (missingIncompleteObservation)
      fail("repair quality accepted a missing incomplete observation");
    const std::string missingObservationError =
        llvm::toString(missingIncompleteObservation.takeError());
    if (!llvm::StringRef(missingObservationError)
             .contains("has no observation"))
      fail("repair quality missing-observation rejection lost its reason");
  }

  auto mappedDataflow = take(dataflow::importCanonicalDataflow(
      plan.frontier.pairs.front().software.dataflow, store));
  auto mappedDataflowView = take(mappedDataflow.view());
  std::vector<dataflow::RootThreadLaunchRef> mappedRoots;
  for (const auto &root : mappedDataflowView.rootThreadLaunches())
    mappedRoots.push_back(root.ref);
  if (mappedRoots.size() != 1 ||
      systemView.artifact().accCoreOccurrences().size() < 2)
    fail("adjacent resource-time repair fixture lacks one root and two cores");
  loom::dse::JointDesignExecution parentExecution{
      std::move(execution), {{plan.frontier.pairs.front(), mappings}}, {}};
  parentExecution.summary.selectedMapping = mappings.front();
  parentExecution.summary.selectedPlanOrdinal = 0;
  parentExecution.summary.verifiedAlternatives = mappings.size();
  const auto targetModules =
      take(loom::dse::projectJointDesignTargetModules(system, store));
  std::vector<loom::pnr::SystemModuleCorrespondence>
      identityModuleCorrespondence;
  for (const auto &module : targetModules)
    identityModuleCorrespondence.push_back({module, module});
  loom::dse::HardwareImpactProjection systemOnlyImpact{system, system, {},
                                                       {},     {},     {}};
  systemOnlyImpact.family = loom::dse::HardwareMutationFamily::SystemTransport;
  systemOnlyImpact.locality = loom::dse::HardwareMutationLocality::LocalCone;
  systemOnlyImpact.system.kind = loom::dse::HardwareMappingImpactKind::Reopen;
  if (!systemView.transportResources().empty())
    systemOnlyImpact.system.transportRoots.push_back(
        systemView.transportResources().front());
  const auto preservedFrontier = take(loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence,
      systemOnlyImpact, store));
  if (preservedFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::Preserved ||
      preservedFrontier.seed.techMappings.empty() ||
      preservedFrontier.seed.spatialMappings.empty() ||
      preservedFrontier.accounting.invalidatedTechMappings != 0 ||
      preservedFrontier.accounting.invalidatedSpatialMappings != 0)
    fail("System-only impact did not preserve lower Mapping layers");
  if (preservedFrontier.accounting.parentThreadBindingCount == 0 ||
      preservedFrontier.accounting.parentGraphBindingCount == 0 ||
      preservedFrontier.accounting.reopenedThreadBindingCount !=
          preservedFrontier.accounting.parentThreadBindingCount ||
      preservedFrontier.accounting.reopenedGraphBindingCount !=
          preservedFrontier.accounting.parentGraphBindingCount ||
      preservedFrontier.accounting.preservedThreadBindingCount != 0 ||
      preservedFrontier.accounting.preservedGraphBindingCount != 0)
    fail("System transport impact did not reopen its exact binding cone");

  auto rootlessTechImpact = systemOnlyImpact;
  rootlessTechImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Reopen;
  auto rootlessTech = loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence,
      rootlessTechImpact, store);
  if (rootlessTech ||
      llvm::toString(rootlessTech.takeError())
              .find("typed Tech impact has no realization root") ==
          std::string::npos)
    fail("rootless Tech reopen was not rejected");

  auto rootlessSpatialImpact = systemOnlyImpact;
  rootlessSpatialImpact.spatial.kind =
      loom::dse::HardwareMappingImpactKind::Reopen;
  auto rootlessSpatial = loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence,
      rootlessSpatialImpact, store);
  if (rootlessSpatial ||
      llvm::toString(rootlessSpatial.takeError())
              .find("typed Spatial impact has no placement or route root") ==
          std::string::npos)
    fail("rootless Spatial reopen was not rejected");

  auto targetModule =
      take(loom::fabric::importEntireFabricRoot(targetModules.front(), store));
  if (targetModule.view().fifoOccurrences().empty())
    fail("FIFO feedback fixture has no physical FIFO");
  auto feedbackParentMapping =
      take(loom::mapping::importSystemMapping(mappings.front(), store));
  std::optional<loom::ArtifactRootReference> feedbackSpatialMapping;
  std::optional<loom::fabric::FabricFifoOccurrenceRef> feedbackFifo;
  for (const auto &reference : feedbackParentMapping.view()
                                   .executionBindings()
                                   .spatialMappingImports()) {
    auto spatial = take(loom::mapping::importSpatialMapping(reference, store));
    if (spatial.view().fabricIdentity() != targetModule.view().identity())
      continue;
    for (const auto fifo : targetModule.view().fifoOccurrences())
      if (loom::mapping::spatialMappingUsesFifoOccurrence(spatial.view(),
                                                          fifo)) {
        feedbackSpatialMapping = reference;
        feedbackFifo = fifo;
        break;
      }
    if (feedbackSpatialMapping)
      break;
  }
  if (!feedbackSpatialMapping || !feedbackFifo)
    fail("FIFO feedback fixture has no selected physical FIFO");
  const auto verifyIsolatedLowerMappingClosure =
      [&](const loom::ArtifactRootReference &mapping,
          llvm::StringRef directoryName) {
        auto closure = take(loom::deployment::deriveLowerMappingPackageClosure(
            mapping, store, blobs));
        if (!llvm::is_contained(closure.artifacts(), mapping))
          fail("isolated lower-Mapping package lost its requested root");

        const std::filesystem::path staged =
            std::filesystem::path(temporary.path().str()) /
            directoryName.str();
        const std::filesystem::path stagedArtifacts = staged / "objects";
        const std::filesystem::path stagedBlobs = staged / "blobs";
        std::error_code stagingError;
        std::filesystem::create_directories(stagedArtifacts, stagingError);
        if (!stagingError)
          std::filesystem::create_directories(stagedBlobs, stagingError);
        if (stagingError)
          fail("cannot create isolated lower-Mapping closure: " +
               stagingError.message());
        for (const loom::ArtifactRootReference &root : closure.artifacts()) {
          std::filesystem::copy_file(
              std::filesystem::path(temporary.path().str()) /
                  loom::formatArtifactIdentityHex(root.artifact),
              stagedArtifacts / loom::formatArtifactIdentityHex(root.artifact),
              std::filesystem::copy_options::none, stagingError);
          if (stagingError)
            fail("cannot stage a lower-Mapping Artifact: " +
                 stagingError.message());
        }
        for (const loom::BlobDigest &digest : closure.blobs()) {
          std::filesystem::copy_file(
              std::filesystem::path(blobPath.str().str()) /
                  loom::formatBlobDigestHex(digest),
              stagedBlobs / loom::formatBlobDigestHex(digest),
              std::filesystem::copy_options::none, stagingError);
          if (stagingError)
            fail("cannot stage a lower-Mapping Blob: " +
                 stagingError.message());
        }
        loom::ArtifactStore stagedStore(stagedArtifacts.string());
        loom::BlobStore stagedBlobStore(stagedBlobs.string());
        auto stagedClosure = take(
            loom::deployment::deriveLowerMappingPackageClosure(
                mapping, stagedStore, stagedBlobStore));
        if (stagedClosure.artifacts() != closure.artifacts() ||
            stagedClosure.blobs() != closure.blobs())
          fail("isolated lower-Mapping closure changed");
        take(loom::mapping::importLowerMapping(mapping, stagedStore));
      };
  verifyIsolatedLowerMappingClosure(*feedbackSpatialMapping,
                                    "feedback-lower-mapping-closure");
  std::optional<loom::fabric::FabricPeOccurrenceRef> operandPe;
  for (const auto pe : targetModule.view().peOccurrences())
    if (targetModule.view().peSchedule(pe) == ::fabric::Schedule::Temporal) {
      operandPe = pe;
      break;
    }
  if (!operandPe)
    fail("operand-buffer feedback fixture has no Temporal PE");
  const auto operandMode = targetModule.view().peOperandBufferMode(*operandPe);
  const std::uint32_t operandEntries =
      targetModule.view().peOperandBufferSize(*operandPe);
  if (!operandMode || operandEntries == 0 ||
      operandEntries == std::numeric_limits<std::uint32_t>::max())
    fail("operand-buffer feedback fixture has no growable Temporal PE");

  if (!mutationFamily.empty()) {
    joint_fixture::exerciseJointDesignMutationFamilies(
        mutationFamily, temporary.path(), plan, parentExecution, policy,
        mappings.front(), config, system, store, blobs);
    return;
  }

  std::optional<::fabric::OperandBufferMode> separatedMode;
  if (*operandMode == ::fabric::OperandBufferMode::AllFuShare)
    separatedMode = ::fabric::OperandBufferMode::PerInputPort;
  else if (*operandMode == ::fabric::OperandBufferMode::PerInputPort)
    separatedMode = ::fabric::OperandBufferMode::PerInstruction;
  loom::ArtifactRootReference operandTech{
      loom::mapping::mappingArtifactSchema.identity.str(),
      loom::mapping::mappingArtifactSchema.version,
      take(loom::mapping::importSpatialMapping(*feedbackSpatialMapping, store))
          .view()
          .techMappingIdentity()};
  loom::dse::SpatialOperandQueueRuntimeFeedback operandFeedback;
  operandFeedback.parentMapping = mappings.front();
  operandFeedback.owners = loom::sim::CgraExecutionOwnerReferences{
      plan.frontier.pairs.front().software.dataflow, targetModules.front(),
      operandTech, *feedbackSpatialMapping};
  operandFeedback.disposition =
      loom::dse::SpatialOperandQueueRuntimeFeedbackDisposition::Exact;
  operandFeedback.reason =
      loom::dse::SpatialOperandQueueRuntimeFeedbackReason::ExactClosedWait;
  operandFeedback.repairTarget = loom::dse::SpatialOperandBufferRepairTarget{
      *operandPe, *operandMode, operandEntries, separatedMode,
      operandEntries + 1};
  if (runOperandHardwareRepair) {
    llvm::SmallString<128> operandJournal(temporary.path());
    llvm::sys::path::append(operandJournal, "operand-buffer-repair");
    const auto operandRepair =
        take(loom::dse::executeSpatialOperandBufferHardwareFeedbackReopen(
            plan, parentExecution, policy, operandFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_operand_buffer_feedback.v1")),
             operandJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::FirstVerified,
             std::nullopt,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (operandRepair.childSystems.empty() ||
        operandRepair.repairRecords.size() !=
            operandRepair.childSystems.size() ||
        operandRepair.executions.empty() ||
        operandRepair.reuseDispositions.empty())
      fail("exact operand-buffer feedback did not materialize a bounded child");
    if (separatedMode && operandRepair.childSystems.size() != 2)
      fail("exact operand-buffer feedback did not retain both bounded "
           "mode/depth alternatives");
    const std::uint64_t expectedOperandCandidateLimit = separatedMode ? 2 : 1;
    if (operandRepair.candidateLimit != expectedOperandCandidateLimit ||
        operandRepair.candidatesPlanned != operandRepair.candidatesReserved ||
        operandRepair.candidatesReserved !=
            operandRepair.candidatesConsumed +
                operandRepair.candidatesRejected +
                operandRepair.candidatesCancelled)
      fail("operand-buffer hardware child budget ledger is not closed: limit=" +
           llvm::Twine(operandRepair.candidateLimit) +
           " planned=" + llvm::Twine(operandRepair.candidatesPlanned) +
           " reserved=" + llvm::Twine(operandRepair.candidatesReserved) +
           " consumed=" + llvm::Twine(operandRepair.candidatesConsumed) +
           " rejected=" + llvm::Twine(operandRepair.candidatesRejected) +
           " cancelled=" + llvm::Twine(operandRepair.candidatesCancelled));
    bool operandMappingVerified = false;
    for (std::size_t ordinal = 0; ordinal != operandRepair.executions.size();
         ++ordinal) {
      for (const auto &pair : operandRepair.executions[ordinal].mappedPairs)
        for (const auto &mapping : pair.systemMappings) {
          auto imported =
              take(loom::mapping::importSystemMapping(mapping, store));
          if (imported.view().fabricIdentity() !=
              operandRepair.childSystems[ordinal].artifact)
            fail("operand-buffer child Mapping names the parent System");
          operandMappingVerified = true;
        }
    }
    if (!operandMappingVerified)
      fail("exact operand-buffer feedback produced no verified SystemMapping");
    auto incompleteOperandFeedback = operandFeedback;
    incompleteOperandFeedback.disposition = loom::dse::
        SpatialOperandQueueRuntimeFeedbackDisposition::ProofNotEstablished;
    const auto rejectedOperandRepair =
        take(loom::dse::executeSpatialOperandBufferHardwareFeedbackReopen(
            plan, parentExecution, policy, incompleteOperandFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_operand_buffer_feedback.negative.v1")),
             operandJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::FirstVerified,
             std::nullopt,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (!rejectedOperandRepair.childSystems.empty() ||
        !rejectedOperandRepair.executions.empty())
      fail("incomplete operand-buffer feedback synthesized a hardware child");
  }

  if (qualityRuns("operand") && separatedMode) {
    if (!incompleteRepairQuality)
      fail("bounded operand cap fixture lost its quality policy");
    auto cappedOperandQuality = *incompleteRepairQuality;
    cappedOperandQuality.maximumHardwareRepairProbes = 1;
    llvm::SmallString<128> cappedOperandJournal(temporary.path());
    llvm::sys::path::append(cappedOperandJournal,
                            "operand-buffer-repair-capped");
    const auto cappedOperandRepair =
        take(loom::dse::executeSpatialOperandBufferHardwareFeedbackReopen(
            plan, parentExecution, policy, operandFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_operand_buffer_feedback.capped.v1")),
             cappedOperandJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::BoundedQuality,
             std::move(cappedOperandQuality),
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (cappedOperandRepair.candidateLimit != 1 ||
        cappedOperandRepair.candidatesPlanned != 1 ||
        cappedOperandRepair.candidatesReserved != 1 ||
        cappedOperandRepair.candidatesConsumed +
                cappedOperandRepair.candidatesRejected +
                cappedOperandRepair.candidatesCancelled !=
            cappedOperandRepair.candidatesReserved ||
        cappedOperandRepair.childSystems.size() > 1 ||
        cappedOperandRepair.executions.size() > 1)
      fail("bounded operand hardware repair exceeded the shared probe cap");

    auto invalidOperandQuality = *incompleteRepairQuality;
    invalidOperandQuality.maximumHardwareRepairProbes = 0;
    auto invalidOperandRepair =
        loom::dse::executeSpatialOperandBufferHardwareFeedbackReopen(
            plan, parentExecution, policy, operandFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_operand_buffer_feedback.invalid.v1")),
             cappedOperandJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::BoundedQuality,
             std::move(invalidOperandQuality),
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs);
    if (invalidOperandRepair)
      fail("bounded operand repair accepted a zero probe limit");
    const std::string invalidOperandMessage =
        llvm::toString(invalidOperandRepair.takeError());
    if (!llvm::StringRef(invalidOperandMessage)
             .contains("positive probe limit"))
      fail("bounded operand zero-limit rejection lost its typed reason");
  }

  // A runtime certificate can no longer be paired with an unrelated Evidence
  // object. The only durable entry point follows the Evidence output binding
  // to its own Halted witness; a retired or otherwise foreign execution fails
  // strict import before DSE sees any certificate bytes.
  const loom::ArtifactRootReference transportWorkload =
      plan.frontier.pairs.front().software.workloads.front();
  const loom::ArtifactRootReference transportRuntimeInput =
      joint_fixture::publishApplicationRuntimeInput(transportWorkload, 7,
                                                    store);
  auto preparedTransport =
      take(loom::evaluation::models::prepareCgraSimulationEvaluation(
          plan.frontier.pairs.front().software.dataflow, targetModules.front(),
          *feedbackSpatialMapping, transportWorkload, transportRuntimeInput,
          plan.resolvedConfig, store, blobs));
  auto transportEvidence =
      take(loom::evaluation::models::evaluateCgraSimulation(
          preparedTransport, {100000, std::nullopt}, store, blobs));
  const loom::ArtifactRootReference transportEvidenceReference = take(
      loom::evaluation::publishEvaluationEvidence(transportEvidence, store));

  auto transportDataflow = take(dataflow::importCanonicalDataflow(
      plan.frontier.pairs.front().software.dataflow, store));
  auto transportDataflowView = take(transportDataflow.view());
  auto transportTech =
      take(loom::mapping::importTechMapping(operandTech, store));
  auto parentTransportConstraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          transportDataflowView, transportTech.view(), targetModule.view(),
          store));
  const auto projectedParentTransportConstraints =
      take(loom::dse::projectJointSpatialMappingConstraintSet(
          parentExecution, *feedbackSpatialMapping, store));
  if (!projectedParentTransportConstraints ||
      *projectedParentTransportConstraints !=
          parentTransportConstraints.reference())
    fail("root-complete SpatialMapping lost its empty constraint lineage");

  auto verifiedTransportEvidence =
      loom::evaluation::models::importVerifiedCgraClosedWaitEvidence(
          transportEvidenceReference, store, blobs);
  if (verifiedTransportEvidence) {
    const auto transportFeedback =
        take(loom::dse::deriveSpatialTransportRuntimeFeedback(
            *feedbackSpatialMapping, parentTransportConstraints.reference(),
            *verifiedTransportEvidence, store, mappings.front()));
    if (transportFeedback.disposition !=
            loom::dse::SpatialTransportRuntimeFeedbackDisposition::
                ProofNotEstablished ||
        transportFeedback.reason !=
            loom::dse::SpatialTransportRuntimeFeedbackReason::
                CausalCoreNotEstablished ||
        transportFeedback.constraintSet ||
        !transportFeedback.literals.empty() ||
        !transportFeedback.alternatives.empty())
      fail("partial runtime causal core published a persistent no-good");
  } else {
    llvm::consumeError(verifiedTransportEvidence.takeError());
  }

  loom::sim::CgraClosedWaitSetDiagnostic exactFifoWait;
  exactFifoWait.pendingActorFirings = 1;
  exactFifoWait.pendingTransfers = 1;
  exactFifoWait.pendingPhysicalActions = 1;
  exactFifoWait.actorFirings.push_back({0, 0, 0, 1, 0, true, false});
  loom::sim::CgraClosedWaitSetDiagnostic::Transfer blockedTransfer;
  blockedTransfer.bindingOrdinal = 0;
  blockedTransfer.occurrenceOrdinal = 0;
  blockedTransfer.producerActorOrdinal = 0;
  blockedTransfer.blocked = true;
  blockedTransfer.blockingActorOrdinal = 0;
  blockedTransfer.blockingFifoOccurrence = *feedbackFifo;
  // The claimed capacity must match the physical FIFO: the feedback path grows
  // the occurrence to capacity + 1, which must not be a no-op resize.
  const std::uint32_t feedbackFifoCapacity =
      loom::adg::builtinSmallTarget.scale.interconnectFifoDepth;
  blockedTransfer.blockingStorageOccupancy = feedbackFifoCapacity;
  blockedTransfer.blockingStorageCapacity = feedbackFifoCapacity;
  exactFifoWait.transfers.push_back(std::move(blockedTransfer));
  exactFifoWait.physicalActions.push_back(
      {0, 0, 0, 0, true, true, true, true, false});
  exactFifoWait.transferWaitCycle.push_back({0, 0, 0, 0, 0});
  const auto exactFifoFeedback =
      take(loom::dse::deriveSpatialFifoRuntimeFeedback(
          mappings.front(), *feedbackSpatialMapping, exactFifoWait, store));
  if (exactFifoFeedback.disposition !=
          loom::dse::SpatialFifoRuntimeFeedbackDisposition::Exact ||
      exactFifoFeedback.minimumCandidateDepth != feedbackFifoCapacity + 1 ||
      exactFifoFeedback.occupancy != feedbackFifoCapacity ||
      exactFifoFeedback.capacity != feedbackFifoCapacity)
    fail("exact FIFO wait did not admit the minimal hardware candidate");

  auto crossTagFifoWait = exactFifoWait;
  using ClosedWait = loom::sim::CgraClosedWaitSetDiagnostic;
  ClosedWait::WaitEdge orderWait;
  orderWait.from =
      ClosedWait::WaitOwnerKey{ClosedWait::WaitActorFiringKey{0, 0}};
  orderWait.to = ClosedWait::WaitOwnerKey{ClosedWait::WaitStorageQueueKey{
      ClosedWait::WaitStorageDomain::TraversalStorage, 0,
      ClosedWait::WaitQueueClass::global()}};
  orderWait.kind = ClosedWait::WaitEdgeKind::StorageOrder;
  orderWait.fifoOccurrence = *feedbackFifo;
  orderWait.awaitedTagValue = llvm::APInt(4, 1);
  orderWait.headTagValue = llvm::APInt(4, 2);
  ClosedWait::WaitEdge consumerWait;
  consumerWait.from = orderWait.to;
  consumerWait.to = orderWait.from;
  consumerWait.kind = ClosedWait::WaitEdgeKind::StorageConsumer;
  crossTagFifoWait.waitCertificate = {orderWait, consumerWait};
  const auto crossTagFeedback =
      take(loom::dse::deriveSpatialFifoRuntimeFeedback(
          mappings.front(), *feedbackSpatialMapping, crossTagFifoWait, store));
  if (crossTagFeedback.disposition !=
          loom::dse::SpatialFifoRuntimeFeedbackDisposition::Exact ||
      crossTagFeedback.reason != loom::dse::SpatialFifoRuntimeFeedbackReason::
                                     ExactCrossTagGlobalHolCycle ||
      crossTagFeedback.currentQueueDiscipline !=
          ::fabric::FifoQueueDiscipline::StrictFifo ||
      crossTagFeedback.candidateQueueDiscipline !=
          ::fabric::FifoQueueDiscipline::PerTagVirtualChannel ||
      crossTagFeedback.disciplineTargets !=
          std::vector<loom::fabric::FabricFifoOccurrenceRef>{*feedbackFifo} ||
      crossTagFeedback.minimumCandidateDepth)
    fail("cross-tag global HOL did not admit the VC hardware candidate");
  if (qualityRuns("fifo")) {
    if (!incompleteRepairQuality)
      fail("quality-promotion fixture lost its incomplete repair policy");
    llvm::SmallString<128> incompleteFifoJournal(temporary.path());
    llvm::sys::path::append(incompleteFifoJournal,
                            "fifo-hardware-feedback-quality-incomplete");
    const auto incompleteQualityFifo =
        take(loom::dse::executeSpatialFifoHardwareFeedbackReopen(
            plan, parentExecution, policy, exactFifoFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_fifo_feedback.quality_incomplete.v1")),
             incompleteFifoJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::BoundedQuality,
             *incompleteRepairQuality,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (incompleteQualityFifo.executions.size() != 1)
      fail("bounded incomplete FIFO repair lost its exact child execution");
    const auto &incompleteExecution = incompleteQualityFifo.executions.front();
    const bool incompleteFifoHasMapping =
        llvm::any_of(incompleteExecution.mappedPairs, [](const auto &pair) {
          return !pair.systemMappings.empty();
        });
    if (!incompleteFifoHasMapping ||
        incompleteExecution.summary.qualityDisposition !=
            loom::dse::JointDesignQualityDisposition::Unsupported ||
        incompleteExecution.summary.selectedMapping ||
        incompleteExecution.summary.selectedPlanOrdinal)
      fail("bounded incomplete FIFO repair retained a selected Mapping");
  }
  if (runFifoHardwareRepair) {
    llvm::SmallString<128> fifoJournal(temporary.path());
    llvm::sys::path::append(fifoJournal, "fifo-hardware-feedback");
    const auto fifoHardwareRepair =
        take(loom::dse::executeSpatialFifoHardwareFeedbackReopen(
            plan, parentExecution, policy, exactFifoFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_fifo_feedback.v1")),
             fifoJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::FirstVerified,
             std::nullopt,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (fifoHardwareRepair.childSystems.size() != 1 ||
        fifoHardwareRepair.repairRecords.size() != 1 ||
        fifoHardwareRepair.executions.size() != 1 ||
        fifoHardwareRepair.reuseDispositions.size() != 1 ||
        fifoHardwareRepair.childSystems.front() == system)
      fail("exact FIFO feedback did not materialize one typed System child");
    if (fifoHardwareRepair.candidateLimit != 1 ||
        fifoHardwareRepair.candidatesPlanned !=
            fifoHardwareRepair.candidatesReserved ||
        fifoHardwareRepair.candidatesReserved !=
            fifoHardwareRepair.candidatesConsumed +
                fifoHardwareRepair.candidatesRejected +
                fifoHardwareRepair.candidatesCancelled)
      fail("FIFO hardware child budget ledger is not closed");
    std::vector<loom::ArtifactRootReference> fifoChildMappings;
    for (const auto &pair : fifoHardwareRepair.executions.front().mappedPairs)
      fifoChildMappings.insert(fifoChildMappings.end(),
                               pair.systemMappings.begin(),
                               pair.systemMappings.end());
    if (fifoChildMappings.empty())
      fail("exact FIFO hardware child produced no verified SystemMapping");
    const auto &repairSummary = fifoHardwareRepair.executions.front().summary;
    if (repairSummary.parentSpatialDecisions == 0 ||
        repairSummary.repairedTechDecisions == 0 ||
        repairSummary.parentRouteNodeCount == 0 ||
        (fifoHardwareRepair.reuseDispositions.front() ==
                 loom::dse::JointMappingReuseDisposition::ColdFallback
             ? (repairSummary.repairedSpatialDecisions != 0 ||
                repairSummary.coldReopenWallTimeNanoseconds == 0 ||
                repairSummary.reopenedSpatialDecisions == 0)
             : repairSummary.repairedSpatialDecisions == 0))
      fail("FIFO hardware repair did not expose decision and route-cone "
           "accounting");
    for (const auto &reference : fifoChildMappings) {
      auto childMapping =
          take(loom::mapping::importSystemMapping(reference, store));
      if (childMapping.view().fabricIdentity() !=
          fifoHardwareRepair.childSystems.front().artifact)
        fail("FIFO hardware repair Mapping names the parent System");
    }

    llvm::SmallString<128> disciplineJournal(temporary.path());
    llvm::sys::path::append(disciplineJournal,
                            "fifo-discipline-recipe-feedback");
    const auto disciplineRepair =
        take(loom::dse::executeSpatialFifoHardwareFeedbackReopen(
            plan, parentExecution, policy, crossTagFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_fifo_recipe_feedback.v1")),
             disciplineJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::FirstVerified,
             std::nullopt,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (disciplineRepair.childSystems.size() != 1 ||
        disciplineRepair.repairRecords.size() != 1 ||
        disciplineRepair.executions.size() != 1 ||
        disciplineRepair.reuseDispositions !=
            std::vector<loom::dse::JointMappingReuseDisposition>{
                loom::dse::JointMappingReuseDisposition::ColdFallback})
      fail("global FIFO recipe feedback lost its durable cold repair");
    const auto disciplineRecord =
        take(loom::dse::importHardwareMutationRepairRecord(
            disciplineRepair.repairRecords.front(), store));
    if (disciplineRecord.record().parentSystem != system ||
        disciplineRecord.record().childSystem !=
            disciplineRepair.childSystems.front() ||
        disciplineRecord.record().impacts.size() != 1 ||
        disciplineRecord.record().impacts.front().family !=
            loom::dse::HardwareMutationFamily::SpatialFifo ||
        disciplineRecord.record().impacts.front().locality !=
            loom::dse::HardwareMutationLocality::GlobalReopen ||
        disciplineRecord.record().incremental.mappings.empty())
      fail("global FIFO recipe repair record lost its typed execution");
  }
  auto incompleteFifoWait = exactFifoWait;
  incompleteFifoWait.transferWaitCycle.clear();
  const auto incompleteFifoFeedback =
      take(loom::dse::deriveSpatialFifoRuntimeFeedback(
          mappings.front(), *feedbackSpatialMapping, incompleteFifoWait,
          store));
  if (incompleteFifoFeedback.disposition !=
          loom::dse::SpatialFifoRuntimeFeedbackDisposition::
              ProofNotEstablished ||
      incompleteFifoFeedback.reason !=
          loom::dse::SpatialFifoRuntimeFeedbackReason::MissingWaitCycle ||
      incompleteFifoFeedback.minimumCandidateDepth)
    fail("probe-incomplete FIFO wait synthesized a hardware child");
  if (runRuntimeWitnessBudget) {
    // The two runtime-witness repair families never share a budget: a fixed
    // System frontier or an exhausted shared probe ledger withholds the
    // hardware child, while an admitted FIFO witness with no transport
    // witness still materializes exactly one child and reserves one parent
    // cost for it.
    constexpr std::uint64_t parentCostNanoseconds = 7;
    const auto witnessRequest =
        [&](llvm::StringRef journal,
            loom::dse::JointHardwareExplorationScope scope) {
          llvm::SmallString<128> root(temporary.path());
          llvm::sys::path::append(root, journal);
          loom::dse::JointHardwareReopenRequest request{
              take(loom::dse::DseProducerSemanticBuildIdentity::get(
                  "loom.test.runtime_witness_budget.v1")),
              root.str().str(),
              {},
              loom::dse::JointDesignStoppingPolicy::FirstVerified,
              std::nullopt,
              std::nullopt,
              take(loom::dse::SiteCapacity::get(2, 0, 0)),
              take(loom::dse::PlanExecutionPolicy::get(
                  2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))};
          request.hardwareExplorationScope = scope;
          return request;
        };
    loom::dse::JointRuntimeWitnessSet witnesses;
    witnesses.fifo = exactFifoFeedback;
    const auto withheld =
        [&](const loom::dse::JointRuntimeWitnessRepair &repair,
            llvm::StringRef gate) {
          if (repair.fifoReopen || repair.mappingRepair ||
              !repair.childSystems.empty() ||
              !repair.hardwareMutationRepairRecords.empty() ||
              !repair.executions.empty() ||
              repair.hardwareReopenReservedNanoseconds != 0 ||
              repair.hardwareReopenLedger.reserved != 0 ||
              repair.mappingRepairLedger.candidateLimit != 0)
            fail(gate + " admitted a runtime-witness hardware child");
        };
    withheld(
        take(loom::dse::executeJointRuntimeWitnessRepair(
            plan, parentExecution, policy, witnesses, parentCostNanoseconds,
            std::nullopt,
            witnessRequest(
                "runtime-witness-fixed",
                loom::dse::JointHardwareExplorationScope::FixedSystemFrontier),
            store, blobs)),
        "fixed System frontier");
    withheld(take(loom::dse::executeJointRuntimeWitnessRepair(
                 plan, parentExecution, policy, witnesses,
                 parentCostNanoseconds, std::optional<std::uint64_t>(0),
                 witnessRequest("runtime-witness-exhausted",
                                loom::dse::JointHardwareExplorationScope::
                                    BoundedHardwareReopen),
                 store, blobs)),
             "exhausted shared probe ledger");
    const auto admitted = take(loom::dse::executeJointRuntimeWitnessRepair(
        plan, parentExecution, policy, witnesses, parentCostNanoseconds,
        std::nullopt,
        witnessRequest(
            "runtime-witness-admitted",
            loom::dse::JointHardwareExplorationScope::BoundedHardwareReopen),
        store, blobs));
    const loom::dse::JointRepairWorkLedger &hardware =
        admitted.hardwareReopenLedger;
    if (!admitted.fifoReopen || admitted.mappingRepair ||
        admitted.childSystems.size() != 1 || admitted.executions.size() != 1 ||
        admitted.hardwareMutationRepairRecords.size() != 1 ||
        !admitted.hardwareMutationRepairRecords.front() ||
        admitted.childSystems.front() == system ||
        admitted.hardwareReopenReservedNanoseconds != parentCostNanoseconds ||
        hardware.candidateLimit != 1 || hardware.planned != 1 ||
        hardware.reserved != 1 ||
        hardware.reserved !=
            hardware.consumed + hardware.rejected + hardware.cancelled ||
        admitted.mappingRepairLedger.candidateLimit != 0 ||
        admitted.mappingRepairLedger.reserved != 0)
      fail("admitted FIFO witness did not materialize exactly one budgeted "
           "hardware child");
    bool childMappingVerified = false;
    for (const auto &pair : admitted.executions.front().mappedPairs)
      for (const auto &mapping : pair.systemMappings) {
        auto imported =
            take(loom::mapping::importSystemMapping(mapping, store));
        if (imported.view().fabricIdentity() !=
            admitted.childSystems.front().artifact)
          fail("runtime-witness child Mapping names the parent System");
        childMappingVerified = true;
      }
    if (!childMappingVerified)
      fail("admitted FIFO witness produced no verified child SystemMapping");

    const auto unboundedPolicy = take(loom::dse::PlanExecutionPolicy::get(
        2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))));
    if (take(loom::dse::reserveDispatchWindow(unboundedPolicy, 1000))
            .dispatchNotAfterUnixNanoseconds())
      fail("reserving a window on an unbounded policy invented a deadline");
    const std::uint64_t nowNanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    const std::uint64_t farDeadline = nowNanoseconds + 3'600'000'000'000ULL;
    const auto boundedPolicy = take(loom::dse::PlanExecutionPolicy::get(
        2, take(loom::dse::SiteResourceClaim::get(1, 0, 0)), std::nullopt, {},
        std::nullopt, farDeadline));
    if (take(loom::dse::reserveDispatchWindow(boundedPolicy, 1000))
            .dispatchNotAfterUnixNanoseconds() != farDeadline - 1000)
      fail("window reservation did not move the deadline earlier by the "
           "reserved amount");
    const auto clamped = take(loom::dse::reserveDispatchWindow(
        boundedPolicy, farDeadline + farDeadline));
    if (!clamped.dispatchNotAfterUnixNanoseconds() ||
        *clamped.dispatchNotAfterUnixNanoseconds() < nowNanoseconds ||
        *clamped.dispatchNotAfterUnixNanoseconds() >= farDeadline)
      fail("over-reserved window did not clamp its deadline to the present");
  }
  if (runFifoHardwareRepair || runRuntimeWitnessBudget)
    return;
  const auto moduleRoot =
      take(loom::fabric::FabricModulePhysicalOwnerRef::create(*feedbackFifo));
  loom::dse::HardwareImpactProjection localSpatialImpact{
      targetModules.front(), system, {}, {}, {}, {}};
  localSpatialImpact.family = loom::dse::HardwareMutationFamily::SpatialFifo;
  localSpatialImpact.locality = loom::dse::HardwareMutationLocality::LocalCone;
  localSpatialImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Rebase;
  localSpatialImpact.spatial.kind =
      loom::dse::HardwareMappingImpactKind::Reopen;
  localSpatialImpact.spatial.placementRoots.push_back(moduleRoot);
  localSpatialImpact.moduleEntities =
      joint_fixture::identityModuleEntityCorrespondence(targetModule.view());
  const auto localRepairFrontier = take(loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence,
      localSpatialImpact, store));
  if (localRepairFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::Preserved ||
      localRepairFrontier.seed.techMappings.empty() ||
      localRepairFrontier.seed.spatialMappings.empty() ||
      localRepairFrontier.accounting.invalidatedSpatialMappings != 0 ||
      localRepairFrontier.accounting.repairedSpatialMappings == 0)
    fail("typed local Spatial impact did not preserve and revalidate its "
         "selected cone");

  auto impactSpatial =
      take(loom::mapping::importSpatialMapping(*feedbackSpatialMapping, store));
  std::optional<loom::fabric::FabricFifoOccurrenceRef> unusedFifo;
  for (const auto fifo : targetModule.view().fifoOccurrences())
    if (!loom::mapping::spatialMappingUsesFifoOccurrence(impactSpatial.view(),
                                                         fifo)) {
      unusedFifo = fifo;
      break;
    }
  if (!unusedFifo)
    fail("mapping-reuse fixture has no unused FIFO for a zero-cone witness");
  loom::dse::HardwareImpactProjection unusedImpact{
      targetModules.front(), system, {}, {}, {}, {}};
  unusedImpact.family = loom::dse::HardwareMutationFamily::SpatialFifo;
  unusedImpact.locality = loom::dse::HardwareMutationLocality::LocalCone;
  unusedImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Rebase;
  unusedImpact.spatial.kind = loom::dse::HardwareMappingImpactKind::Reopen;
  unusedImpact.spatial.placementRoots.push_back(
      take(loom::fabric::FabricModulePhysicalOwnerRef::create(*unusedFifo)));
  unusedImpact.moduleEntities =
      joint_fixture::identityModuleEntityCorrespondence(targetModule.view());
  const auto unusedFrontier = take(loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence, unusedImpact,
      store));
  if (unusedFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::Preserved ||
      unusedFrontier.accounting.invalidatedSpatialMappings != 0 ||
      unusedFrontier.accounting.repairedSpatialMappings != 0 ||
      unusedFrontier.seed.spatialMappings.empty())
    fail("unused FIFO impact did not produce a zero-cone Spatial preserve");

  auto globalImpact = localSpatialImpact;
  globalImpact.family = loom::dse::HardwareMutationFamily::FuCapability;
  globalImpact.locality = loom::dse::HardwareMutationLocality::GlobalReopen;
  globalImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Reopen;
  globalImpact.tech.realizationRoots.push_back(moduleRoot);
  const auto coldFallbackFrontier = take(loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence, globalImpact,
      store));
  if (coldFallbackFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::ColdFallback ||
      !coldFallbackFrontier.seed.techMappings.empty() ||
      !coldFallbackFrontier.seed.spatialMappings.empty() ||
      coldFallbackFrontier.accounting.parentTechMappings == 0 ||
      coldFallbackFrontier.accounting.parentSpatialMappings == 0 ||
      coldFallbackFrontier.accounting.invalidatedTechMappings !=
          coldFallbackFrontier.accounting.parentTechMappings ||
      coldFallbackFrontier.accounting.invalidatedSpatialMappings !=
          coldFallbackFrontier.accounting.parentSpatialMappings ||
      coldFallbackFrontier.accounting.invalidationRootCount == 0 ||
      coldFallbackFrontier.accounting.invalidationConeDecisionCount == 0)
    fail("typed global impact did not preserve a cold fallback");

  const auto requireLocalModuleRebase =
      [&](loom::dse::HardwareMutationFamily family,
          loom::fabric::FabricModulePhysicalOwnerRef owner) {
        auto impact = localSpatialImpact;
        impact.family = family;
        impact.tech.kind = loom::dse::HardwareMappingImpactKind::Rebase;
        impact.spatial.kind = loom::dse::HardwareMappingImpactKind::Rebase;
        impact.spatial.placementRoots = {owner};
        const auto result = take(loom::dse::rebaseJointMappingFrontier(
            plan, parentExecution, system, identityModuleCorrespondence, impact,
            store));
        if (result.disposition !=
                loom::dse::JointMappingReuseDisposition::Preserved ||
            result.seed.techMappings.empty() ||
            result.seed.spatialMappings.empty() ||
            result.accounting.invalidatedTechMappings != 0 ||
            result.accounting.invalidatedSpatialMappings != 0)
          fail("local Module mutation did not preserve its typed Mapping "
               "frontier");
      };
  if (!targetModule.view().memoryOccurrences().empty())
    requireLocalModuleRebase(
        loom::dse::HardwareMutationFamily::SpatialMemory,
        take(loom::fabric::FabricModulePhysicalOwnerRef::create(
            targetModule.view().memoryOccurrences().front())));
  if (!targetModule.view().peOccurrences().empty())
    requireLocalModuleRebase(
        loom::dse::HardwareMutationFamily::InstructionCapacity,
        take(loom::fabric::FabricModulePhysicalOwnerRef::create(
            targetModule.view().peOccurrences().front())));
  if (!targetModule.view().switchOccurrences().empty()) {
    auto switchImpact = localSpatialImpact;
    switchImpact.family = loom::dse::HardwareMutationFamily::SpatialSwitch;
    switchImpact.locality = loom::dse::HardwareMutationLocality::GlobalReopen;
    switchImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Reopen;
    switchImpact.spatial.kind = loom::dse::HardwareMappingImpactKind::Reopen;
    switchImpact.tech.realizationRoots = {
        take(loom::fabric::FabricModulePhysicalOwnerRef::create(
            targetModule.view().switchOccurrences().front()))};
    switchImpact.spatial.placementRoots = switchImpact.tech.realizationRoots;
    const auto switchFallback = take(loom::dse::rebaseJointMappingFrontier(
        plan, parentExecution, system, identityModuleCorrespondence,
        switchImpact, store));
    if (switchFallback.disposition !=
            loom::dse::JointMappingReuseDisposition::ColdFallback ||
        !switchFallback.seed.techMappings.empty() ||
        !switchFallback.seed.spatialMappings.empty())
      fail("global switch mutation did not produce a typed cold fallback");
  }
  joint_fixture::exerciseAdjacentResourceTimeMappingRepair(
      temporary.path(), plan, parentExecution, policy, mappedRoots.front(),
      system, systemView, alternateSystem, mappings.front(),
      qualityRuns("adjacent"),
      incompleteRepairQuality ? &*incompleteRepairQuality : nullptr, store,
      blobs);
  const std::vector<loom::ArtifactRootReference> systems = {system,
                                                            alternateSystem};
  const std::vector<loom::dse::JointMemberPromotion> memberPromotions = {
      {plan.frontier.pairs.front().software.dataflow,
       loom::dse::CompletedSelection{mappings, {}}}};
  auto selected = take(loom::dse::selectJointDesignSystems(
      systems, memberPromotions, {}, loom::dse::AllPassingSelection{}, nullptr,
      store));
  const bool covered = joint_fixture::everyCoreIsUsed(system, mappings, store);
  bool sawMissingAlternate = false;
  bool sawUnusedPrimary = false;
  std::vector<loom::dse::JointSystemGateOutcome> *outcomes = nullptr;
  if (auto *completedSelection =
          std::get_if<loom::dse::JointDesignSelection>(&selected)) {
    outcomes = &completedSelection->systemOutcomes;
    if (!covered ||
        completedSelection->selectedSystems !=
            std::vector<loom::ArtifactRootReference>{system} ||
        completedSelection->acceptedMappings != mappings)
      fail("aggregate selection bypassed member-local System gates");
  } else {
    auto &noFeasible =
        std::get<loom::dse::JointDesignNoFeasibleSystem>(selected);
    outcomes = &noFeasible.systemOutcomes;
    if (covered)
      fail("fully covered System was rejected before aggregate selection");
  }
  for (const loom::dse::JointSystemGateOutcome &outcome : *outcomes) {
    if (const auto *missing =
            std::get_if<loom::dse::JointSystemMissingMember>(&outcome))
      sawMissingAlternate |= missing->system == alternateSystem;
    if (const auto *unused =
            std::get_if<loom::dse::JointSystemUnusedAccCore>(&outcome))
      sawUnusedPrimary |= unused->system == system;
  }
  if (!sawMissingAlternate || sawUnusedPrimary == covered)
    fail("typed System dispositions lost missing-member or AccCore coverage");

  auto oversized = loom::dse::buildBoundedJointFrontier(
      {{{firstWorkload}, {secondWorkload}}, {system}},
      take(loom::dse::JointDesignPolicy::get(1, 1, 1, 1, 1)), store);
  if (oversized)
    fail("joint frontier accepted a software set beyond its resolved bound");
  const std::string oversizedMessage = llvm::toString(oversized.takeError());
  if (!llvm::StringRef(oversizedMessage).contains("exceeds"))
    fail("frontier-bound rejection lost its diagnostic");
}

} // namespace

int main(int argc, char **argv) {
  const llvm::StringRef mode = argc == 2 ? argv[1] : "";
  // `mutation-matrix` and `quality-promotion` run every section in one
  // process; the `=NAME` forms run one independent section so the sections can
  // be sharded across processes.
  llvm::StringRef mutationFamily;
  if (mode == "mutation-matrix")
    mutationFamily = joint_fixture::allJointDesignTestSections;
  else if (mode.starts_with("mutation-matrix="))
    mutationFamily =
        mode.drop_front(llvm::StringRef("mutation-matrix=").size());
  llvm::StringRef qualitySection;
  if (mode == "quality-promotion")
    qualitySection = joint_fixture::allJointDesignTestSections;
  else if (mode.starts_with("quality-promotion="))
    qualitySection =
        mode.drop_front(llvm::StringRef("quality-promotion=").size());
  if (!qualitySection.empty() &&
      qualitySection != joint_fixture::allJointDesignTestSections &&
      qualitySection != "promotion" && qualitySection != "operand" &&
      qualitySection != "fifo" && qualitySection != "adjacent")
    fail("unknown quality-promotion section: " + qualitySection);
  if (argc > 2 ||
      (argc == 2 && mutationFamily.empty() && qualitySection.empty() &&
       mode != "fifo-feedback" && mode != "runtime-witness-budget" &&
       mode != "operand-feedback" && mode != "transport-feedback"))
    fail("expected no workflow, fifo-feedback, runtime-witness-budget, "
         "operand-feedback, transport-feedback, quality-promotion, "
         "quality-promotion=SECTION, mutation-matrix, or "
         "mutation-matrix=FAMILY");
  exerciseJointExploration(
      mode == "fifo-feedback", mode == "runtime-witness-budget",
      mode == "operand-feedback", mode == "transport-feedback", qualitySection,
      mutationFamily);
  return 0;
}
