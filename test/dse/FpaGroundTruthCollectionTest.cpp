#include "DSE/CampaignRunner.h"
#include "DSE/CandidateGeneratorRecovery.h"
#include "DSE/GroundTruthPlan.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "EDA/Adapters/OpenSource/OpenRoadRouted.h"
#include "EDA/Adapters/OpenSource/OpenRoadStaticFpa.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "OpenRoadPhysicalTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Request.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/Support/Error.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;
using namespace loom::eda::open_source;
using namespace loom::eda::open_source::test;
using namespace loom::evaluation;
using namespace loom::evaluation::models;
using namespace loom::external_tool;
using namespace loom::hardware;

constexpr llvm::StringLiteral kProviderBuild =
    "OpenROAD synthetic 21512b0ab68c";

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "DSE ground-truth campaign test failure: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

template <typename T>
void requireErrorContains(llvm::Expected<T> value, llvm::StringRef needle) {
  if (value)
    fail("expected rejection containing '" + needle.str() + "'");
  const std::string message = llvm::toString(value.takeError());
  if (!llvm::StringRef(message).contains(needle))
    fail("unexpected rejection: " + message);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error;
    const std::filesystem::path base = std::filesystem::current_path() / "temp";
    std::filesystem::create_directories(base, error);
    const auto identity =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = base / ("loom-ground-truth-" + std::to_string(identity));
    if (error || !std::filesystem::create_directories(path_, error) || error)
      fail("cannot create temporary directory");
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

struct RoutedFixture final {
  OpenRoadGateFixture gate;
  FinalizedHardwareImplementation routed;
};

RoutedFixture makeRoutedFixture(const std::filesystem::path &root,
                                llvm::StringRef designIdentity,
                                std::uint32_t designPortBitWidth,
                                const ArtifactStore &artifacts,
                                const BlobStore &blobs) {
  OpenRoadGateFixture gate =
      take(makeOpenRoadGateFixture(root, artifacts, blobs, kProviderBuild,
                                   syntheticOpenRoadTechnologyFixture(),
                                   designIdentity, designPortBitWidth));
  const std::filesystem::path tool = take(writeAuthoredOpenRoadRouteTool(root));
  const LocalToolConfig local = makeOpenRoadLocalToolConfig(gate, tool);
  OpenRoadRouteHarness harness =
      take(makeOpenRoadRouteHarness(root / "route-bundle", gate, local));
  FinalizedHardwareImplementation routed = take(runOpenRoadRouteFixture(
      gate, harness,
      makeOpenRoadResolvedExecution(tool.string(), kProviderBuild, false),
      artifacts, blobs));
  return RoutedFixture{std::move(gate), std::move(routed)};
}

std::filesystem::path
writeDelayedFpaTool(const std::filesystem::path &root,
                    const std::filesystem::path &underlying) {
  const std::filesystem::path tool = root / "delayed-openroad-fpa";
  std::ofstream output(tool);
  output << "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "tool_dir=$(cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
            "if [[ \"${1:-}\" == \"-version\" || "
            "\"${1:-}\" == \"--version\" ]]; then\n"
            "  exec \"$tool_dir/"
         << underlying.filename().string()
         << "\" \"$@\"\n"
            "fi\n"
            "sleep 5\n"
            "exec \"$tool_dir/"
         << underlying.filename().string() << "\" \"$@\"\n";
  output.close();
  if (!output)
    fail("cannot write delayed FPA tool");
  std::error_code error;
  std::filesystem::permissions(tool,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec,
                               std::filesystem::perm_options::replace, error);
  if (error)
    fail("cannot make delayed FPA tool executable");
  return tool;
}

SubjectTargetRef rootTarget(const ArtifactRootReference &hardware) {
  return {hardwareImplementationPhysicalSubjectRole(), hardware,
          SubjectTarget{hardware}};
}

loom::fabric::FabricModuleDomainMemberRef
transportCharacterizationLeaf(const RoutedFixture &fixture,
                              const ArtifactStore &artifacts) {
  auto fabricRoot = take(loom::fabric::importEntireFabricRoot(
      fixture.routed.implementation().fabric(), artifacts));
  auto system = take(loom::fabric::requireSystemRoot(fabricRoot.view()));
  const auto selected =
      system.spatialCoreTarget(fixture.routed.implementation().subject().core);
  if (!selected ||
      selected->dependencyOrdinal >= system.artifact().importedModules().size())
    fail("routed fixture subject has no imported SpatialCore Module");
  const loom::fabric::FabricArtifactView &module =
      system.artifact().importedModules()[selected->dependencyOrdinal];
  const auto moduleTemplate = module.moduleRootTemplate();
  if (!moduleTemplate ||
      module.moduleBoundaryEndpointCount(
          *moduleTemplate, loom::fabric::FabricPortDirection::Input) == 0)
    fail("routed fixture Module has no transport input boundary");
  const loom::fabric::FabricModuleBoundaryEndpointRef boundary{
      *moduleTemplate, loom::fabric::FabricPortDirection::Input, 0};
  if (module.moduleBoundaryEndpointPlane(boundary) !=
      loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport)
    fail("routed fixture input boundary is not transport");
  return loom::fabric::FabricModuleDomainMemberRef::of(boundary);
}

std::vector<EvaluationCondition>
operatingConditions(const RoutedFixture &prototype) {
  const ArtifactRootReference &hardware = prototype.routed.reference();
  const SubjectTargetRef target = rootTarget(hardware);
  return {
      EvaluationCondition{
          ProcessCornerCondition{target,
                                 {prototype.gate.platform.reference().artifact,
                                  platform::TechnologyCornerId(0)}}},
      EvaluationCondition{
          SupplyVoltageCondition{target, take(DecimalValue::get(1050, -3))}},
      EvaluationCondition{
          TemperatureCondition{target, take(DecimalValue::get(3, 2))}},
      EvaluationCondition{
          RequiredClockPeriodCondition{target, take(DecimalValue::get(2, -9))}},
      EvaluationCondition{ActivityBindingCondition{
          target, ExplicitAssumptionSource{target, take(ExactRatio::get(1, 2)),
                                           take(ExactRatio::get(1, 10))}}}};
}

ResolvedConfig campaignConfig() {
  ResolvedConfig config = defaultResolvedConfig();
  config.evaluation.openRoadStaticFpa =
      OpenRoadStaticFpaProviderBinding{kProviderBuild.str()};
  return config;
}

PlanExecutionPolicy executionPolicy(const LocalToolConfig &local,
                                    ExternalAttemptDisposition disposition,
                                    std::optional<std::uint64_t> dispatches) {
  ExternalExecutionSite site{local, disposition, 1, 0, 0, false};
  return take(PlanExecutionPolicy::get(2, take(SiteResourceClaim::get(1, 0, 0)),
                                       std::move(site), {}, dispatches));
}

SiteScheduler scheduler(llvm::ArrayRef<BlobDigest> bindings) {
  std::vector<CountedSiteResource> tools;
  for (const BlobDigest &binding : bindings)
    tools.push_back({SiteResourceKey::externalToolBinding(binding), 2});
  llvm::sort(tools, [](const auto &lhs, const auto &rhs) {
    return lhs.key < rhs.key;
  });
  return take(
      SiteScheduler::create(take(SiteCapacity::get(2, 0, 0, tools, {}))));
}

std::vector<BlobDigest> preparedBindings(const ExecutionJournal &journal) {
  std::vector<BlobDigest> bindings;
  for (const JournalWorkUnitRecord &record : take(journal.workUnits())) {
    if (!record.preparedInvocation)
      continue;
    bindings.push_back(take(
        deriveExternalToolExecutionBindingDigest(*record.preparedInvocation)));
  }
  llvm::sort(bindings, [](const BlobDigest &lhs, const BlobDigest &rhs) {
    return lhs.bytes() < rhs.bytes();
  });
  bindings.erase(std::unique(bindings.begin(), bindings.end()), bindings.end());
  return bindings;
}

void verifyEvidence(const ArtifactRootReference &evidenceReference,
                    const ArtifactRootReference &candidate,
                    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto externalContracts =
      take(loom::eda::makeKnownAsicStandardCellContractCatalog());
  CaseArtifactResolution resolution =
      take(resolveHardwareImplementationPhysicalCase(
          candidate, externalContracts, artifacts, blobs));
  ArtifactRootReference requestReference = take(
      importEvaluationEvidenceRequestReference(evidenceReference, artifacts));
  EvaluationRequest request = take(
      importEvaluationRequest(requestReference, resolution, artifacts, blobs));
  EvaluationEvidence evidence = take(importEvaluationEvidence(
      evidenceReference, resolution, artifacts, blobs));
  const auto subjects = request.subjectBindings().subjects(
      hardwareImplementationPhysicalSubjectRole());
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  if (subjects.size() != 1 || subjects.front() != candidate || !completed ||
      completed->metricResults.size() != 4)
    fail("imported campaign Evidence lost its candidate or FPA results");
  for (const EvaluationCondition &condition : request.baseConditions())
    for (const SubjectTargetRef *target : conditionOrderedTargets(condition)) {
      const auto *root = std::get_if<ArtifactRootReference>(&target->target);
      if (target->anchorSubjectArtifact != candidate || !root ||
          *root != candidate)
        fail("dynamic obligation target was not rebound to the candidate");
    }
}

void verifyCalibrationEvidence(const ArtifactRootReference &calibrationEvidence,
                               const ArtifactRootReference &parameterBundle,
                               const ArtifactRootReference &sourceEvidence,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs) {
  const std::array sources = {sourceEvidence};
  CaseArtifactResolution resolution =
      take(resolveFpaCalibrationCaseArtifactResolution(parameterBundle, sources,
                                                       artifacts, blobs));
  EvaluationEvidence evidence = take(importEvaluationEvidence(
      calibrationEvidence, resolution, artifacts, blobs));
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  if (!completed || completed->metricResults.size() != 8)
    fail("calibration Evidence did not contain median and P90 FPA errors");
}

void exerciseGroundTruthCampaign() {
  TemporaryDirectory temporary;
  const std::filesystem::path artifactsPath = temporary.path() / "artifacts";
  const std::filesystem::path blobsPath = temporary.path() / "blobs";
  const std::filesystem::path runPath = temporary.path() / "run";
  std::filesystem::create_directories(artifactsPath);
  std::filesystem::create_directories(blobsPath);
  std::filesystem::create_directories(runPath);
  ArtifactStore artifacts(artifactsPath.string());
  BlobStore blobs(blobsPath.string());
  requireSuccess(registerOpenRoadRoutedCandidateGeneratorDescriptor());
  requireSuccess(registerOpenRoadStaticFpaEvaluationProvider());

  RoutedFixture training =
      makeRoutedFixture(temporary.path() / "training", "ground-truth-training",
                        8, artifacts, blobs);
  RoutedFixture validation =
      makeRoutedFixture(temporary.path() / "validation",
                        "ground-truth-validation", 16, artifacts, blobs);
  RoutedFixture heldOut =
      makeRoutedFixture(temporary.path() / "held-out", "ground-truth-held-out",
                        32, artifacts, blobs);
  const std::vector<EvaluationCondition> conditions =
      operatingConditions(training);
  const ResolvedConfig base = campaignConfig();

  requireErrorContains(
      buildFpaGroundTruthCollectionPlan({{{training.routed.reference()}},
                                         {{training.routed.reference()}},
                                         {{heldOut.routed.reference()}},
                                         conditions},
                                        base, artifacts, blobs),
      "multiple partitions");
  requireErrorContains(
      buildFpaGroundTruthCollectionPlan({{{training.routed.reference()}},
                                         {{validation.routed.reference()}},
                                         {{}},
                                         conditions},
                                        base, artifacts, blobs),
      "HeldOut partition is empty");

  FpaGroundTruthCollectionPlan plan =
      take(buildFpaGroundTruthCollectionPlan({{{training.routed.reference()}},
                                              {{validation.routed.reference()}},
                                              {{heldOut.routed.reference()}},
                                              conditions},
                                             base, artifacts, blobs));
  ResolvedDseConfigView view =
      take(projectResolvedDseConfigView(plan.resolvedConfig));
  if (view.plan().nodes().size() != 3 ||
      view.plan().resolve(plan.trainingEvidence)->calibrationPartitionRole ||
      view.plan().resolve(plan.validationEvidence)->calibrationPartitionRole ||
      view.plan().resolve(plan.heldOutEvidence)->calibrationPartitionRole)
    fail("ground-truth collection leaked a calibration partition role");

  const ArtifactIdentity storedConfig =
      take(artifacts.put(ResolvedConfig::artifactSchema,
                         canonicalResolvedConfigBytes(plan.resolvedConfig)));
  if (storedConfig != resolvedConfigIdentity(plan.resolvedConfig))
    fail("campaign ResolvedConfig publication changed identity");
  DseProducerSemanticBuildIdentity producer =
      take(DseProducerSemanticBuildIdentity::get("loom.test.ground_truth.v1"));
  const std::array<ArtifactRootReference, 3> semanticInputs = {
      training.routed.reference(), validation.routed.reference(),
      heldOut.routed.reference()};
  DseRunClosure closure = take(DseRunClosure::get(
      std::move(producer), semanticInputs, plan.resolvedConfig, {}, artifacts));
  ExecutionJournal journal =
      take(openExecutionJournal(runPath.string(), closure, view));

  CampaignExecutionPolicy campaignPolicy =
      take(makeFpaGroundTruthCampaignPolicy(1, 1));

  SiteScheduler unavailableScheduler = scheduler({});
  PlanExecutionPolicy unavailableExecution =
      take(PlanExecutionPolicy::get(1, take(SiteResourceClaim::get(1, 0, 0))));
  CampaignExecutionResult unavailable = take(runFpaGroundTruthCampaign(
      view, closure, campaignPolicy, unavailableExecution, unavailableScheduler,
      journal, artifacts, blobs));
  const auto *unavailableRefusal =
      std::get_if<CampaignAdmissionRefusal>(&unavailable);
  const auto *unavailablePlan = unavailableRefusal
                                    ? std::get_if<IncompleteDsePlanExecution>(
                                          &unavailableRefusal->outcome)
                                    : nullptr;
  const auto *unavailableReason =
      unavailablePlan ? std::get_if<PromotionAcquisitionIncompleteReason>(
                            &unavailablePlan->reason())
                      : nullptr;
  if (!unavailableRefusal ||
      unavailableRefusal->reason !=
          CampaignAdmissionFailureReason::InsufficientPilotObservations ||
      !unavailableReason ||
      *unavailableReason !=
          PromotionAcquisitionIncompleteReason::ProviderUnavailable)
    fail("missing OpenROAD execution site was not typed unavailable");
  requireSuccess(journal.releaseInvocationOccurrence());

  const std::filesystem::path fpaTool =
      take(writeAuthoredOpenRoadStaticFpaTool(temporary.path()));
  const LocalToolConfig local =
      makeOpenRoadLocalToolConfig(training.gate, fpaTool);

  const std::filesystem::path delayedTool =
      writeDelayedFpaTool(temporary.path(), fpaTool);
  const LocalToolConfig delayedLocal =
      makeOpenRoadLocalToolConfig(training.gate, delayedTool);
  const std::filesystem::path deadlineRunPath =
      temporary.path() / "deadline-run";
  std::filesystem::create_directories(deadlineRunPath);
  ExecutionJournal deadlineJournal =
      take(openExecutionJournal(deadlineRunPath.string(), closure, view));
  SiteScheduler deadlinePreparationScheduler = scheduler({});
  DsePlanExecutionOutcome deadlinePrepared = take(resumeDsePlan(
      view, closure, deadlineJournal, deadlinePreparationScheduler,
      executionPolicy(delayedLocal, ExternalAttemptDisposition::PrepareOnly, 1),
      artifacts, blobs, InvocationManifestRetention::Release));
  if (!std::holds_alternative<IncompleteDsePlanExecution>(deadlinePrepared))
    fail("deadline fixture unexpectedly completed during preparation");
  const std::vector<BlobDigest> deadlineBindings =
      preparedBindings(deadlineJournal);
  if (deadlineBindings.size() != 1)
    fail("deadline fixture did not prepare one exact external binding");
  SiteScheduler deadlineScheduler = scheduler(deadlineBindings);
  CampaignExecutionPolicy shortCampaign = take(
      CampaignExecutionPolicy::get(1, 1, 1'000'000'000ULL, 150'000'000ULL));
  const auto deadlineBegin = std::chrono::steady_clock::now();
  CampaignExecutionResult deadlineResult = take(runFpaGroundTruthCampaign(
      view, closure, shortCampaign,
      executionPolicy(delayedLocal, ExternalAttemptDisposition::ExecutePrepared,
                      std::nullopt),
      deadlineScheduler, deadlineJournal, artifacts, blobs));
  const auto deadlineElapsed = std::chrono::steady_clock::now() - deadlineBegin;
  if (deadlineElapsed >= std::chrono::seconds(2))
    fail("FPA campaign did not enforce its selected hard wall");
  const DsePlanExecutionOutcome *deadlineOutcome = nullptr;
  if (const auto *refusal =
          std::get_if<CampaignAdmissionRefusal>(&deadlineResult))
    deadlineOutcome = &refusal->outcome;
  else
    deadlineOutcome = &std::get<CampaignExecution>(deadlineResult).outcome;
  const auto *deadlineIncomplete =
      std::get_if<IncompleteDsePlanExecution>(deadlineOutcome);
  const auto *deadlineReason =
      deadlineIncomplete ? std::get_if<PromotionAcquisitionIncompleteReason>(
                               &deadlineIncomplete->reason())
                         : nullptr;
  if (!deadlineReason ||
      *deadlineReason !=
          PromotionAcquisitionIncompleteReason::CancelledOrTimeout)
    fail("FPA hard wall did not preserve its typed timeout outcome");
  const std::vector<JournalWorkUnitRecord> deadlineRecords =
      take(deadlineJournal.workUnits());
  if (deadlineRecords.size() != 1 ||
      deadlineRecords.front().status != JournalWorkUnitStatus::TimedOut ||
      !deadlineRecords.front().finalizedOutputs.empty())
    fail("stopped FPA provider published a terminal output");

  SiteScheduler prepareScheduler = scheduler({});
  DsePlanExecutionOutcome preparedOutcome = take(resumeDsePlan(
      view, closure, journal, prepareScheduler,
      executionPolicy(local, ExternalAttemptDisposition::PrepareOnly, 1),
      artifacts, blobs, InvocationManifestRetention::Release));
  if (!std::holds_alternative<IncompleteDsePlanExecution>(preparedOutcome))
    fail("prepare-only prefix unexpectedly completed the campaign");
  const std::vector<BlobDigest> bindings = preparedBindings(journal);
  if (bindings.size() != 1)
    fail("authored campaign did not derive one exact execution binding");

  const loom::fabric::FabricModuleDomainMemberRef transportLeaf =
      transportCharacterizationLeaf(training, artifacts);
  FpaCharacterizationUnavailable leafUnavailable =
      take(assessFpaLeafCharacterizationTarget(
          {training.routed.reference(), transportLeaf}, artifacts, blobs));
  if (leafUnavailable.target.hardwareImplementation !=
          training.routed.reference() ||
      leafUnavailable.target.leaf != transportLeaf ||
      leafUnavailable.reason != FpaCharacterizationUnavailableReason::
                                    IndependentlyRoutedLeafUnavailable)
    fail("routed SpatialCore boundary was admitted as an independent leaf");
  FpaCharacterizationUnavailable gateUnavailable =
      take(assessFpaLeafCharacterizationTarget(
          {training.gate.gate.reference(), transportLeaf}, artifacts, blobs));
  if (gateUnavailable.reason !=
      FpaCharacterizationUnavailableReason::RoutedAsicImplementationUnavailable)
    fail("unrouted FPA target was not typed unavailable");
  const auto moduleTemplate =
      std::get<loom::fabric::FabricModuleBoundaryEndpointRef>(
          transportLeaf.payload)
          .module;
  requireErrorContains(
      assessFpaLeafCharacterizationTarget(
          {training.routed.reference(),
           loom::fabric::FabricModuleDomainMemberRef::of(
               loom::fabric::FabricModuleBoundaryEndpointRef{
                   moduleTemplate, loom::fabric::FabricPortDirection::Input,
                   1'000'000})},
          artifacts, blobs),
      "outside");

  SiteScheduler executionScheduler = scheduler(bindings);
  CampaignExecutionResult result = take(runFpaGroundTruthCampaign(
      view, closure, campaignPolicy,
      executionPolicy(local, ExternalAttemptDisposition::ExecutePrepared,
                      std::nullopt),
      executionScheduler, journal, artifacts, blobs));
  if (const auto *refusal = std::get_if<CampaignAdmissionRefusal>(&result))
    fail("ground-truth campaign admission was refused with reason " +
         std::to_string(static_cast<std::uint32_t>(refusal->reason)));
  const auto *execution = std::get_if<CampaignExecution>(&result);
  const auto *completed =
      execution ? std::get_if<CompletedDsePlanExecution>(&execution->outcome)
                : nullptr;
  if (!completed) {
    const auto &incomplete =
        std::get<IncompleteDsePlanExecution>(execution->outcome);
    fail("ground-truth campaign stopped at node " +
         std::to_string(incomplete.nodeOrdinal()) + " with reason " +
         toString(incomplete.reason()).str());
  }
  const std::array<std::pair<PlanOutputRef, ArtifactRootReference>, 3> outputs =
      {{{plan.trainingEvidence, training.routed.reference()},
        {plan.validationEvidence, validation.routed.reference()},
        {plan.heldOutEvidence, heldOut.routed.reference()}}};
  std::vector<ArtifactRootReference> evidence;
  evidence.reserve(outputs.size());
  for (const auto &[output, candidate] : outputs) {
    const auto partitionEvidence = completed->resolve(output);
    if (partitionEvidence.size() != 1)
      fail("ground-truth partition did not publish exactly one Evidence");
    evidence.push_back(partitionEvidence.front());
    verifyEvidence(partitionEvidence.front(), candidate, artifacts, blobs);
  }

  GroundTruthPlanInputs modelInputs;
  modelInputs.fpa = GroundTruthModelTrack{
      GroundTruthEvidencePartitions{
          {evidence[0]}, {evidence[1]}, {evidence[2]}, std::nullopt},
      DeterministicGbdtTrainingConfig{7, 1, 1, 1, 1, 2},
      take(DecimalValue::get(0, 0)), take(DecimalValue::get(0, 0))};
  ResolvedGroundTruthPlan modelPlan = take(
      buildGroundTruthPlan(defaultResolvedConfig(), std::move(modelInputs)));
  const ArtifactIdentity storedModelConfig = take(
      artifacts.put(ResolvedConfig::artifactSchema,
                    canonicalResolvedConfigBytes(modelPlan.resolvedConfig())));
  if (storedModelConfig != resolvedConfigIdentity(modelPlan.resolvedConfig()))
    fail("model ResolvedConfig publication changed identity");

  DseProducerSemanticBuildIdentity modelProducer =
      take(DseProducerSemanticBuildIdentity::get("loom.test.fpa_model.v1"));
  DseRunClosure modelClosure = take(DseRunClosure::get(
      std::move(modelProducer), modelPlan.semanticInputs(),
      modelPlan.resolvedConfig(), modelPlan.preexistingEvidence(), artifacts));
  const std::filesystem::path modelRunPath = temporary.path() / "model-run";
  std::filesystem::create_directories(modelRunPath);
  ExecutionJournal modelJournal = take(openExecutionJournal(
      modelRunPath.string(), modelClosure, modelPlan.view()));
  SiteScheduler modelScheduler = scheduler({});
  const PlanExecutionPolicy modelPolicy =
      take(PlanExecutionPolicy::get(1, take(SiteResourceClaim::get(1, 0, 0))));
  DsePlanExecutionOutcome modelOutcome = take(resumeDsePlan(
      modelPlan.view(), modelClosure, modelJournal, modelScheduler, modelPolicy,
      artifacts, blobs, InvocationManifestRetention::Release));
  const auto *modelCompleted =
      std::get_if<CompletedDsePlanExecution>(&modelOutcome);
  if (!modelCompleted || modelCompleted->generateInvocations().size() != 1 ||
      !modelCompleted->generateInvocationWasDispatched(0))
    fail("FPA model plan did not dispatch and complete one trainer");
  if (!modelPlan.fpaOutputs())
    fail("FPA model plan omitted its typed output table");
  const GroundTruthTrackOutputs &modelOutputs = *modelPlan.fpaOutputs();
  const auto trainedBundles =
      modelCompleted->resolve(modelOutputs.trainedBundle);
  const auto validationEvidence =
      modelCompleted->resolve(modelOutputs.validationEvidence);
  const auto releasedBundles =
      modelCompleted->resolve(modelOutputs.releasedBundle);
  const auto heldOutEvidence =
      modelCompleted->resolve(modelOutputs.heldOutEvidence);
  if (trainedBundles.size() != 1 || validationEvidence.size() != 1 ||
      releasedBundles.size() != 1 || heldOutEvidence.size() != 1 ||
      trainedBundles.front() != releasedBundles.front())
    fail("FPA model plan lost its exact calibration output closure");
  const ArtifactRootReference releasedBundle = releasedBundles.front();
  verifyCalibrationEvidence(validationEvidence.front(), releasedBundle,
                            evidence[1], artifacts, blobs);
  verifyCalibrationEvidence(heldOutEvidence.front(), releasedBundle,
                            evidence[2], artifacts, blobs);

  const std::vector<JournalWorkUnitRecord> modelRecords =
      take(modelJournal.workUnits());
  const auto trainerRecord =
      llvm::find_if(modelRecords, [](const JournalWorkUnitRecord &record) {
        return record.key.planNodeOrdinal() == 0;
      });
  if (trainerRecord == modelRecords.end() ||
      trainerRecord->status != JournalWorkUnitStatus::Completed ||
      !trainerRecord->finalizedWorkRecord ||
      trainerRecord->finalizedWorkRecord->schemaIdentity !=
          candidateGeneratorFinalizedWorkRecordSchemaIdentity ||
      trainerRecord->finalizedWorkRecord->schemaVersion !=
          candidateGeneratorFinalizedWorkRecordSchemaVersion)
    fail("FPA trainer did not retain its exact finalized recovery record");

  ExecutionJournal replayJournal = take(openExecutionJournal(
      modelRunPath.string(), modelClosure, modelPlan.view()));
  SiteScheduler replayScheduler = scheduler({});
  DsePlanExecutionOutcome replayed = take(resumeDsePlan(
      modelPlan.view(), modelClosure, replayJournal, replayScheduler,
      modelPolicy, artifacts, blobs, InvocationManifestRetention::Release));
  const auto *replayedCompleted =
      std::get_if<CompletedDsePlanExecution>(&replayed);
  if (!replayedCompleted ||
      replayedCompleted->generateInvocations().size() != 1 ||
      replayedCompleted->generateInvocationWasDispatched(0) ||
      replayedCompleted->resolve(modelOutputs.releasedBundle).size() != 1 ||
      replayedCompleted->resolve(modelOutputs.releasedBundle).front() !=
          releasedBundle)
    fail("FPA model journal replay changed or redispatched the trainer");

  EdaPredictionModelWeight weight =
      take(importEdaPredictionModelWeight(releasedBundle, artifacts, blobs));
  if (weight.reference() != releasedBundle ||
      weight.bundle().bundle().parameterContract() !=
          fpaModelParameterContractRef())
    fail("released FPA weight lost its immutable bundle identity");
  FpaTrainingEvidenceSample heldOutSample =
      take(importFpaTrainingEvidenceSample(evidence.back(), artifacts, blobs));
  ModelParameterInferenceOutcome heldOutInference =
      take(inferEdaPredictionModelWeight(weight, heldOutSample.features));
  if (!std::holds_alternative<ModelParameterPrediction>(heldOutInference))
    fail("released FPA bundle rejected its completed held-out feature view");
  FpaFeatureView outOfDomain = heldOutSample.features;
  const DecimalValue outsideVoltage = take(DecimalValue::get(1200, -3));
  outOfDomain.conditions.minimumSupplyVoltage = outsideVoltage;
  outOfDomain.conditions.maximumSupplyVoltage = outsideVoltage;
  ModelParameterInferenceOutcome outOfDomainInference =
      take(inferEdaPredictionModelWeight(weight, outOfDomain));
  if (!std::holds_alternative<OutOfDomainModelParameterInference>(
          outOfDomainInference))
    fail("released FPA bundle did not preserve typed OOD refusal");
}

} // namespace

int main() {
  exerciseGroundTruthCampaign();
  return 0;
}
