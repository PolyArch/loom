#include "DSE/CampaignRunner.h"
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
#include "Evaluation/Request.h"

#include "llvm/Support/Error.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
    "OpenROAD synthetic cbc7678e45cc";

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

SubjectTargetRef rootTarget(const ArtifactRootReference &hardware) {
  return {hardwareImplementationPhysicalSubjectRole(), hardware,
          SubjectTarget{hardware}};
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

  const std::filesystem::path fpaTool =
      take(writeAuthoredOpenRoadStaticFpaTool(temporary.path()));
  const LocalToolConfig local =
      makeOpenRoadLocalToolConfig(training.gate, fpaTool);
  SiteScheduler prepareScheduler = scheduler({});
  DsePlanExecutionOutcome prepared = take(resumeDsePlan(
      view, closure, journal, prepareScheduler,
      executionPolicy(local, ExternalAttemptDisposition::PrepareOnly, 1),
      artifacts, blobs));
  if (!std::holds_alternative<IncompleteDsePlanExecution>(prepared))
    fail("prepare-only prefix unexpectedly completed the campaign");
  const std::vector<BlobDigest> bindings = preparedBindings(journal);
  if (bindings.size() != 1)
    fail("authored campaign did not derive one exact execution binding");

  SiteScheduler executionScheduler = scheduler(bindings);
  CampaignExecutionPolicy campaign = take(CampaignExecutionPolicy::get(1, 1));
  CampaignExecutionResult result = take(runGroundTruthCampaign(
      view, closure, campaign,
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
  for (const auto &[output, candidate] : outputs) {
    const auto evidence = completed->resolve(output);
    if (evidence.size() != 1)
      fail("ground-truth partition did not publish exactly one Evidence");
    verifyEvidence(evidence.front(), candidate, artifacts, blobs);
  }
}

} // namespace

int main() {
  exerciseGroundTruthCampaign();
  return 0;
}
