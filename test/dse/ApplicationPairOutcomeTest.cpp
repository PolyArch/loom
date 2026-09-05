#include "BuildInternal.h"
#include "QualityInternal.h"

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ResolvedConfigView.h"
#include "Evaluation/Evidence.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "application pair outcome test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-application-pair-outcome", path_))
      fail("cannot create temporary directory: " + error.message());
  }
  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  std::string makeDirectory(llvm::StringRef name) const {
    llvm::SmallString<128> path(path_);
    llvm::sys::path::append(path, name);
    if (std::error_code error = llvm::sys::fs::create_directory(path))
      fail("cannot create test directory: " + error.message());
    return path.str().str();
  }

private:
  llvm::SmallString<128> path_;
};

void typedReasonProjection() {
  using PairDisposition = loom::application::ApplicationPairDecisionDisposition;
  using FrontierReason = loom::dse::ResourceTimeFrontierIncompleteReason;
  using RuntimeDisposition =
      loom::application::ApplicationMappingRuntimeDisposition;
  using namespace loom::application::build_detail;

  require(mapResourceTimeFrontierReasonToPairDisposition(
              FrontierReason::BudgetExhausted) ==
              PairDisposition::BudgetExhausted,
          "frontier budget reason lost its pair disposition");
  require(mapResourceTimeFrontierReasonToPairDisposition(
              FrontierReason::CancelledOrTimeout) ==
              PairDisposition::CancelledOrTimeout,
          "frontier cancellation lost its pair disposition");
  require(mapResourceTimeFrontierReasonToPairDisposition(
              FrontierReason::ProofNotEstablished) ==
              PairDisposition::MappingProofNotEstablished,
          "frontier proof gap became a budget outcome");
  require(mapResourceTimeFrontierReasonToPairDisposition(
              FrontierReason::Unsupported) ==
              PairDisposition::UnsupportedSemantic,
          "frontier unsupported reason lost its pair disposition");

  const std::array runtimeCases = {
      std::pair{RuntimeDisposition::Unsupported,
                PairDisposition::UnsupportedSemantic},
      std::pair{RuntimeDisposition::ProofNotEstablished,
                PairDisposition::MappingProofNotEstablished},
      std::pair{RuntimeDisposition::ExecutionFailed,
                PairDisposition::ImplementationFailure},
      std::pair{RuntimeDisposition::CancelledOrTimeout,
                PairDisposition::CancelledOrTimeout}};
  for (const auto &[runtime, expected] : runtimeCases) {
    const auto projected = mapRuntimeDispositionToPairDisposition(runtime);
    require(projected && *projected == expected,
            "runtime reason lost its canonical pair disposition");
  }
}

void spectrumSelectionProjection() {
  using PairDisposition = loom::application::ApplicationPairDecisionDisposition;
  using SpectrumClass = loom::dse::PreMappingSpectrumClass;
  using SpectrumReason = loom::dse::ResourceTimeSpectrumIncompleteReason;
  using namespace loom::application::build_detail;

  require(!classifyResourceTimeSelectionOutcome(std::nullopt, std::nullopt),
          "automatic spectrum selection required absent evidence");
  require(classifyResourceTimeSelectionOutcome(std::nullopt,
                                               SpectrumClass::MaxTemporal) ==
              PairDisposition::UnsupportedSemantic,
          "explicit endpoint without evidence lost its unsupported outcome");

  const std::array incompleteCases = {
      std::pair{SpectrumReason::Unsupported,
                PairDisposition::UnsupportedSemantic},
      std::pair{SpectrumReason::ProofNotEstablished,
                PairDisposition::MappingProofNotEstablished},
      std::pair{SpectrumReason::CancelledOrTimeout,
                PairDisposition::CancelledOrTimeout}};
  for (const auto &[reason, expected] : incompleteCases) {
    std::optional<loom::dse::ResourceTimeSpectrumFunnelResult> spectrum{
        loom::dse::ResourceTimeSpectrumFunnelResult{
            loom::dse::ResourceTimeSpectrumVerification{
                loom::dse::IncompleteResourceTimeSpectrum{reason, "typed", 0}},
            loom::dse::ResourceTimeSpectrumFunnelAccounting{}}};
    require(classifyResourceTimeSelectionOutcome(
                spectrum, SpectrumClass::MaxTemporal) == expected,
            "spectrum incomplete reason lost its pair disposition");
  }

  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> bytes{};
  loom::ArtifactRootReference root{
      "loom.test.application_pair",
      {1, 0},
      take(loom::ArtifactIdentity::fromBytes(bytes))};
  loom::dse::VerifiedResourceTimeSpectrumScenario scenario;
  scenario.spectrumClass = SpectrumClass::Intermediate;
  scenario.systemMappings = {root};
  std::optional<loom::dse::ResourceTimeSpectrumFunnelResult> verified{
      loom::dse::ResourceTimeSpectrumFunnelResult{
          loom::dse::ResourceTimeSpectrumVerification{
              loom::dse::VerifiedResourceTimeSpectrum{root, root, {scenario}}},
          loom::dse::ResourceTimeSpectrumFunnelAccounting{}}};
  require(!classifyResourceTimeSelectionOutcome(verified,
                                                SpectrumClass::Intermediate, root),
          "verified requested endpoint was rejected");
  require(classifyResourceTimeSelectionOutcome(verified,
                                               SpectrumClass::MaxSpatial, root) ==
              PairDisposition::UnsupportedSemantic,
          "verified non-endpoint schedule lost its unsupported outcome");
  std::array<std::uint8_t, loom::ComponentViewDigest::byteSize> digestBytes{};
  const loom::ComponentViewDigest digest =
      take(loom::ComponentViewDigest::fromBytes(digestBytes));
  loom::application::ApplicationIncrementalMappingObservation mismatchedRepair(
      root, root, digest, digest);
  mismatchedRepair.spectrumEndpoint =
      loom::dse::PreMappingSpectrumEndpoint::MaxSpatial;
  mismatchedRepair.coldSelectionSpectrum = *verified;
  mismatchedRepair.incrementalSelectionSpectrum = *verified;
  const ApplicationIncrementalMappingOutcome mismatchedOutcome =
      deriveIncrementalMappingOutcome(mismatchedRepair);
  require(
      !mismatchedOutcome.verified && mismatchedOutcome.incompleteReason &&
          std::holds_alternative<loom::dse::CandidateGeneratorIncompleteReason>(
              *mismatchedOutcome.incompleteReason) &&
          std::get<loom::dse::CandidateGeneratorIncompleteReason>(
              *mismatchedOutcome.incompleteReason) ==
              loom::dse::CandidateGeneratorIncompleteReason::Unsupported,
      "adjacent endpoint mismatch lost its typed unsupported outcome");
}

void incompleteCausePriority() {
  using PairDisposition = loom::application::ApplicationPairDecisionDisposition;
  using loom::application::build_detail::prioritizeIncompletePairDisposition;

  const std::array proofCause = {PairDisposition::MappingProofNotEstablished};
  require(prioritizeIncompletePairDisposition(proofCause, true) ==
              PairDisposition::MappingProofNotEstablished,
          "declared work exhaustion masked a typed proof gap");
  const std::array mixedCauses = {PairDisposition::UnsupportedSemantic,
                                  PairDisposition::CancelledOrTimeout};
  require(prioritizeIncompletePairDisposition(mixedCauses, true) ==
              PairDisposition::CancelledOrTimeout,
          "earlier incomplete evidence masked cancellation");
  require(prioritizeIncompletePairDisposition({}, true) ==
              PairDisposition::BudgetExhausted,
          "unattributed declared work exhaustion lost its fallback");
}

void noFeasibleOutcomePreservesTypedCause() {
  using PairDisposition = loom::application::ApplicationPairDecisionDisposition;
  using PlanningDisposition = loom::dse::PreMappingCandidatePlanningDisposition;
  using loom::application::build_detail::classifyPreMappingNoFeasibleOutcome;

  loom::dse::CompletedPreMappingNoFeasibleCandidate proofGap;
  loom::dse::PreMappingCandidatePlanningRecord record;
  record.disposition = PlanningDisposition::Unknown;
  record.incompleteReason = loom::dse::DsePlanIncompleteReason{
      loom::dse::CandidateGeneratorIncompleteReason::ProofNotEstablished};
  proofGap.candidateInventory.push_back(std::move(record));
  require(classifyPreMappingNoFeasibleOutcome(proofGap) ==
              PairDisposition::MappingProofNotEstablished,
          "no-feasible preparation collapsed a proof gap into budget");

  loom::dse::CompletedPreMappingNoFeasibleCandidate exactRejection;
  exactRejection.completeness = {true, true, true, true, true};
  loom::dse::PreMappingCandidatePlanningRecord rejected;
  rejected.disposition = PlanningDisposition::ExactGateRejected;
  exactRejection.candidateInventory.push_back(std::move(rejected));
  require(classifyPreMappingNoFeasibleOutcome(exactRejection) ==
              PairDisposition::NoPromisingCandidate,
          "exact candidate rejection became an incomplete outcome");

  exactRejection.completeness.selectionComplete = false;
  require(classifyPreMappingNoFeasibleOutcome(exactRejection) ==
              PairDisposition::MappingProofNotEstablished,
          "partial exact rejection became a complete negative outcome");

  loom::dse::CompletedPreMappingNoFeasibleCandidate heuristicRejection;
  heuristicRejection.completeness = {false, true, true, true, true};
  loom::dse::PreMappingCandidatePlanningRecord heuristic;
  heuristic.disposition = PlanningDisposition::HeuristicPruned;
  heuristicRejection.candidateInventory.push_back(std::move(heuristic));
  require(classifyPreMappingNoFeasibleOutcome(heuristicRejection) ==
              PairDisposition::MappingProofNotEstablished,
          "heuristic pruning was mislabeled as budget exhaustion");
}

void qualityDispositionProjection() {
  using PairDisposition = loom::application::ApplicationPairDecisionDisposition;
  using QualityDisposition = loom::dse::JointDesignQualityDisposition;
  using namespace loom::application;

  TemporaryDirectory directory;
  loom::ArtifactStore artifacts(directory.makeDirectory("artifacts"));
  loom::BlobStore blobs(directory.makeDirectory("blobs"));
  const loom::dse::ResolvedDseConfigView view = take(
      loom::dse::projectResolvedDseConfigView(loom::defaultResolvedConfig()));

  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> rootBytes{};
  rootBytes.back() = 1;
  const loom::ArtifactRootReference root{
      "loom.test.application_pair_quality",
      {1, 0},
      take(loom::ArtifactIdentity::fromBytes(rootBytes))};
  std::array<std::uint8_t, loom::ComponentViewDigest::byteSize> digestBytes{};
  digestBytes.back() = 2;
  const loom::ComponentViewDigest digest =
      take(loom::ComponentViewDigest::fromBytes(digestBytes));
  PreparedApplicationBuild prepared{
      {},
      take(loom::dse::JointDesignPolicy::get(1, 1, 1, 1, 1)),
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
      {},
      {},
      {},
      loom::dse::StructuredOwnershipSelectionMode::SemanticConformance,
      loom::dse::StructuredOwnershipSelectionMode::SemanticConformance,
      {},
      {},
      {},
      root,
      root,
      root,
      root,
      digest,
      0,
      0,
      {},
      {},
      {},
      {},
      std::nullopt};
  const std::array cases = {
      std::pair{QualityDisposition::Unsupported,
                PairDisposition::UnsupportedSemantic},
      std::pair{QualityDisposition::ProofNotEstablished,
                PairDisposition::MappingProofNotEstablished},
      std::pair{QualityDisposition::ExecutionFailed,
                PairDisposition::ImplementationFailure},
      std::pair{QualityDisposition::CancelledOrTimeout,
                PairDisposition::CancelledOrTimeout}};
  for (const auto &[quality, expected] : cases) {
    loom::dse::JointDesignExecutionSummary summary;
    summary.qualityDisposition = quality;
    loom::dse::JointDesignExecution execution{
        take(loom::dse::executeDsePlan(view, artifacts, blobs)),
        {},
        std::move(summary)};
    const ApplicationPairDecisionRecord decision =
        build_detail::deriveApplicationPairDecision(prepared, {}, execution, {},
                                                    {});
    require(decision.disposition == expected,
            "summary quality disposition was not used by the pair owner");
  }

  loom::dse::JointDesignExecutionSummary conflictingSummary;
  conflictingSummary.qualityDisposition =
      QualityDisposition::CancelledOrTimeout;
  loom::dse::JointDesignExecution conflictingExecution{
      take(loom::dse::executeDsePlan(view, artifacts, blobs)),
      {},
      std::move(conflictingSummary)};
  ApplicationPairQualityInvocationRecord invocation;
  invocation.qualityDisposition = QualityDisposition::Unsupported;
  const std::array invocations = {std::move(invocation)};
  const ApplicationPairDecisionRecord invocationDecision =
      build_detail::deriveApplicationPairDecision(
          prepared, {}, conflictingExecution, {}, invocations);
  require(invocationDecision.disposition ==
              PairDisposition::UnsupportedSemantic,
          "summary quality disposition overrode its invocation owner");

  prepared.resourceTimePolicy.spectrumEndpoint =
      loom::dse::PreMappingSpectrumEndpoint::MaxSpatial;
  loom::dse::VerifiedResourceTimeSpectrumScenario nonEndpointScenario;
  nonEndpointScenario.spectrumClass =
      loom::dse::PreMappingSpectrumClass::Intermediate;
  nonEndpointScenario.systemMappings = {root};
  std::optional<loom::dse::ResourceTimeSpectrumFunnelResult>
      nonEndpointSpectrum{loom::dse::ResourceTimeSpectrumFunnelResult{
          loom::dse::ResourceTimeSpectrumVerification{
              loom::dse::VerifiedResourceTimeSpectrum{
                  root, root, {std::move(nonEndpointScenario)}}},
          loom::dse::ResourceTimeSpectrumFunnelAccounting{}}};
  const std::vector<ApplicationMappingCandidateOutcome> nonEndpointOutcomes = {
      ApplicationMappingCandidateOutcome{
          0,
          0,
          digest,
          root,
          root,
          loom::dse::JointDesignAttemptDisposition::Verified,
          std::nullopt,
          std::nullopt,
          {root},
          std::nullopt,
          std::nullopt,
          {},
          ApplicationMappingRuntimeDisposition::NotRequested,
          {},
          {},
          std::move(nonEndpointSpectrum),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          {},
          std::nullopt,
          std::nullopt}};
  loom::dse::JointDesignExecution nonEndpointExecution{
      take(loom::dse::executeDsePlan(view, artifacts, blobs)), {}, {}};
  const ApplicationPairDecisionRecord nonEndpointDecision =
      build_detail::deriveApplicationPairDecision(prepared, nonEndpointOutcomes,
                                                  nonEndpointExecution, {}, {});
  require(nonEndpointDecision.disposition ==
              PairDisposition::UnsupportedSemantic,
          "not-requested runtime masked a completed endpoint mismatch");

  const auto unsupportedSpectrum = [] {
    return loom::dse::ResourceTimeSpectrumFunnelResult{
        loom::dse::ResourceTimeSpectrumVerification{
            loom::dse::IncompleteResourceTimeSpectrum{
                loom::dse::ResourceTimeSpectrumIncompleteReason::Unsupported,
                "typed Mapping rejection", 0}},
        loom::dse::ResourceTimeSpectrumFunnelAccounting{}};
  };
  ApplicationIncrementalMappingObservation incompleteRepair(root, root, digest,
                                                            digest);
  incompleteRepair.spectrumEndpoint =
      prepared.resourceTimePolicy.spectrumEndpoint;
  incompleteRepair.coldSelectionSpectrum = unsupportedSpectrum();
  incompleteRepair.incrementalSelectionSpectrum = unsupportedSpectrum();
  const std::array repairObservations = {std::move(incompleteRepair)};
  const ApplicationPairDecisionRecord repairDecision =
      build_detail::deriveApplicationPairDecision(
          prepared, {}, conflictingExecution, repairObservations, invocations);
  require(
      repairDecision.resourceTimeMappingRepairAttemptCount == 1 &&
          repairDecision.resourceTimeMappingRepairVerifiedCount == 0 &&
          repairDecision.resourceTimeMappingRepairIncompleteReason &&
          std::holds_alternative<loom::dse::CandidateGeneratorIncompleteReason>(
              *repairDecision.resourceTimeMappingRepairIncompleteReason) &&
          std::get<loom::dse::CandidateGeneratorIncompleteReason>(
              *repairDecision.resourceTimeMappingRepairIncompleteReason) ==
              loom::dse::CandidateGeneratorIncompleteReason::Unsupported,
      "pair decision lost an adjacent Mapping repair's typed outcome");

  ApplicationIncrementalMappingObservation planIncompleteRepair(root, root,
                                                                digest, digest);
  planIncompleteRepair.coldExecutionIncompleteReasons.push_back(
      loom::dse::CandidateGeneratorIncompleteReason::Unsupported);
  planIncompleteRepair.incrementalExecutionIncompleteReasons.push_back(
      loom::dse::CandidateGeneratorIncompleteReason::Unsupported);
  const build_detail::ApplicationIncrementalMappingOutcome planOutcome =
      build_detail::deriveIncrementalMappingOutcome(planIncompleteRepair);
  require(
      !planOutcome.verified && planOutcome.incompleteReason &&
          std::holds_alternative<loom::dse::CandidateGeneratorIncompleteReason>(
              *planOutcome.incompleteReason) &&
          std::get<loom::dse::CandidateGeneratorIncompleteReason>(
              *planOutcome.incompleteReason) ==
              loom::dse::CandidateGeneratorIncompleteReason::Unsupported,
      "absent post-plan evidence masked the exact incomplete-plan reason");
}

void qualityRuntimeCompletionProjection() {
  using loom::application::ApplicationMappingRuntimeDisposition;
  using loom::application::detail::classifyApplicationQualityRuntime;
  using loom::dse::JointDesignQualityIncompleteReason;
  using loom::dse::JointDesignQualityProvenanceDomain;
  using loom::dse::JointDesignQualityRuntimeCompletion;

  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> rootBytes{};
  rootBytes.back() = 3;
  const loom::ArtifactRootReference mapping{
      "loom.test.application_runtime_quality",
      {1, 0},
      take(loom::ArtifactIdentity::fromBytes(rootBytes))};
  rootBytes.back() = 4;
  const loom::ArtifactRootReference runtimeEvidence{
      loom::evaluation::EvaluationEvidence::artifactSchema.identity.str(),
      loom::evaluation::EvaluationEvidence::artifactSchema.version,
      take(loom::ArtifactIdentity::fromBytes(rootBytes))};
  rootBytes.back() = 5;
  const loom::ArtifactRootReference verificationEvidence{
      loom::evaluation::EvaluationEvidence::artifactSchema.identity.str(),
      loom::evaluation::EvaluationEvidence::artifactSchema.version,
      take(loom::ArtifactIdentity::fromBytes(rootBytes))};

  loom::dse::JointBoundedQualityPolicy runtimeQuality;
  runtimeQuality.provenanceDomain =
      JointDesignQualityProvenanceDomain::ApplicationRuntime;
  runtimeQuality.objectiveDimensionLabels = {"dfg_cycles", "cgra_cycles",
                                             "acc_core_count"};

  loom::dse::JointDesignQualityObservation fpaRefusal{
      mapping,
      {},
      JointDesignQualityIncompleteReason::Unsupported,
      std::nullopt,
      {{loom::resolvedObjectiveInteger(7), loom::resolvedObjectiveInteger(11),
        loom::resolvedObjectiveInteger(2)},
       {runtimeEvidence, verificationEvidence},
       {verificationEvidence},
       {},
       {},
       {},
       2,
       JointDesignQualityRuntimeCompletion::Completed,
       loom::dse::JointDesignCalibratedModelSupport::OutOfDomain}};
  require(take(classifyApplicationQualityRuntime(runtimeQuality, fpaRefusal)) ==
              ApplicationMappingRuntimeDisposition::Completed,
          "FPA refusal erased completed Application runtime");

  loom::dse::JointDesignQualityObservation runtimeRefusal{
      mapping,
      {},
      JointDesignQualityIncompleteReason::Unsupported,
      std::nullopt,
      {{}, {}, {}, {}, {}, {}, 2}};
  require(
      take(classifyApplicationQualityRuntime(runtimeQuality, runtimeRefusal)) ==
          ApplicationMappingRuntimeDisposition::Unsupported,
      "runtime refusal was not retained independently of quality");

  auto missingMeasures = fpaRefusal;
  missingMeasures.provenance.rawMeasures.clear();
  auto missingMeasuresResult =
      classifyApplicationQualityRuntime(runtimeQuality, missingMeasures);
  require(!missingMeasuresResult,
          "completed runtime accepted missing runtime measures");
  llvm::consumeError(missingMeasuresResult.takeError());

  auto missingResource = fpaRefusal;
  missingResource.provenance.resourceCoreCost.reset();
  auto missingResourceResult =
      classifyApplicationQualityRuntime(runtimeQuality, missingResource);
  require(!missingResourceResult,
          "completed runtime accepted missing resource ownership");
  llvm::consumeError(missingResourceResult.takeError());

  auto missingEvidence = fpaRefusal;
  missingEvidence.provenance.supportingEvidence.clear();
  missingEvidence.provenance.verificationEvidence.clear();
  auto missingEvidenceResult =
      classifyApplicationQualityRuntime(runtimeQuality, missingEvidence);
  require(!missingEvidenceResult,
          "completed runtime accepted missing supporting Evidence");
  llvm::consumeError(missingEvidenceResult.takeError());

  auto missingOracle = fpaRefusal;
  missingOracle.provenance.verificationEvidence.clear();
  auto missingOracleResult =
      classifyApplicationQualityRuntime(runtimeQuality, missingOracle);
  require(!missingOracleResult,
          "completed runtime accepted missing verification Evidence");
  llvm::consumeError(missingOracleResult.takeError());

  auto oracleOnly = fpaRefusal;
  oracleOnly.provenance.supportingEvidence = {verificationEvidence};
  auto oracleOnlyResult =
      classifyApplicationQualityRuntime(runtimeQuality, oracleOnly);
  require(!oracleOnlyResult,
          "completed runtime accepted verification Evidence without an "
          "independent runtime witness");
  llvm::consumeError(oracleOnlyResult.takeError());

  auto ownerlessMeasures = fpaRefusal;
  ownerlessMeasures.provenance.runtimeCompletion =
      JointDesignQualityRuntimeCompletion::NotEstablished;
  auto ownerlessMeasuresResult =
      classifyApplicationQualityRuntime(runtimeQuality, ownerlessMeasures);
  require(!ownerlessMeasuresResult,
          "runtime measures established completion without their owner");
  llvm::consumeError(ownerlessMeasuresResult.takeError());

  auto objectiveOnly = runtimeQuality;
  objectiveOnly.provenanceDomain =
      JointDesignQualityProvenanceDomain::ObjectiveOnly;
  auto foreignDomain =
      classifyApplicationQualityRuntime(objectiveOnly, fpaRefusal);
  require(!foreignDomain,
          "ObjectiveOnly quality manufactured an Application runtime result");
  llvm::consumeError(foreignDomain.takeError());
}

} // namespace

int main() {
  typedReasonProjection();
  spectrumSelectionProjection();
  incompleteCausePriority();
  noFeasibleOutcomePreservesTypedCause();
  qualityDispositionProjection();
  qualityRuntimeCompletionProjection();
  return 0;
}
