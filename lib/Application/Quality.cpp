#include "Application/Build.h"
#include "ApplicationRuntimeValidationInternal.h"
#include "QualityInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/EvidenceObligation.h"
#include "DSE/Objective.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/CalibratedFpa.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/FpaParameterContract.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::application {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_build_invalid: " + message);
}

struct ApplicationFpaCompletedObservation final {
  std::array<ResolvedObjectiveScalar, 4> values;
  ArtifactRootReference evidence;
};

struct ApplicationFpaIncompleteObservation final {
  dse::JointDesignQualityIncompleteReason reason;
  ArtifactRootReference evidence;
  dse::JointDesignCalibratedModelSupport modelSupport =
      dse::JointDesignCalibratedModelSupport::NotEvaluated;
};

using ApplicationFpaObservation =
    std::variant<ApplicationFpaCompletedObservation,
                 ApplicationFpaIncompleteObservation>;

constexpr std::array<evaluation::MetricKind, 4> applicationFpaMetrics = {
    evaluation::MetricKind::LimitingClockFrequency,
    evaluation::MetricKind::TotalArea, evaluation::MetricKind::DynamicPower,
    evaluation::MetricKind::LeakagePower};

bool dispatchDeadlineReached(const dse::PlanExecutionPolicy &policy) {
  if (!policy.dispatchNotAfterUnixNanoseconds())
    return false;
  const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
  const auto now =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  return now > 0 && *policy.dispatchNotAfterUnixNanoseconds() <=
                        static_cast<std::uint64_t>(now);
}

llvm::Expected<ApplicationFpaObservation> acquireFpaObservation(
    const ArtifactRootReference &dataflow, const ArtifactRootReference &system,
    const evaluation::models::EdaPredictionModelWeight &weight,
    llvm::ArrayRef<evaluation::EvaluationCondition> operatingConditions,
    const ResolvedConfig &config, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  // Operating conditions are authored against the product System; every
  // hardware candidate is evaluated under the same conditions rebound to its
  // own Fabric root, so an alternative System never inherits a foreign anchor.
  std::vector<evaluation::EvaluationCondition> conditions(
      operatingConditions.begin(), operatingConditions.end());
  if (llvm::Error error = dse::rebindCandidateConditionTargets(
          conditions,
          evaluation::models::canonicalDataflowFabricAnalyticFabricRole(),
          system))
    return std::move(error);
  auto prepared =
      evaluation::models::prepareCanonicalDataflowFabricCalibratedFpaEvaluation(
          dataflow, system, weight, conditions, config, artifacts, blobs);
  if (!prepared)
    return prepared.takeError();
  auto evaluated = evaluation::evaluateRequest(
      prepared->request, prepared->resolution, artifacts, blobs);
  if (!evaluated)
    return evaluated.takeError();
  auto evidence = evaluation::publishEvaluationEvidence(*evaluated, artifacts);
  if (!evidence)
    return evidence.takeError();

  auto imported = evaluation::importEvaluationEvidence(
      *evidence, prepared->resolution, artifacts, blobs);
  if (!imported)
    return invalid("published calibrated FPA Evidence failed strict import: " +
                   llvm::toString(imported.takeError()));
  if (imported->requestRef() !=
          evaluation::evaluationRequestReference(prepared->request) ||
      imported->outcomeKind() != evaluated->outcomeKind())
    return invalid("strict calibrated FPA Evidence changed its Request or "
                   "outcome");

  if (std::holds_alternative<evaluation::UnsupportedEvidence>(
          imported->outcome())) {
    auto inference = evaluation::models::inferCanonicalDataflowFabricFpa(
        dataflow, system, weight, conditions, artifacts, blobs);
    if (!inference)
      return inference.takeError();
    if (!std::holds_alternative<evaluation::OutOfDomainModelParameterInference>(
            inference->outcome))
      return invalid("unsupported calibrated FPA Evidence was not produced "
                     "by typed out-of-domain inference");
    return ApplicationFpaIncompleteObservation{
        dse::JointDesignQualityIncompleteReason::Unsupported,
        std::move(*evidence),
        dse::JointDesignCalibratedModelSupport::OutOfDomain};
  }
  if (std::holds_alternative<evaluation::ExecutionFailedEvidence>(
          imported->outcome()))
    return ApplicationFpaIncompleteObservation{
        dse::JointDesignQualityIncompleteReason::ExecutionFailed,
        std::move(*evidence)};
  if (std::holds_alternative<evaluation::CancelledOrTimeoutEvidence>(
          imported->outcome()))
    return ApplicationFpaIncompleteObservation{
        dse::JointDesignQualityIncompleteReason::CancelledOrTimeout,
        std::move(*evidence)};

  const auto *completed =
      std::get_if<evaluation::CompletedEvidence>(&imported->outcome());
  if (!completed || completed->metricResults.size() != 4 ||
      prepared->request.metricRequests().size() != 4)
    return invalid("calibrated FPA Evidence has a foreign metric shape");
  std::array<ResolvedObjectiveScalar, 4> values;
  const evaluation::DecimalValue zero =
      llvm::cantFail(evaluation::DecimalValue::get(0, 0));
  for (std::size_t ordinal = 0; ordinal != applicationFpaMetrics.size();
       ++ordinal) {
    if (prepared->request.metricRequests()[ordinal].query().metric !=
        applicationFpaMetrics[ordinal])
      return invalid("calibrated FPA Request changed its metric order");
    const auto *point = std::get_if<evaluation::PointObservation>(
        &completed->metricResults[ordinal].observation);
    const auto *decimal =
        point ? std::get_if<evaluation::DecimalValue>(&point->value) : nullptr;
    if (!decimal)
      return invalid("calibrated FPA Evidence is not a Decimal Point");
    const int zeroComparison = evaluation::compareDecimalValue(*decimal, zero);
    if (zeroComparison < 0 || (zeroComparison == 0 && ordinal < 2))
      return ApplicationFpaIncompleteObservation{
          dse::JointDesignQualityIncompleteReason::ProofNotEstablished,
          std::move(*evidence),
          dse::JointDesignCalibratedModelSupport::InDomain};
    values[ordinal] = resolvedObjectiveDecimal(decimal->coefficient(),
                                               decimal->base10Exponent());
  }
  return ApplicationFpaCompletedObservation{std::move(values),
                                            std::move(*evidence)};
}

llvm::Expected<dse::CandidateMeasureObjectiveDimension>
makeFpaDimension(evaluation::MetricKind metric, std::uint32_t ordinal) {
  auto exponent = evaluation::models::
      canonicalDataflowFabricAnalyticMetricQuantumBase10Exponent(metric);
  if (!exponent)
    return exponent.takeError();
  return dse::CandidateMeasureObjectiveDimension{
      ordinal,
      metric == evaluation::MetricKind::LimitingClockFrequency
          ? ResolvedObjectiveDirection::Maximize
          : ResolvedObjectiveDirection::Minimize,
      0,
      std::numeric_limits<std::uint64_t>::max(),
      resolvedObjectiveDecimal(0, 0),
      resolvedObjectiveDecimal(1, *exponent)};
}

} // namespace

std::optional<dse::JointBoundedQualityPolicy>
detail::rebaseApplicationBoundedQualityPolicy(
    const std::optional<dse::JointBoundedQualityPolicy> &policy,
    std::uint64_t planOrdinalBase) {
  std::optional<dse::JointBoundedQualityPolicy> result = policy;
  if (!result || planOrdinalBase == 0)
    return result;
  dse::JointDesignQualityAcquirer acquire = result->acquire;
  result->acquire = [acquire = std::move(acquire), planOrdinalBase](
                        const dse::JointDesignExecution &execution,
                        std::uint64_t planOrdinal)
      -> llvm::Expected<dse::JointDesignQualityAcquisition> {
    if (planOrdinal >
        std::numeric_limits<std::uint64_t>::max() - planOrdinalBase)
      return invalid("bounded-quality plan ordinal overflowed");
    return acquire(execution, planOrdinal + planOrdinalBase);
  };
  if (result->hardwarePromotion) {
    dse::JointHardwarePromotionQualityAcquirer promote =
        result->hardwarePromotion->acquire;
    result->hardwarePromotion->acquire =
        [promote = std::move(promote),
         planOrdinalBase](const dse::JointDesignExplorationPlan &plan,
                          std::uint64_t planOrdinal)
        -> llvm::Expected<dse::JointDesignQualityAcquisition> {
      if (planOrdinal >
          std::numeric_limits<std::uint64_t>::max() - planOrdinalBase)
        return invalid("hardware-promotion plan ordinal overflowed");
      return promote(plan, planOrdinal + planOrdinalBase);
    };
  }
  return result;
}

llvm::Error detail::recordApplicationQualityInvocation(
    dse::JointDesignExecution &execution, std::uint64_t planOrdinalBase,
    std::vector<ApplicationPairQualityInvocationRecord> &invocations) {
  dse::JointDesignExecutionSummary &summary = execution.summary;
  const auto invocationRunKey = execution.invocationRunKey();
  const std::uint64_t promotedParents =
      static_cast<std::uint64_t>(llvm::count_if(
          summary.hardwarePromotionObservations,
          [](const dse::JointHardwarePromotionObservation &observation) {
            return observation.promotedToExactMapping;
          }));
  if (!summary.hardwarePromotionObservations.empty() &&
      summary.hardwareParentPromotions != promotedParents)
    return invalid("hardware-promotion count disagrees with its invocation "
                   "observations");
  if (!summary.hardwarePromotionObservations.empty() && !invocationRunKey)
    return invalid("hardware-promotion observations have no invocation "
                   "manifest run key");
  for (const dse::JointDesignAttemptRecord &attempt : summary.attempts) {
    if (!attempt.hardwarePromotionParentSystem)
      continue;
    const std::size_t parentCount = llvm::count_if(
        summary.hardwarePromotionObservations,
        [&](const dse::JointHardwarePromotionObservation &observation) {
          return observation.promotedToExactMapping &&
                 observation.planOrdinal == attempt.planOrdinal &&
                 observation.system == *attempt.hardwarePromotionParentSystem;
        });
    if (parentCount != 1)
      return invalid("hardware-promotion Mapping attempt has no unique "
                     "invocation-local parent observation");
  }
  invocations.push_back(ApplicationPairQualityInvocationRecord{
      planOrdinalBase, invocationRunKey, summary.qualityDisposition,
      summary.qualityIncompleteCandidate,
      summary.qualityObjectiveDimensionLabels, summary.qualityObservations,
      summary.hardwarePromotionObjectiveDimensionLabels,
      summary.hardwarePromotionObservations, summary.hardwareParentPromotions,
      summary.attempts, summary.selectedPlanOrdinal, summary.selectedMapping});
  for (dse::JointHardwarePromotionObservation &observation :
       summary.hardwarePromotionObservations) {
    if (observation.planOrdinal >
        std::numeric_limits<std::uint64_t>::max() - planOrdinalBase)
      return invalid("hardware-promotion observation ordinal overflowed");
    observation.planOrdinal += planOrdinalBase;
  }
  return llvm::Error::success();
}

llvm::Expected<detail::ApplicationRepairQualityChoice>
detail::chooseApplicationRepairByQuality(
    llvm::ArrayRef<dse::JointDesignExecution> executions,
    const std::optional<dse::JointBoundedQualityPolicy> &quality,
    const ArtifactStore &artifacts) {
  if (!quality || executions.empty())
    return ApplicationRepairQualityChoice{};
  auto selected =
      dse::selectJointRepairMappingByQuality(executions, *quality, artifacts);
  if (!selected)
    return selected.takeError();
  if (auto *incomplete =
          std::get_if<dse::JointRepairQualityIncomplete>(&*selected)) {
    if (incomplete->executionOrdinal >= executions.size())
      return invalid("repair quality incomplete result lost its execution "
                     "owner");
    const dse::JointDesignExecutionSummary &summary =
        executions[incomplete->executionOrdinal].summary;
    if (summary.selectedMapping ||
        summary.qualityIncompleteCandidate != incomplete->incomplete.candidate)
      return invalid("repair quality incomplete result lost its exact "
                     "Mapping join");
    return ApplicationRepairQualityChoice{std::move(*incomplete)};
  }
  const dse::JointRepairQualitySelection &choice =
      std::get<dse::JointRepairQualitySelection>(*selected);
  if (choice.executionOrdinal >= executions.size() ||
      executions[choice.executionOrdinal].summary.selectedMapping !=
          choice.mapping)
    return invalid("repair quality selection lost its exact Mapping join");
  return ApplicationRepairQualityChoice{choice};
}

llvm::Expected<detail::ApplicationRuntimeValidation>
detail::projectApplicationQualityRuntime(
    const dse::JointDesignExecution &execution,
    const ArtifactRootReference &mapping,
    const dse::JointBoundedQualityPolicy &quality,
    const ArtifactStore &artifacts) {
  if (!quality.objectiveProgram ||
      quality.provenanceDomain !=
          dse::JointDesignQualityProvenanceDomain::ApplicationRuntime ||
      execution.summary.qualityObjectiveDimensionLabels !=
          quality.objectiveDimensionLabels)
    return invalid("application runtime projection has a foreign objective "
                   "domain");
  const auto matching =
      llvm::find_if(execution.summary.qualityObservations,
                    [&](const dse::JointDesignQualityObservation &observation) {
                      return observation.candidate == mapping;
                    });
  if (matching == execution.summary.qualityObservations.end())
    return invalid("application runtime projection has no quality "
                   "observation");
  if (llvm::count_if(execution.summary.qualityObservations,
                     [&](const auto &observation) {
                       return observation.candidate == mapping;
                     }) != 1)
    return invalid("application runtime projection has duplicate Mapping "
                   "observations");
  auto runtimeDisposition =
      classifyApplicationQualityRuntime(quality, *matching);
  if (!runtimeDisposition)
    return runtimeDisposition.takeError();
  if (matching->incompleteReason) {
    if (execution.summary.selectedMapping ||
        execution.summary.qualityIncompleteCandidate != mapping)
      return invalid("application runtime incomplete observation disagrees "
                     "with its summary");
    const auto summaryReason =
        [&]() -> std::optional<dse::JointDesignQualityIncompleteReason> {
      switch (execution.summary.qualityDisposition) {
      case dse::JointDesignQualityDisposition::Unsupported:
        return dse::JointDesignQualityIncompleteReason::Unsupported;
      case dse::JointDesignQualityDisposition::ProofNotEstablished:
        return dse::JointDesignQualityIncompleteReason::ProofNotEstablished;
      case dse::JointDesignQualityDisposition::ExecutionFailed:
        return dse::JointDesignQualityIncompleteReason::ExecutionFailed;
      case dse::JointDesignQualityDisposition::CancelledOrTimeout:
        return dse::JointDesignQualityIncompleteReason::CancelledOrTimeout;
      case dse::JointDesignQualityDisposition::NotRequested:
      case dse::JointDesignQualityDisposition::Complete:
        return std::nullopt;
      }
      llvm_unreachable("unknown application quality disposition");
    }();
    if (summaryReason != matching->incompleteReason)
      return invalid("application runtime incomplete reason disagrees with "
                     "its summary");
  } else {
    const bool selectedComplete =
        execution.summary.qualityDisposition ==
            dse::JointDesignQualityDisposition::Complete &&
        execution.summary.selectedMapping == mapping;
    const bool searchIncomplete =
        execution.summary.qualityDisposition ==
            dse::JointDesignQualityDisposition::ProofNotEstablished &&
        !execution.summary.selectedMapping &&
        execution.summary.qualityIncompleteCandidate == mapping;
    if (!selectedComplete && !searchIncomplete)
      return invalid("application runtime complete observation disagrees with "
                     "its summary");
  }
  if (matching->incompleteReason) {
    if (!matching->objectiveCodes.empty())
      return invalid("application runtime incomplete observation retained an "
                     "objective");
  } else {
    if (matching->provenance.rawMeasures.size() !=
        quality.objectiveDimensionLabels.size())
      return invalid("application runtime projection lost its raw measures");
    if (llvm::Error error = dse::validateJointDesignQualityObjective(
            *quality.objectiveProgram, matching->provenance,
            matching->objectiveCodes))
      return std::move(error);
  }
  std::optional<std::uint64_t> dfgCycles;
  std::optional<std::uint64_t> cgraCycles;
  std::optional<std::uint64_t> resourceCoreCost =
      matching->provenance.resourceCoreCost;
  if (!matching->provenance.rawMeasures.empty()) {
    dfgCycles =
        std::get<ResolvedObjectiveInteger>(matching->provenance.rawMeasures[0])
            .magnitude;
    cgraCycles =
        std::get<ResolvedObjectiveInteger>(matching->provenance.rawMeasures[1])
            .magnitude;
  }
  for (const ArtifactRootReference &reference :
       matching->provenance.supportingEvidence) {
    if (reference.schemaIdentity !=
            evaluation::EvaluationEvidence::artifactSchema.identity ||
        reference.schemaVersion !=
            evaluation::EvaluationEvidence::artifactSchema.version)
      return invalid("application runtime projection has foreign Evidence");
    auto stored = evaluation::importEvaluationEvidenceDependencyProjection(
        reference, artifacts);
    if (!stored)
      return stored.takeError();
    if (matching->provenance.runtimeCompletion ==
            dse::JointDesignQualityRuntimeCompletion::Completed &&
        stored->outcomeKind != evaluation::EvidenceOutcomeKind::Completed)
      return invalid("completed application runtime retained non-completed "
                     "supporting Evidence");
  }
  if (matching->evidence) {
    if (matching->evidence->schemaIdentity !=
            evaluation::EvaluationEvidence::artifactSchema.identity ||
        matching->evidence->schemaVersion !=
            evaluation::EvaluationEvidence::artifactSchema.version)
      return invalid("application runtime projection has foreign primary "
                     "Evidence");
    auto stored = evaluation::importEvaluationEvidenceDependencyProjection(
        *matching->evidence, artifacts);
    if (!stored)
      return stored.takeError();
  }
  for (const ArtifactRootReference &reference :
       matching->provenance.verificationEvidence)
    if (!llvm::is_contained(matching->provenance.supportingEvidence, reference))
      return invalid("application runtime verification Evidence is outside "
                     "the acquired runtime Evidence");
  if (matching->provenance.spatialFifoFeedback &&
      matching->provenance.spatialFifoFeedback->parentMapping != mapping)
    return invalid("application FIFO feedback names a foreign Mapping");
  if (matching->provenance.spatialOperandQueueFeedback &&
      matching->provenance.spatialOperandQueueFeedback->parentMapping &&
      *matching->provenance.spatialOperandQueueFeedback->parentMapping !=
          mapping)
    return invalid("application operand feedback names a foreign Mapping");
  if (matching->provenance.spatialTransportFeedback &&
      matching->provenance.spatialTransportFeedback->parentMapping &&
      *matching->provenance.spatialTransportFeedback->parentMapping != mapping)
    return invalid("application transport feedback names a foreign Mapping");
  return ApplicationRuntimeValidation{
      *runtimeDisposition,
      matching->provenance.supportingEvidence,
      dfgCycles,
      cgraCycles,
      matching->provenance.spatialFifoFeedback,
      matching->provenance.spatialOperandQueueFeedback,
      matching->provenance.spatialTransportFeedback,
      matching->provenance.verificationEvidence,
      resourceCoreCost,
      std::nullopt};
}

llvm::Expected<ApplicationMappingRuntimeDisposition>
detail::classifyApplicationQualityRuntime(
    const dse::JointBoundedQualityPolicy &quality,
    const dse::JointDesignQualityObservation &observation) {
  if (quality.provenanceDomain !=
      dse::JointDesignQualityProvenanceDomain::ApplicationRuntime)
    return invalid("application runtime classification has a foreign quality "
                   "domain");
  if (llvm::Error error = dse::validateJointDesignQualityProvenanceDomain(
          quality, observation.provenance,
          !observation.incompleteReason.has_value()))
    return std::move(error);
  if (observation.provenance.calibratedModelSupport ==
          dse::JointDesignCalibratedModelSupport::OutOfDomain &&
      observation.incompleteReason !=
          dse::JointDesignQualityIncompleteReason::Unsupported)
    return invalid("out-of-domain application quality has a foreign "
                   "incomplete disposition");
  if (observation.provenance.runtimeCompletion ==
      dse::JointDesignQualityRuntimeCompletion::Completed)
    return ApplicationMappingRuntimeDisposition::Completed;
  if (!observation.incompleteReason)
    return invalid("application quality observation has neither completed "
                   "runtime nor an incomplete disposition");
  switch (*observation.incompleteReason) {
  case dse::JointDesignQualityIncompleteReason::Unsupported:
    return ApplicationMappingRuntimeDisposition::Unsupported;
  case dse::JointDesignQualityIncompleteReason::ProofNotEstablished:
    return ApplicationMappingRuntimeDisposition::ProofNotEstablished;
  case dse::JointDesignQualityIncompleteReason::ExecutionFailed:
    return ApplicationMappingRuntimeDisposition::ExecutionFailed;
  case dse::JointDesignQualityIncompleteReason::CancelledOrTimeout:
    return ApplicationMappingRuntimeDisposition::CancelledOrTimeout;
  }
  llvm_unreachable("unknown application quality disposition");
}

llvm::Expected<dse::JointBoundedQualityPolicy>
makeApplicationBoundedQualityPolicy(
    const PreparedApplicationBuild &prepared,
    const dse::PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  std::shared_ptr<const evaluation::models::EdaPredictionModelWeight> fpaWeight;
  if (prepared.edaPredictionModelWeight) {
    auto imported = evaluation::models::importEdaPredictionModelWeight(
        *prepared.edaPredictionModelWeight, artifacts, blobs);
    if (!imported)
      return imported.takeError();
    fpaWeight =
        std::make_shared<const evaluation::models::EdaPredictionModelWeight>(
            std::move(*imported));
  } else if (!prepared.fpaOperatingConditions.empty()) {
    return invalid("prepared FPA conditions have no frozen model weight");
  }

  dse::CandidateMeasureObjectiveCatalogs catalogs;
  const auto integerDimension = [](std::uint32_t ordinal) {
    return dse::CandidateMeasureObjectiveDimension{
        ordinal, ResolvedObjectiveDirection::Minimize, 0,
        std::numeric_limits<std::uint64_t>::max()};
  };
  catalogs.dimensions = {integerDimension(0), integerDimension(1),
                         integerDimension(2)};
  if (fpaWeight)
    for (const auto indexed : llvm::enumerate(applicationFpaMetrics)) {
      auto dimension = makeFpaDimension(
          indexed.value(), static_cast<std::uint32_t>(indexed.index() + 3));
      if (!dimension)
        return dimension.takeError();
      catalogs.dimensions.push_back(std::move(*dimension));
    }
  catalogs.weightedLevels.reserve(catalogs.dimensions.size());
  std::vector<std::uint32_t> finalOrdering;
  finalOrdering.reserve(catalogs.dimensions.size());
  for (std::uint32_t ordinal = 0; ordinal != catalogs.dimensions.size();
       ++ordinal) {
    catalogs.weightedLevels.push_back({{{ordinal, 1}}});
    finalOrdering.push_back(ordinal);
  }
  catalogs.totalOrderings = {{std::move(finalOrdering)}};
  auto program = dse::ObjectiveProgram::getCandidateMeasures(catalogs);
  if (!program)
    return program.takeError();
  auto sharedProgram =
      std::make_shared<const dse::ObjectiveProgram>(std::move(*program));

  dse::JointBoundedQualityPolicy result;
  result.objectiveProgram = sharedProgram;
  result.objectiveDimensionLabels = {"dfg_cycles", "cgra_cycles",
                                     "acc_core_count"};
  result.provenanceDomain =
      dse::JointDesignQualityProvenanceDomain::ApplicationRuntime;
  if (fpaWeight)
    result.objectiveDimensionLabels.insert(
        result.objectiveDimensionLabels.end(),
        {"limiting_clock_frequency", "total_area", "dynamic_power",
         "leakage_power"});
  result.paretoDimensions.resize(catalogs.dimensions.size());
  std::iota(result.paretoDimensions.begin(), result.paretoDimensions.end(), 0);
  result.finalTotalOrdering = 0;
  result.acquire = [&prepared, executionPolicy, &artifacts, &blobs,
                    sharedProgram,
                    fpaWeight](const dse::JointDesignExecution &execution,
                               std::uint64_t planOrdinal)
      -> llvm::Expected<dse::JointDesignQualityAcquisition> {
    if (planOrdinal >= prepared.mappingAlternatives.size())
      return invalid("bounded-quality selected a foreign software plan");
    auto imported = detail::importApplicationMapping(execution, artifacts);
    if (!imported)
      return imported.takeError();
    auto runtime = detail::validateApplicationMappingRuntime(
        prepared, prepared.mappingAlternatives[planOrdinal], execution,
        executionPolicy, artifacts, blobs);
    if (!runtime)
      return runtime.takeError();
    std::vector<ArtifactRootReference> runtimeEvidence = runtime->evidence;
    llvm::sort(runtimeEvidence, artifactRootReferenceLess);
    runtimeEvidence.erase(
        std::unique(runtimeEvidence.begin(), runtimeEvidence.end()),
        runtimeEvidence.end());
    std::vector<ArtifactRootReference> runtimeVerificationEvidence =
        runtime->oracleEvidence;
    llvm::sort(runtimeVerificationEvidence, artifactRootReferenceLess);
    runtimeVerificationEvidence.erase(
        std::unique(runtimeVerificationEvidence.begin(),
                    runtimeVerificationEvidence.end()),
        runtimeVerificationEvidence.end());
    const auto runtimeProvenance =
        [&](std::vector<ResolvedObjectiveScalar> measures = {},
            dse::JointDesignCalibratedModelSupport modelSupport =
                dse::JointDesignCalibratedModelSupport::NotEvaluated) {
          return dse::JointDesignQualityProvenance{
              std::move(measures),
              runtimeEvidence,
              runtimeVerificationEvidence,
              runtime->spatialFifoFeedback,
              runtime->spatialOperandQueueFeedback,
              runtime->spatialTransportFeedback,
              runtime->resourceCoreCost,
              runtime->disposition ==
                      ApplicationMappingRuntimeDisposition::Completed
                  ? dse::JointDesignQualityRuntimeCompletion::Completed
                  : dse::JointDesignQualityRuntimeCompletion::NotEstablished,
              modelSupport};
        };
    switch (runtime->disposition) {
    case ApplicationMappingRuntimeDisposition::Completed:
      break;
    case ApplicationMappingRuntimeDisposition::Unsupported:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::Unsupported,
              execution.summary.selectedMapping, std::nullopt,
              runtimeProvenance()}};
    case ApplicationMappingRuntimeDisposition::CancelledOrTimeout:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::CancelledOrTimeout,
              execution.summary.selectedMapping, std::nullopt,
              runtimeProvenance()}};
    case ApplicationMappingRuntimeDisposition::ExecutionFailed:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::ExecutionFailed,
              execution.summary.selectedMapping, std::nullopt,
              runtimeProvenance()}};
    case ApplicationMappingRuntimeDisposition::ProofNotEstablished:
    case ApplicationMappingRuntimeDisposition::NotRequested:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::ProofNotEstablished,
              execution.summary.selectedMapping, std::nullopt,
              runtimeProvenance()}};
    }
    if (!runtime->dfgCycles || !runtime->cgraCycles)
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::ProofNotEstablished,
              execution.summary.selectedMapping, std::nullopt,
              runtimeProvenance()}};
    std::vector<ResolvedObjectiveScalar> measures = {
        resolvedObjectiveInteger(*runtime->dfgCycles),
        resolvedObjectiveInteger(*runtime->cgraCycles),
        resolvedObjectiveInteger(static_cast<std::uint64_t>(
            imported->system.view().accCoreOccurrences().size()))};
    std::optional<ArtifactRootReference> fpaEvidence;
    dse::JointDesignCalibratedModelSupport modelSupport =
        dse::JointDesignCalibratedModelSupport::NotEvaluated;
    if (fpaWeight) {
      if (dispatchDeadlineReached(executionPolicy))
        return dse::JointDesignQualityAcquisition{
            dse::IncompleteJointDesignQuality{
                dse::JointDesignQualityIncompleteReason::CancelledOrTimeout,
                execution.summary.selectedMapping, std::nullopt,
                runtimeProvenance(measures)}};
      auto fpa = acquireFpaObservation(
          prepared.mappingAlternatives[planOrdinal].dataflow,
          imported->system.reference(), *fpaWeight,
          prepared.fpaOperatingConditions,
          prepared.mappingAlternatives[planOrdinal].plan.resolvedConfig,
          artifacts, blobs);
      if (!fpa)
        return fpa.takeError();
      if (auto *incomplete =
              std::get_if<ApplicationFpaIncompleteObservation>(&*fpa))
        return dse::JointDesignQualityAcquisition{
            dse::IncompleteJointDesignQuality{
                incomplete->reason, execution.summary.selectedMapping,
                incomplete->evidence,
                runtimeProvenance(std::move(measures),
                                  incomplete->modelSupport)}};
      auto completed =
          std::get<ApplicationFpaCompletedObservation>(std::move(*fpa));
      measures.insert(measures.end(), completed.values.begin(),
                      completed.values.end());
      fpaEvidence = std::move(completed.evidence);
      modelSupport = dse::JointDesignCalibratedModelSupport::InDomain;
    }
    dse::ObjectiveVector objective = sharedProgram->makeVector();
    if (llvm::Error error =
            sharedProgram->evaluateCandidateMeasures(measures, objective))
      return std::move(error);
    if (!execution.summary.selectedMapping)
      return invalid("bounded-quality acquisition has no selected mapping");
    return dse::JointDesignQualityAcquisition{
        std::vector<dse::JointDesignQualityCandidate>{
            {{*execution.summary.selectedMapping, std::move(objective)},
             std::move(fpaEvidence),
             runtimeProvenance(std::move(measures), modelSupport)}}};
  };

  if (fpaWeight) {
    dse::CandidateMeasureObjectiveCatalogs promotionCatalogs;
    for (const auto indexed : llvm::enumerate(applicationFpaMetrics)) {
      auto dimension = makeFpaDimension(
          indexed.value(), static_cast<std::uint32_t>(indexed.index()));
      if (!dimension)
        return dimension.takeError();
      promotionCatalogs.dimensions.push_back(std::move(*dimension));
      promotionCatalogs.weightedLevels.push_back(
          {{{static_cast<std::uint32_t>(indexed.index()), 1}}});
    }
    promotionCatalogs.totalOrderings = {{{0, 1, 2, 3}}};
    auto promotionProgram =
        dse::ObjectiveProgram::getCandidateMeasures(promotionCatalogs);
    if (!promotionProgram)
      return promotionProgram.takeError();
    auto sharedPromotionProgram = std::make_shared<const dse::ObjectiveProgram>(
        std::move(*promotionProgram));
    result.hardwarePromotion = dse::JointHardwarePromotionQualityPolicy{
        sharedPromotionProgram,
        {"limiting_clock_frequency", "total_area", "dynamic_power",
         "leakage_power"},
        0,
        [&prepared, executionPolicy, &artifacts, &blobs, fpaWeight,
         sharedPromotionProgram](const dse::JointDesignExplorationPlan &plan,
                                 std::uint64_t)
            -> llvm::Expected<dse::JointDesignQualityAcquisition> {
          if (plan.frontier.softwareFrontier.size() != 1 ||
              plan.frontier.systemFrontier.size() != 1)
            return invalid("FPA promotion requires one exact Dataflow/System "
                           "pair");
          const ArtifactRootReference &system =
              plan.frontier.systemFrontier.front();
          if (dispatchDeadlineReached(executionPolicy))
            return dse::JointDesignQualityAcquisition{
                dse::IncompleteJointDesignQuality{
                    dse::JointDesignQualityIncompleteReason::CancelledOrTimeout,
                    system, std::nullopt}};
          auto fpa = acquireFpaObservation(
              plan.frontier.softwareFrontier.front().dataflow, system,
              *fpaWeight, prepared.fpaOperatingConditions, plan.resolvedConfig,
              artifacts, blobs);
          if (!fpa)
            return fpa.takeError();
          if (auto *incomplete =
                  std::get_if<ApplicationFpaIncompleteObservation>(&*fpa))
            return dse::JointDesignQualityAcquisition{
                dse::IncompleteJointDesignQuality{
                    incomplete->reason, system, incomplete->evidence,
                    dse::JointDesignQualityProvenance{
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        dse::JointDesignQualityRuntimeCompletion::
                            NotEstablished,
                        incomplete->modelSupport}}};
          auto completed =
              std::get<ApplicationFpaCompletedObservation>(std::move(*fpa));
          dse::ObjectiveVector objective = sharedPromotionProgram->makeVector();
          if (llvm::Error error =
                  sharedPromotionProgram->evaluateCandidateMeasures(
                      completed.values, objective))
            return std::move(error);
          return dse::JointDesignQualityAcquisition{
              std::vector<dse::JointDesignQualityCandidate>{
                  {{system, std::move(objective)},
                   completed.evidence,
                   {std::vector<ResolvedObjectiveScalar>(
                        completed.values.begin(), completed.values.end()),
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    dse::JointDesignQualityRuntimeCompletion::NotEstablished,
                    dse::JointDesignCalibratedModelSupport::InDomain}}}};
        }};

    result.semanticInputs.push_back(fpaWeight->reference());
    for (const PreparedApplicationMappingAlternative &alternative :
         prepared.mappingAlternatives) {
      if (alternative.plan.frontier.systemFrontier.size() != 1)
        return invalid("FPA ranking input has no exact System");
      auto baseRequest = evaluation::models::
          prepareCanonicalDataflowFabricCalibratedFpaEvaluation(
              alternative.dataflow,
              alternative.plan.frontier.systemFrontier.front(), *fpaWeight,
              prepared.fpaOperatingConditions, alternative.plan.resolvedConfig,
              artifacts, blobs);
      if (!baseRequest)
        return baseRequest.takeError();
      result.semanticInputs.push_back(
          evaluation::evaluationRequestReference(baseRequest->request));
    }
    llvm::sort(result.semanticInputs, artifactRootReferenceLess);
    result.semanticInputs.erase(
        std::unique(result.semanticInputs.begin(), result.semanticInputs.end()),
        result.semanticInputs.end());
  }
  return result;
}

} // namespace loom::application
