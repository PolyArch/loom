#include "Application/Build.h"
#include "ApplicationRuntimeValidationInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
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
  auto prepared =
      evaluation::models::prepareCanonicalDataflowFabricCalibratedFpaEvaluation(
          dataflow, system, weight, operatingConditions, config, artifacts,
          blobs);
  if (!prepared)
    return prepared.takeError();
  auto evaluated = evaluation::evaluateRequest(
      prepared->request, prepared->resolution, artifacts, blobs);
  if (!evaluated)
    return evaluated.takeError();
  auto evidence = evaluation::publishEvaluationEvidence(*evaluated, artifacts);
  if (!evidence)
    return evidence.takeError();

  if (std::holds_alternative<evaluation::UnsupportedEvidence>(
          evaluated->outcome()))
    return ApplicationFpaIncompleteObservation{
        dse::JointDesignQualityIncompleteReason::Unsupported,
        std::move(*evidence)};
  if (std::holds_alternative<evaluation::ExecutionFailedEvidence>(
          evaluated->outcome()))
    return ApplicationFpaIncompleteObservation{
        dse::JointDesignQualityIncompleteReason::ExecutionFailed,
        std::move(*evidence)};
  if (std::holds_alternative<evaluation::CancelledOrTimeoutEvidence>(
          evaluated->outcome()))
    return ApplicationFpaIncompleteObservation{
        dse::JointDesignQualityIncompleteReason::CancelledOrTimeout,
        std::move(*evidence)};

  const auto *completed =
      std::get_if<evaluation::CompletedEvidence>(&evaluated->outcome());
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
          std::move(*evidence)};
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
    switch (runtime->disposition) {
    case ApplicationMappingRuntimeDisposition::Completed:
      break;
    case ApplicationMappingRuntimeDisposition::Unsupported:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::Unsupported,
              execution.summary.selectedMapping, std::nullopt}};
    case ApplicationMappingRuntimeDisposition::CancelledOrTimeout:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::CancelledOrTimeout,
              execution.summary.selectedMapping, std::nullopt}};
    case ApplicationMappingRuntimeDisposition::ExecutionFailed:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::ExecutionFailed,
              execution.summary.selectedMapping, std::nullopt}};
    case ApplicationMappingRuntimeDisposition::ProofNotEstablished:
    case ApplicationMappingRuntimeDisposition::NotRequested:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::ProofNotEstablished,
              execution.summary.selectedMapping, std::nullopt}};
    }
    if (!runtime->dfgCycles || !runtime->cgraCycles)
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::ProofNotEstablished,
              execution.summary.selectedMapping, std::nullopt}};
    std::vector<ResolvedObjectiveScalar> measures = {
        resolvedObjectiveInteger(*runtime->dfgCycles),
        resolvedObjectiveInteger(*runtime->cgraCycles),
        resolvedObjectiveInteger(static_cast<std::uint64_t>(
            imported->system.view().accCoreOccurrences().size()))};
    std::optional<ArtifactRootReference> fpaEvidence;
    if (fpaWeight) {
      if (dispatchDeadlineReached(executionPolicy))
        return dse::JointDesignQualityAcquisition{
            dse::IncompleteJointDesignQuality{
                dse::JointDesignQualityIncompleteReason::CancelledOrTimeout,
                execution.summary.selectedMapping, std::nullopt}};
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
            dse::IncompleteJointDesignQuality{incomplete->reason,
                                              execution.summary.selectedMapping,
                                              incomplete->evidence}};
      auto completed =
          std::get<ApplicationFpaCompletedObservation>(std::move(*fpa));
      measures.insert(measures.end(), completed.values.begin(),
                      completed.values.end());
      fpaEvidence = std::move(completed.evidence);
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
             std::move(fpaEvidence)}}};
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
                dse::IncompleteJointDesignQuality{incomplete->reason, system,
                                                  incomplete->evidence}};
          auto completed =
              std::get<ApplicationFpaCompletedObservation>(std::move(*fpa));
          dse::ObjectiveVector objective = sharedPromotionProgram->makeVector();
          if (llvm::Error error =
                  sharedPromotionProgram->evaluateCandidateMeasures(
                      completed.values, objective))
            return std::move(error);
          return dse::JointDesignQualityAcquisition{
              std::vector<dse::JointDesignQualityCandidate>{
                  {{system, std::move(objective)}, completed.evidence}}};
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
