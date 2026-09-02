#include "Evaluation/Models/FabricLowConfidence.h"

#include "AnalyticModelSupport.h"

#include "Common/ArtifactStore.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Artifact/FabricArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr CaseSubjectRoleRef kFabricRole(0);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("fabric_low_confidence_invalid: ") + message);
}

llvm::Expected<std::uint64_t> scaledRatio(ExactRatio ratio) {
  const unsigned __int128 scaled =
      static_cast<unsigned __int128>(ratio.numerator()) * 1024;
  const unsigned __int128 rounded =
      (scaled + ratio.denominator() - 1) / ratio.denominator();
  if (rounded > std::numeric_limits<std::uint64_t>::max())
    return invalid("activity ratio exceeds the model domain");
  return static_cast<std::uint64_t>(rounded);
}

llvm::Expected<std::optional<std::uint64_t>>
projectExplicitActivity(const EvaluationRequest &request) {
  std::uint64_t total = 0;
  bool found = false;
  for (const EvaluationCondition &condition : request.baseConditions()) {
    if (condition.kind() != EvaluationConditionKind::ActivityBinding)
      continue;
    const auto &binding = std::get<ActivityBindingCondition>(condition.payload);
    const auto *assumption =
        std::get_if<ExplicitAssumptionSource>(&binding.source);
    if (!assumption)
      return std::optional<std::uint64_t>{};
    auto probability = scaledRatio(assumption->staticProbability);
    if (!probability)
      return probability.takeError();
    auto transitions = scaledRatio(assumption->transitionsPerClock);
    if (!transitions)
      return transitions.takeError();
    const unsigned __int128 sum =
        static_cast<unsigned __int128>(total) + *probability + *transitions;
    if (sum > std::numeric_limits<std::uint64_t>::max())
      return invalid("activity projection overflowed");
    total = static_cast<std::uint64_t>(sum);
    found = true;
  }
  if (!found)
    return std::optional<std::uint64_t>{};
  return std::optional<std::uint64_t>{std::max<std::uint64_t>(total, 1)};
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request, const CaseArtifactResolution &,
         const ArtifactStore &artifactStore, const BlobStore &) {
  const auto fabrics = request.subjectBindings().subjects(kFabricRole);
  if (fabrics.size() != 1)
    return invalid("case does not bind exactly one Fabric");
  auto fabric = fabric::importEntireFabricRoot(fabrics.front(), artifactStore);
  if (!fabric)
    return fabric.takeError();

  const bool requestsDynamic =
      llvm::any_of(request.metricRequests(), [](const MetricRequest &metric) {
        return metric.query().metric == MetricKind::DynamicPower;
      });
  auto activity = projectExplicitActivity(request);
  if (!activity)
    return activity.takeError();
  if (requestsDynamic && !*activity)
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  auto metrics = detail::estimateLowConfidenceFabricMetrics(
      *fabric, activity->value_or(0));
  if (!metrics)
    return metrics.takeError();
  std::vector<MetricResult> results;
  results.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    auto result = metrics->result(metric.query().metric);
    if (!result)
      return result.takeError();
    results.push_back(std::move(*result));
  }
  return EvaluationModelResult{{}, CompletedEvidence{std::move(results), {}}};
}

const EvaluationModelProvider kProvider{
    fabricLowConfidenceModelDescriptorRef(),
    EvaluationModelInProcessProvider{&evaluate}};

} // namespace

EvaluationModelDescriptorRef fabricLowConfidenceModelDescriptorRef() {
  return llvm::cantFail(builtinEvaluationModelDescriptorRef(
      BuiltinEvaluationModel::FabricLowConfidence));
}

llvm::Error registerFabricLowConfidenceProvider() {
  return registerEvaluationModelProvider(kProvider);
}

llvm::Expected<std::uint64_t> fabricLowConfidenceClockPeriodPicoseconds(
    const fabric::FinalizedFabricRoot &fabricRoot) {
  return detail::lowConfidenceClockPeriodPicoseconds(fabricRoot);
}

} // namespace loom::evaluation::models
