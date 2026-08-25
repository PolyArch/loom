#include "Config/ResolvedConfig.h"
#include "DSE/GroundTruthPlan.h"
#include "DSE/ModelParameterCalibrationAcquisition.h"
#include "Evaluation/Evidence.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <variant>

namespace {

using namespace loom;
using namespace loom::dse;
using namespace loom::evaluation;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "ground-truth campaign integration test failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

ArtifactRootReference evidence(std::uint8_t fill) {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes{};
  bytes.fill(fill);
  return {EvaluationEvidence::artifactSchema.identity.str(),
          EvaluationEvidence::artifactSchema.version,
          take(ArtifactIdentity::fromBytes(bytes))};
}

GroundTruthModelTrack track(std::uint8_t base, std::uint64_t seed) {
  return {{{evidence(base), evidence(static_cast<std::uint8_t>(base + 1))},
           {evidence(static_cast<std::uint8_t>(base + 2))},
           {evidence(static_cast<std::uint8_t>(base + 3))},
           std::nullopt},
          {seed, 3, 2, 1, 1, 2},
          take(DecimalValue::get(2, -1)),
          take(DecimalValue::get(3, -1))};
}

void exactDualTrackPlanIsDeterministic() {
  GroundTruthPlanInputs inputs;
  inputs.fpa = track(0x10, 7);
  inputs.systemRuntime = track(0x20, 11);
  ResolvedGroundTruthPlan first =
      take(buildGroundTruthPlan(defaultResolvedConfig(), inputs));

  std::swap(inputs.fpa->evidence.training.front(),
            inputs.fpa->evidence.training.back());
  ResolvedGroundTruthPlan repeated =
      take(buildGroundTruthPlan(defaultResolvedConfig(), std::move(inputs)));
  require(resolvedConfigIdentity(first.resolvedConfig()) ==
              resolvedConfigIdentity(repeated.resolvedConfig()),
          "input order changed the canonical campaign plan");
  require(first.view().plan().nodes().size() == 6 &&
              first.preexistingEvidence().size() == 8,
          "dual-track campaign did not retain its finite evidence closure");
  const ExactRatio median = take(ExactRatio::get(1, 2));
  const ExactRatio p90 = take(ExactRatio::get(9, 10));
  std::array<std::size_t, 2> metricCounts{};
  for (const EvidenceObligationTemplate &obligation :
       first.resolvedConfig().dse.evidenceObligationTemplates) {
    const std::size_t count = obligation.metricRequests().size();
    if (count == 8)
      ++metricCounts[0];
    else if (count == 2)
      ++metricCounts[1];
    else
      fail("calibration obligation lost its median/P90 metric shape");
    std::array<std::size_t, 2> quantileCounts{};
    for (const MetricRequestTemplate &metric : obligation.metricRequests()) {
      if (metric.conditions.size() != 1 ||
          metric.conditions.front().kind() != EvaluationConditionKind::Quantile)
        fail("calibration metric lost its exact quantile condition");
      const ExactRatio quantile =
          std::get<QuantileCondition>(metric.conditions.front().payload)
              .probability;
      if (quantile == median)
        ++quantileCounts[0];
      else if (quantile == p90)
        ++quantileCounts[1];
      else
        fail("calibration metric admitted a non-canonical quantile");
    }
    if (quantileCounts[0] != count / 2 || quantileCounts[1] != count / 2)
      fail("calibration obligation duplicated or omitted an error quantile");
  }
  require(metricCounts == std::array<std::size_t, 2>{2, 2},
          "dual-track campaign did not retain both error quantiles");

  const std::array outputs = {first.fpaOutputs(), first.systemRuntimeOutputs()};
  for (const auto &trackOutputs : outputs) {
    require(trackOutputs.has_value(), "campaign omitted a requested track");
    const ResolvedDsePlan &plan = first.view().plan();
    const auto &validation = std::get<ResolvedPromotePlanNode>(
        plan.nodes()[trackOutputs->validationEvidence.producerNodeOrdinal]);
    const auto &heldOut = std::get<ResolvedPromotePlanNode>(
        plan.nodes()[trackOutputs->heldOutEvidence.producerNodeOrdinal]);
    require(validation.purpose() == PromotePurpose::CandidateSelection &&
                heldOut.purpose() == PromotePurpose::ModelRelease,
            "calibration partitions acquired the wrong promotion purpose");
    require(std::get<PlanOutputRef>(heldOut.inputBindings().front()) ==
                PlanOutputRef{
                    trackOutputs->validationEvidence.producerNodeOrdinal, 0},
            "held-out release did not consume the validation survivor");
    const PlanValueDescriptor *released =
        plan.resolve(trackOutputs->releasedBundle);
    const PlanValueDescriptor *heldOutEvidence =
        plan.resolve(trackOutputs->heldOutEvidence);
    require(released && heldOutEvidence &&
                released->role == PlanValueRole::CandidateSet &&
                heldOutEvidence->role == PlanValueRole::EvidenceSet &&
                heldOutEvidence->calibrationPartitionRole ==
                    CalibrationPartitionRole::HeldOut,
            "terminal release outputs lost their typed partition contract");
  }
}

void heldOutCannotBecomeCandidateSelection() {
  GroundTruthPlanInputs inputs;
  inputs.fpa = track(0x30, 13);
  ResolvedGroundTruthPlan resolved =
      take(buildGroundTruthPlan(defaultResolvedConfig(), std::move(inputs)));
  ResolvedConfig invalid = resolved.resolvedConfig();
  auto &heldOut =
      std::get<PromotePlanNodeDefinition>(invalid.dse.planNodes.back());
  heldOut.purpose = PromotePurpose::CandidateSelection;
  auto rejected = projectResolvedDseConfigView(invalid);
  require(!rejected, "held-out Evidence entered candidate selection");
  const std::string message = llvm::toString(rejected.takeError());
  require(message.find("held-out Evidence") != std::string::npos,
          "held-out rejection did not identify the violated invariant");
}

} // namespace

int main() {
  exactDualTrackPlanIsDeterministic();
  heldOutCannotBecomeCandidateSelection();
  return EXIT_SUCCESS;
}
