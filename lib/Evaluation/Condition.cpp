#include "Evaluation/Case.h"

#include "CanonicalSupport.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>

namespace loom::evaluation {
namespace {

using detail::appendDecimalValue;
using detail::appendExactRatio;
using detail::appendSubjectTargetKey;
using detail::appendU32Be;
using detail::appendU64Be;
using detail::evaluationError;

constexpr std::uint8_t locationBit(ConditionLocation location) {
  return static_cast<std::uint8_t>(1u << static_cast<std::uint8_t>(location));
}

constexpr std::uint8_t baseOnly = locationBit(ConditionLocation::Base);
constexpr std::uint8_t metricRequestOnly =
    locationBit(ConditionLocation::MetricRequest);

constexpr std::uint32_t executionActivityDiscriminator = 0;
constexpr std::uint32_t explicitAssumptionDiscriminator = 1;

const std::array<EvaluationConditionDescriptor, 7> conditionDescriptors = {{
    {EvaluationConditionKind::ProcessCorner, "process_corner",
     "The exact technology corner applied to one target.", baseOnly},
    {EvaluationConditionKind::SupplyVoltage, "supply_voltage",
     "The supply voltage applied to one power domain.", baseOnly},
    {EvaluationConditionKind::Temperature, "temperature",
     "The temperature applied to one thermal domain or root.", baseOnly},
    {EvaluationConditionKind::RequiredClockPeriod, "required_clock_period",
     "The required period of one clock domain.", baseOnly},
    {EvaluationConditionKind::RelativeClockSchedule, "relative_clock_schedule",
     "The exact schedule of one dependent clock in reference cycles.",
     baseOnly},
    {EvaluationConditionKind::ActivityBinding, "activity_binding",
     "The switching activity projected onto one target.", baseOnly},
    {EvaluationConditionKind::Quantile, "quantile",
     "The nearest-rank sample quantile selected for one metric request.",
     metricRequestOnly},
}};

bool isProbability(ExactRatio value) {
  return value.numerator() <= value.denominator();
}

llvm::Error validatePositiveDecimal(DecimalValue value, llvm::StringRef what) {
  if (value.coefficient() <= 0)
    return evaluationError(what + " must be positive");
  return llvm::Error::success();
}

llvm::Error
validateRelativeClockSchedule(const RelativeClockScheduleCondition &schedule) {
  if (schedule.referenceClock == schedule.dependentClock)
    return evaluationError("relative clock schedule requires distinct clock "
                           "domains");
  if (schedule.dependentPeriodPerReferencePeriod.isZero())
    return evaluationError("relative clock period ratio must be positive");

  llvm::Expected<ExactRatio> normalized =
      schedule.dependentPhaseInReferenceCycles.reducedModulo(
          schedule.dependentPeriodPerReferencePeriod);
  if (!normalized)
    return normalized.takeError();
  if (*normalized != schedule.dependentPhaseInReferenceCycles)
    return evaluationError("dependent phase must be normalized into "
                           "[0, dependent period)");
  return llvm::Error::success();
}

llvm::Error validateActivityBinding(const ActivityBindingCondition &activity) {
  const auto *assumption =
      std::get_if<ExplicitAssumptionSource>(&activity.source);
  if (!assumption)
    return llvm::Error::success();
  if (!isProbability(assumption->staticProbability))
    return evaluationError("static probability must be in [0, 1]");
  return llvm::Error::success();
}

llvm::Error validateConditionPayload(const EvaluationCondition &condition) {
  switch (condition.kind()) {
  case EvaluationConditionKind::ProcessCorner:
    return llvm::Error::success();
  case EvaluationConditionKind::SupplyVoltage:
    return validatePositiveDecimal(
        std::get<SupplyVoltageCondition>(condition.payload).volts,
        "supply voltage");
  case EvaluationConditionKind::Temperature:
    return validatePositiveDecimal(
        std::get<TemperatureCondition>(condition.payload).kelvin,
        "temperature");
  case EvaluationConditionKind::RequiredClockPeriod:
    return validatePositiveDecimal(
        std::get<RequiredClockPeriodCondition>(condition.payload).seconds,
        "required clock period");
  case EvaluationConditionKind::RelativeClockSchedule:
    return validateRelativeClockSchedule(
        std::get<RelativeClockScheduleCondition>(condition.payload));
  case EvaluationConditionKind::ActivityBinding:
    return validateActivityBinding(
        std::get<ActivityBindingCondition>(condition.payload));
  case EvaluationConditionKind::Quantile:
    if (!isProbability(
            std::get<QuantileCondition>(condition.payload).probability))
      return evaluationError("quantile probability must be in [0, 1]");
    return llvm::Error::success();
  }
  llvm_unreachable("unknown EvaluationConditionKind");
}

/// Every SubjectTargetRef a condition payload carries, in canonical payload
/// order. A condition target obeys exactly the same case-bound rules as a
/// scope target.
llvm::SmallVector<const SubjectTargetRef *, 2>
conditionTargets(const EvaluationCondition &condition) {
  llvm::SmallVector<const SubjectTargetRef *, 2> targets;
  switch (condition.kind()) {
  case EvaluationConditionKind::ProcessCorner:
    targets.push_back(
        &std::get<ProcessCornerCondition>(condition.payload).target);
    break;
  case EvaluationConditionKind::SupplyVoltage:
    targets.push_back(
        &std::get<SupplyVoltageCondition>(condition.payload).powerDomain);
    break;
  case EvaluationConditionKind::Temperature:
    targets.push_back(
        &std::get<TemperatureCondition>(condition.payload).thermalDomainOrRoot);
    break;
  case EvaluationConditionKind::RequiredClockPeriod:
    targets.push_back(
        &std::get<RequiredClockPeriodCondition>(condition.payload).clockDomain);
    break;
  case EvaluationConditionKind::RelativeClockSchedule: {
    const auto &schedule =
        std::get<RelativeClockScheduleCondition>(condition.payload);
    targets.push_back(&schedule.referenceClock);
    targets.push_back(&schedule.dependentClock);
    break;
  }
  case EvaluationConditionKind::ActivityBinding: {
    const auto &activity =
        std::get<ActivityBindingCondition>(condition.payload);
    targets.push_back(&activity.target);
    if (const auto *assumption =
            std::get_if<ExplicitAssumptionSource>(&activity.source))
      targets.push_back(&assumption->clockDomain);
    break;
  }
  case EvaluationConditionKind::Quantile:
    break;
  }
  return targets;
}

void appendActivitySource(std::vector<std::uint8_t> &key,
                          const ActivitySource &source) {
  if (const auto *execution = std::get_if<ExecutionActivitySource>(&source)) {
    appendU32Be(key, executionActivityDiscriminator);
    detail::appendArtifactIdentity(key, execution->simulationExecution);
    appendU64Be(key, execution->activitySummaryOrdinal);
    return;
  }
  const auto &assumption = std::get<ExplicitAssumptionSource>(source);
  appendU32Be(key, explicitAssumptionDiscriminator);
  appendSubjectTargetKey(key, assumption.clockDomain);
  appendExactRatio(key, assumption.staticProbability);
  appendExactRatio(key, assumption.transitionsPerClock);
}

} // namespace

EvaluationConditionKind EvaluationCondition::kind() const {
  return static_cast<EvaluationConditionKind>(payload.index());
}

bool EvaluationConditionDescriptor::permitsLocation(
    ConditionLocation location) const {
  return (allowedLocations & locationBit(location)) != 0;
}

const EvaluationConditionDescriptor &
conditionDescriptor(EvaluationConditionKind kind) {
  for (const EvaluationConditionDescriptor &descriptor : conditionDescriptors)
    if (descriptor.kind == kind)
      return descriptor;
  llvm_unreachable("unknown EvaluationConditionKind");
}

llvm::StringRef toString(EvaluationConditionKind kind) {
  return conditionDescriptor(kind).spelling;
}

llvm::StringRef toString(ConditionLocation location) {
  switch (location) {
  case ConditionLocation::Base:
    return "base";
  case ConditionLocation::MetricRequest:
    return "metric-request";
  case ConditionLocation::FindingRequest:
    return "finding-request";
  }
  llvm_unreachable("unknown ConditionLocation");
}

std::vector<std::uint8_t>
conditionAssignmentKey(const EvaluationCondition &condition) {
  std::vector<std::uint8_t> key;
  switch (condition.kind()) {
  case EvaluationConditionKind::ProcessCorner:
  case EvaluationConditionKind::SupplyVoltage:
  case EvaluationConditionKind::Temperature:
  case EvaluationConditionKind::RequiredClockPeriod:
  case EvaluationConditionKind::ActivityBinding:
    appendSubjectTargetKey(key, *conditionTargets(condition)[0]);
    return key;
  case EvaluationConditionKind::RelativeClockSchedule: {
    const auto &schedule =
        std::get<RelativeClockScheduleCondition>(condition.payload);
    appendSubjectTargetKey(key, schedule.referenceClock);
    appendSubjectTargetKey(key, schedule.dependentClock);
    return key;
  }
  case EvaluationConditionKind::Quantile:
    return key;
  }
  llvm_unreachable("unknown EvaluationConditionKind");
}

std::vector<std::uint8_t>
conditionPayloadKey(const EvaluationCondition &condition) {
  std::vector<std::uint8_t> key;
  appendU32Be(key, static_cast<std::uint32_t>(condition.kind()));
  switch (condition.kind()) {
  case EvaluationConditionKind::ProcessCorner: {
    const auto &corner = std::get<ProcessCornerCondition>(condition.payload);
    appendSubjectTargetKey(key, corner.target);
    detail::appendArtifactIdentity(key, corner.corner.provider());
    appendU64Be(key, corner.corner.corner());
    return key;
  }
  case EvaluationConditionKind::SupplyVoltage: {
    const auto &voltage = std::get<SupplyVoltageCondition>(condition.payload);
    appendSubjectTargetKey(key, voltage.powerDomain);
    appendDecimalValue(key, voltage.volts);
    return key;
  }
  case EvaluationConditionKind::Temperature: {
    const auto &temperature = std::get<TemperatureCondition>(condition.payload);
    appendSubjectTargetKey(key, temperature.thermalDomainOrRoot);
    appendDecimalValue(key, temperature.kelvin);
    return key;
  }
  case EvaluationConditionKind::RequiredClockPeriod: {
    const auto &period =
        std::get<RequiredClockPeriodCondition>(condition.payload);
    appendSubjectTargetKey(key, period.clockDomain);
    appendDecimalValue(key, period.seconds);
    return key;
  }
  case EvaluationConditionKind::RelativeClockSchedule: {
    const auto &schedule =
        std::get<RelativeClockScheduleCondition>(condition.payload);
    appendSubjectTargetKey(key, schedule.referenceClock);
    appendSubjectTargetKey(key, schedule.dependentClock);
    appendExactRatio(key, schedule.dependentPeriodPerReferencePeriod);
    appendExactRatio(key, schedule.dependentPhaseInReferenceCycles);
    return key;
  }
  case EvaluationConditionKind::ActivityBinding: {
    const auto &activity =
        std::get<ActivityBindingCondition>(condition.payload);
    appendSubjectTargetKey(key, activity.target);
    appendActivitySource(key, activity.source);
    return key;
  }
  case EvaluationConditionKind::Quantile:
    appendExactRatio(
        key, std::get<QuantileCondition>(condition.payload).probability);
    return key;
  }
  llvm_unreachable("unknown EvaluationConditionKind");
}

llvm::Expected<std::vector<EvaluationCondition>>
canonicalizeEvaluationConditions(llvm::ArrayRef<EvaluationCondition> conditions,
                                 const ConditionApplicability &applicability,
                                 const CaseTargetContext &context) {
  struct KeyedCondition {
    EvaluationCondition condition;
    std::vector<std::uint8_t> assignmentKey;
    std::vector<std::uint8_t> payloadKey;
  };

  std::vector<KeyedCondition> keyed;
  keyed.reserve(conditions.size());
  for (const EvaluationCondition &condition : conditions) {
    const EvaluationConditionDescriptor &descriptor =
        conditionDescriptor(condition.kind());
    if (!descriptor.permitsLocation(applicability.location))
      return evaluationError("condition '" + descriptor.spelling +
                             "' is not permitted in " +
                             toString(applicability.location) + " conditions");
    const ConditionPattern *pattern =
        applicability.findPattern(condition.kind());
    if (!pattern)
      return evaluationError("condition '" + descriptor.spelling +
                             "' is not applicable to '" +
                             applicability.permittingOwner + "'");
    if (llvm::Error error = validateConditionPayload(condition))
      return std::move(error);
    for (const SubjectTargetRef *target : conditionTargets(condition)) {
      if (std::find(pattern->permittedTargetRoles.begin(),
                    pattern->permittedTargetRoles.end(),
                    target->caseSubjectRole) ==
          pattern->permittedTargetRoles.end())
        return evaluationError(
            "'" + applicability.permittingOwner +
            "' does not permit case subject role " +
            std::to_string(target->caseSubjectRole.ordinal()) +
            " as a target of condition '" + descriptor.spelling + "'");
      if (llvm::Error error = validateSubjectTargetRef(*target, context))
        return std::move(error);
    }
    keyed.push_back({condition, conditionAssignmentKey(condition),
                     conditionPayloadKey(condition)});
  }

  std::sort(keyed.begin(), keyed.end(),
            [](const KeyedCondition &lhs, const KeyedCondition &rhs) {
              if (lhs.condition.kind() != rhs.condition.kind())
                return lhs.condition.kind() < rhs.condition.kind();
              if (lhs.assignmentKey != rhs.assignmentKey)
                return lhs.assignmentKey < rhs.assignmentKey;
              return lhs.payloadKey < rhs.payloadKey;
            });

  std::vector<EvaluationCondition> canonical;
  canonical.reserve(keyed.size());
  for (std::size_t index = 0; index < keyed.size(); ++index) {
    if (index != 0 &&
        keyed[index - 1].condition.kind() == keyed[index].condition.kind() &&
        keyed[index - 1].assignmentKey == keyed[index].assignmentKey) {
      const llvm::StringRef spelling =
          conditionDescriptor(keyed[index].condition.kind()).spelling;
      if (keyed[index - 1].payloadKey == keyed[index].payloadKey)
        return evaluationError("duplicate evaluation condition '" + spelling +
                               "'");
      return evaluationError("conflicting '" + spelling +
                             "' conditions for one assignment key");
    }
    canonical.push_back(keyed[index].condition);
  }
  return canonical;
}

} // namespace loom::evaluation
