#include "Evaluation/Case.h"

#include "CanonicalSupport.h"

#include "ImplementationPlatform/TechnologyCorner.h"

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
using detail::appendFramedBytes;
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
  if (std::holds_alternative<ExecutionActivitySource>(activity.source))
    return evaluationError(
        "SimulationExecution activity validation is unavailable");
  const auto *assumption =
      std::get_if<ExplicitAssumptionSource>(&activity.source);
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

/// A process corner requires the TechnologyCorner owner codec and exact
/// platform import. The platform must also be admitted by the selected
/// target's anchor closure.
llvm::Error validateProcessCorner(const ProcessCornerCondition &corner,
                                  const CaseTargetContext &context) {
  const EncodedArtifactLocalReference encoded =
      platform::encodeTechnologyCornerRef(corner.corner);
  const CaseArtifactResolution::Entry *platformEntry =
      context.resolution().find(encoded.artifact);
  if (!platformEntry)
    return evaluationError("the implementation platform artifact of a process "
                           "corner is unresolved");
  if (llvm::Error error =
          validateArtifactLocalReference(context.artifactStore(), encoded))
    return error;

  const CaseArtifactResolution::Entry *anchor =
      context.resolution().find(corner.target.anchorSubjectArtifact);
  if (!anchor || !CaseArtifactResolution::reaches(*anchor, encoded.artifact))
    return evaluationError(
        "the exact implementation platform is not admitted by the selected "
        "subject's dependency closure");
  return llvm::Error::success();
}

void appendActivitySource(std::vector<std::uint8_t> &key,
                          const ActivitySource &source) {
  if (const auto *execution = std::get_if<ExecutionActivitySource>(&source)) {
    appendU32Be(key, executionActivityDiscriminator);
    appendFramedBytes(
        key, encodeArtifactRootReference(execution->simulationExecution));
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

std::vector<const SubjectTargetRef *>
conditionOrderedTargets(const EvaluationCondition &condition) {
  std::vector<const SubjectTargetRef *> targets;
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

ConditionApplicabilityPattern
deriveConditionApplicabilityPattern(const EvaluationCondition &condition,
                                    EvaluationCaseSignatureRef caseSignature) {
  std::vector<const SubjectTargetRef *> targets =
      conditionOrderedTargets(condition);
  llvm::SmallVector<SubjectTargetRef, 2> owned;
  owned.reserve(targets.size());
  for (const SubjectTargetRef *target : targets)
    owned.push_back(*target);
  return ConditionApplicabilityPattern{
      condition.kind(), deriveOrderedTargetPattern(owned, caseSignature)};
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
    appendSubjectTargetKey(key, *conditionOrderedTargets(condition)[0]);
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
    appendFramedBytes(key,
                      encodeArtifactLocalReference(
                          platform::encodeTechnologyCornerRef(corner.corner)));
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
canonicalizeEvaluationConditions(
    llvm::ArrayRef<EvaluationCondition> conditions, ConditionLocation location,
    llvm::StringRef permittingOwner,
    llvm::ArrayRef<ConditionApplicabilityPattern> permittedPatterns,
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
    if (!descriptor.permitsLocation(location))
      return evaluationError("condition '" + descriptor.spelling +
                             "' is not permitted in " + toString(location) +
                             " conditions");
    if (llvm::Error error = validateConditionPayload(condition))
      return std::move(error);
    if (condition.kind() == EvaluationConditionKind::ProcessCorner)
      if (llvm::Error error = validateProcessCorner(
              std::get<ProcessCornerCondition>(condition.payload), context))
        return std::move(error);
    for (const SubjectTargetRef *target : conditionOrderedTargets(condition))
      if (llvm::Error error = validateSubjectTargetRef(*target, context))
        return std::move(error);

    // Owner validation ran first; now the derived exact pattern must match
    // one complete pattern of the permitting owner.
    const ConditionApplicabilityPattern derived =
        deriveConditionApplicabilityPattern(condition, context.signatureRef());
    bool permitted = false;
    for (const ConditionApplicabilityPattern &pattern : permittedPatterns)
      permitted = permitted || pattern == derived;
    if (!permitted)
      return evaluationError("condition '" + descriptor.spelling +
                             "' is not applicable to '" + permittingOwner +
                             "'");

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
