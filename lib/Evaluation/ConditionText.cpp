#include "Evaluation/ConditionText.h"

#include "CanonicalSupport.h"
#include "Evaluation/CaseText.h"

#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <utility>

namespace loom::evaluation {
namespace {

using detail::evaluationError;
using detail::rejectUnknownFields;
using detail::requireInteger;
using detail::requireObject;
using detail::requireString;
using detail::requireUnsigned;

void writeDecimal(llvm::json::OStream &json, DecimalValue value) {
  json.object([&] {
    json.attribute("coefficient", value.coefficient());
    json.attribute("base10_exponent", value.base10Exponent());
  });
}

llvm::Expected<DecimalValue> parseDecimal(const llvm::json::Object &object,
                                          llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"coefficient", "base10_exponent"}))
    return std::move(error);
  auto coefficient = requireInteger(object, "coefficient", context);
  if (!coefficient)
    return coefficient.takeError();
  auto exponent = requireInteger(object, "base10_exponent", context);
  if (!exponent)
    return exponent.takeError();
  return DecimalValue::get(*coefficient, *exponent);
}

void writeRatio(llvm::json::OStream &json, ExactRatio value) {
  json.object([&] {
    json.attribute("numerator", value.numerator());
    json.attribute("denominator", value.denominator());
  });
}

llvm::Expected<ExactRatio> parseRatio(const llvm::json::Object &object,
                                      llvm::StringRef context) {
  if (llvm::Error error =
          rejectUnknownFields(object, context, {"numerator", "denominator"}))
    return std::move(error);
  auto numerator = requireUnsigned(object, "numerator", context);
  if (!numerator)
    return numerator.takeError();
  auto denominator = requireUnsigned(object, "denominator", context);
  if (!denominator)
    return denominator.takeError();
  return ExactRatio::get(*numerator, *denominator);
}

llvm::Expected<SubjectTargetRef>
parseTargetField(const llvm::json::Object &object, llvm::StringRef key,
                 llvm::StringRef context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return evaluationError(context + " field '" + key + "' is required");
  return parseSubjectTargetRefJson(*value);
}

llvm::Expected<ArtifactRootReference>
parseRootField(const llvm::json::Object &object, llvm::StringRef key,
               llvm::StringRef context) {
  auto value = requireObject(object, key, context);
  if (!value)
    return value.takeError();
  return parseArtifactRootReferenceJson(**value);
}

} // namespace

void writeEvaluationConditionJson(llvm::json::OStream &json,
                                  const EvaluationCondition &condition) {
  json.object([&] {
    json.attribute("kind", toString(condition.kind()));
    switch (condition.kind()) {
    case EvaluationConditionKind::ProcessCorner: {
      const auto &value = std::get<ProcessCornerCondition>(condition.payload);
      json.attributeBegin("target");
      writeSubjectTargetRefJson(json, value.target);
      json.attributeEnd();
      json.attributeBegin("corner");
      writeEncodedArtifactLocalReferenceJson(
          json, platform::encodeTechnologyCornerRef(value.corner));
      json.attributeEnd();
      return;
    }
    case EvaluationConditionKind::SupplyVoltage: {
      const auto &value = std::get<SupplyVoltageCondition>(condition.payload);
      json.attributeBegin("power_domain");
      writeSubjectTargetRefJson(json, value.powerDomain);
      json.attributeEnd();
      json.attributeBegin("volts");
      writeDecimal(json, value.volts);
      json.attributeEnd();
      return;
    }
    case EvaluationConditionKind::Temperature: {
      const auto &value = std::get<TemperatureCondition>(condition.payload);
      json.attributeBegin("thermal_domain_or_root");
      writeSubjectTargetRefJson(json, value.thermalDomainOrRoot);
      json.attributeEnd();
      json.attributeBegin("kelvin");
      writeDecimal(json, value.kelvin);
      json.attributeEnd();
      return;
    }
    case EvaluationConditionKind::RequiredClockPeriod: {
      const auto &value =
          std::get<RequiredClockPeriodCondition>(condition.payload);
      json.attributeBegin("clock_domain");
      writeSubjectTargetRefJson(json, value.clockDomain);
      json.attributeEnd();
      json.attributeBegin("seconds");
      writeDecimal(json, value.seconds);
      json.attributeEnd();
      return;
    }
    case EvaluationConditionKind::RelativeClockSchedule: {
      const auto &value =
          std::get<RelativeClockScheduleCondition>(condition.payload);
      json.attributeBegin("reference_clock");
      writeSubjectTargetRefJson(json, value.referenceClock);
      json.attributeEnd();
      json.attributeBegin("dependent_clock");
      writeSubjectTargetRefJson(json, value.dependentClock);
      json.attributeEnd();
      json.attributeBegin("dependent_period_per_reference_period");
      writeRatio(json, value.dependentPeriodPerReferencePeriod);
      json.attributeEnd();
      json.attributeBegin("dependent_phase_in_reference_cycles");
      writeRatio(json, value.dependentPhaseInReferenceCycles);
      json.attributeEnd();
      return;
    }
    case EvaluationConditionKind::ActivityBinding: {
      const auto &value = std::get<ActivityBindingCondition>(condition.payload);
      json.attributeBegin("target");
      writeSubjectTargetRefJson(json, value.target);
      json.attributeEnd();
      json.attributeBegin("source");
      json.object([&] {
        if (const auto *execution =
                std::get_if<ExecutionActivitySource>(&value.source)) {
          json.attribute("kind", "execution_activity");
          json.attributeBegin("simulation_execution_ref");
          writeArtifactRootReferenceJson(json, execution->simulationExecution);
          json.attributeEnd();
          json.attribute("activity_summary_ordinal",
                         execution->activitySummaryOrdinal);
          return;
        }
        const auto &assumption =
            std::get<ExplicitAssumptionSource>(value.source);
        json.attribute("kind", "explicit_assumption");
        json.attributeBegin("clock_domain");
        writeSubjectTargetRefJson(json, assumption.clockDomain);
        json.attributeEnd();
        json.attributeBegin("static_probability");
        writeRatio(json, assumption.staticProbability);
        json.attributeEnd();
        json.attributeBegin("transitions_per_clock");
        writeRatio(json, assumption.transitionsPerClock);
        json.attributeEnd();
      });
      json.attributeEnd();
      return;
    }
    case EvaluationConditionKind::Quantile: {
      const auto &value = std::get<QuantileCondition>(condition.payload);
      json.attributeBegin("probability");
      writeRatio(json, value.probability);
      json.attributeEnd();
      return;
    }
    }
  });
}

llvm::Expected<EvaluationCondition>
parseEvaluationConditionJson(const llvm::json::Value &value) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return evaluationError("evaluation condition must be an object");
  auto kindSpelling = requireString(*object, "kind", "evaluation condition");
  if (!kindSpelling)
    return kindSpelling.takeError();
  auto kind = parseEvaluationConditionKind(*kindSpelling);
  if (!kind)
    return kind.takeError();

  switch (*kind) {
  case EvaluationConditionKind::ProcessCorner: {
    if (llvm::Error error = rejectUnknownFields(
            *object, "process_corner condition", {"kind", "target", "corner"}))
      return std::move(error);
    auto target =
        parseTargetField(*object, "target", "process_corner condition");
    if (!target)
      return target.takeError();
    auto cornerObject =
        requireObject(*object, "corner", "process_corner condition");
    if (!cornerObject)
      return cornerObject.takeError();
    auto encoded = parseEncodedArtifactLocalReferenceJson(**cornerObject);
    if (!encoded)
      return encoded.takeError();
    auto corner = platform::decodeTechnologyCornerRef(*encoded);
    if (!corner)
      return corner.takeError();
    return EvaluationCondition{
        ProcessCornerCondition{std::move(*target), std::move(*corner)}};
  }
  case EvaluationConditionKind::SupplyVoltage: {
    if (llvm::Error error =
            rejectUnknownFields(*object, "supply_voltage condition",
                                {"kind", "power_domain", "volts"}))
      return std::move(error);
    auto target =
        parseTargetField(*object, "power_domain", "supply_voltage condition");
    if (!target)
      return target.takeError();
    auto decimalObject =
        requireObject(*object, "volts", "supply_voltage condition");
    if (!decimalObject)
      return decimalObject.takeError();
    auto decimal = parseDecimal(**decimalObject, "supply_voltage volts");
    if (!decimal)
      return decimal.takeError();
    return EvaluationCondition{
        SupplyVoltageCondition{std::move(*target), *decimal}};
  }
  case EvaluationConditionKind::Temperature: {
    if (llvm::Error error =
            rejectUnknownFields(*object, "temperature condition",
                                {"kind", "thermal_domain_or_root", "kelvin"}))
      return std::move(error);
    auto target = parseTargetField(*object, "thermal_domain_or_root",
                                   "temperature condition");
    if (!target)
      return target.takeError();
    auto decimalObject =
        requireObject(*object, "kelvin", "temperature condition");
    if (!decimalObject)
      return decimalObject.takeError();
    auto decimal = parseDecimal(**decimalObject, "temperature kelvin");
    if (!decimal)
      return decimal.takeError();
    return EvaluationCondition{
        TemperatureCondition{std::move(*target), *decimal}};
  }
  case EvaluationConditionKind::RequiredClockPeriod: {
    if (llvm::Error error =
            rejectUnknownFields(*object, "required_clock_period condition",
                                {"kind", "clock_domain", "seconds"}))
      return std::move(error);
    auto target = parseTargetField(*object, "clock_domain",
                                   "required_clock_period condition");
    if (!target)
      return target.takeError();
    auto decimalObject =
        requireObject(*object, "seconds", "required_clock_period condition");
    if (!decimalObject)
      return decimalObject.takeError();
    auto decimal = parseDecimal(**decimalObject, "required clock seconds");
    if (!decimal)
      return decimal.takeError();
    return EvaluationCondition{
        RequiredClockPeriodCondition{std::move(*target), *decimal}};
  }
  case EvaluationConditionKind::RelativeClockSchedule: {
    if (llvm::Error error =
            rejectUnknownFields(*object, "relative_clock_schedule condition",
                                {"kind", "reference_clock", "dependent_clock",
                                 "dependent_period_per_reference_period",
                                 "dependent_phase_in_reference_cycles"}))
      return std::move(error);
    auto reference = parseTargetField(*object, "reference_clock",
                                      "relative_clock_schedule condition");
    if (!reference)
      return reference.takeError();
    auto dependent = parseTargetField(*object, "dependent_clock",
                                      "relative_clock_schedule condition");
    if (!dependent)
      return dependent.takeError();
    auto periodObject =
        requireObject(*object, "dependent_period_per_reference_period",
                      "relative_clock_schedule condition");
    if (!periodObject)
      return periodObject.takeError();
    auto period = parseRatio(**periodObject, "dependent period ratio");
    if (!period)
      return period.takeError();
    auto phaseObject =
        requireObject(*object, "dependent_phase_in_reference_cycles",
                      "relative_clock_schedule condition");
    if (!phaseObject)
      return phaseObject.takeError();
    auto phase = parseRatio(**phaseObject, "dependent phase ratio");
    if (!phase)
      return phase.takeError();
    return EvaluationCondition{RelativeClockScheduleCondition{
        std::move(*reference), std::move(*dependent), *period, *phase}};
  }
  case EvaluationConditionKind::ActivityBinding: {
    if (llvm::Error error =
            rejectUnknownFields(*object, "activity_binding condition",
                                {"kind", "target", "source"}))
      return std::move(error);
    auto target =
        parseTargetField(*object, "target", "activity_binding condition");
    if (!target)
      return target.takeError();
    auto source =
        requireObject(*object, "source", "activity_binding condition");
    if (!source)
      return source.takeError();
    auto sourceKind =
        requireString(**source, "kind", "activity_binding source");
    if (!sourceKind)
      return sourceKind.takeError();
    if (*sourceKind == "execution_activity") {
      if (llvm::Error error = rejectUnknownFields(
              **source, "execution_activity source",
              {"kind", "simulation_execution_ref", "activity_summary_ordinal"}))
        return std::move(error);
      auto execution = parseRootField(**source, "simulation_execution_ref",
                                      "execution_activity source");
      if (!execution)
        return execution.takeError();
      auto ordinal = requireUnsigned(**source, "activity_summary_ordinal",
                                     "execution_activity source");
      if (!ordinal)
        return ordinal.takeError();
      return EvaluationCondition{ActivityBindingCondition{
          std::move(*target),
          ExecutionActivitySource{std::move(*execution), *ordinal}}};
    }
    if (*sourceKind != "explicit_assumption")
      return evaluationError("unknown activity_binding source kind '" +
                             *sourceKind + "'");
    if (llvm::Error error =
            rejectUnknownFields(**source, "explicit_assumption source",
                                {"kind", "clock_domain", "static_probability",
                                 "transitions_per_clock"}))
      return std::move(error);
    auto clock = parseTargetField(**source, "clock_domain",
                                  "explicit_assumption source");
    if (!clock)
      return clock.takeError();
    auto probabilityObject = requireObject(**source, "static_probability",
                                           "explicit_assumption source");
    if (!probabilityObject)
      return probabilityObject.takeError();
    auto probability = parseRatio(**probabilityObject, "static probability");
    if (!probability)
      return probability.takeError();
    auto transitionsObject = requireObject(**source, "transitions_per_clock",
                                           "explicit_assumption source");
    if (!transitionsObject)
      return transitionsObject.takeError();
    auto transitions = parseRatio(**transitionsObject, "transitions per clock");
    if (!transitions)
      return transitions.takeError();
    return EvaluationCondition{ActivityBindingCondition{
        std::move(*target),
        ExplicitAssumptionSource{std::move(*clock), *probability,
                                 *transitions}}};
  }
  case EvaluationConditionKind::Quantile: {
    if (llvm::Error error = rejectUnknownFields(*object, "quantile condition",
                                                {"kind", "probability"}))
      return std::move(error);
    auto probabilityObject =
        requireObject(*object, "probability", "quantile condition");
    if (!probabilityObject)
      return probabilityObject.takeError();
    auto probability = parseRatio(**probabilityObject, "quantile probability");
    if (!probability)
      return probability.takeError();
    return EvaluationCondition{QuantileCondition{*probability}};
  }
  }
  llvm_unreachable("unknown EvaluationConditionKind");
}

} // namespace loom::evaluation
