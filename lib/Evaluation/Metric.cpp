#include "Evaluation/Metric.h"
#include "Evaluation/CaseText.h"

#include "CanonicalSupport.h"

#include "Common/ArtifactText.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::evaluationError;
using detail::rejectUnknownFields;
using detail::requireInteger;
using detail::requireObject;
using detail::requireString;

constexpr llvm::StringLiteral metricSchemaIdentity = "evaluation.metric";
constexpr SchemaVersion metricSchemaVersion{1, 0};
constexpr llvm::StringLiteral metricQuerySchemaIdentity =
    "evaluation.metric_query";
constexpr SchemaVersion metricQuerySchemaVersion{1, 0};

constexpr std::uint8_t observationFormBit(ObservationForm form) {
  return std::uint8_t{1} << static_cast<std::uint8_t>(form);
}

constexpr std::uint8_t allObservationForms =
    observationFormBit(ObservationForm::Point) |
    observationFormBit(ObservationForm::Interval) |
    observationFormBit(ObservationForm::Censored) |
    observationFormBit(ObservationForm::NotApplicable);

constexpr std::uint8_t nonCensoredObservationForms =
    observationFormBit(ObservationForm::Point) |
    observationFormBit(ObservationForm::Interval) |
    observationFormBit(ObservationForm::NotApplicable);

constexpr CensoredReasonPolicy subjectDidNotCompletePolicy{
    CensoredReason::SubjectDidNotComplete, true, false};

// Every registered metric owns the same two scope forms: the entire exact
// Evaluation case, and one exact case subject Artifact root.
const ScopeRoleDescriptor subjectRootRole[] = {
    {ScopeRoleRef(0), "subject", true, {}}};

const ScopeFormDescriptor metricScopeForms[] = {
    {ScopeFormRef(0), "the entire exact Evaluation case", {}, nullptr},
    {ScopeFormRef(1), "one exact case subject Artifact root", subjectRootRole,
     nullptr},
};

// A quantile selects among samples of one metric request and names no target.
const ConditionPattern sampledRequestConditions[] = {
    {EvaluationConditionKind::Quantile, {}}};

const std::array<MetricDescriptor, 3> metricDescriptors = {{
    {MetricKind::CycleCount, "cycle_count",
     "Number of subject clock cycles required by the observed work.",
     MetricValueKind::Integer, MetricDimension::Cycle, "cycle",
     MetricValueDomain::NonNegative, metricScopeForms, sampledRequestConditions,
     allObservationForms, subjectDidNotCompletePolicy},
    {MetricKind::ClockPeriod,
     "clock_period",
     "Duration of one clock cycle for the evaluated operating condition.",
     MetricValueKind::Decimal,
     MetricDimension::Time,
     "second",
     MetricValueDomain::Positive,
     metricScopeForms,
     {},
     nonCensoredObservationForms,
     std::nullopt},
    {MetricKind::Runtime, "runtime",
     "Elapsed physical time for the observed work.", MetricValueKind::Decimal,
     MetricDimension::Time, "second", MetricValueDomain::NonNegative,
     metricScopeForms, sampledRequestConditions, allObservationForms,
     subjectDidNotCompletePolicy},
}};

template <typename Enum>
llvm::Expected<Enum> unknownEnum(llvm::StringRef type,
                                 llvm::StringRef spelling) {
  return evaluationError("unknown " + type + " '" + spelling + "'");
}

int compareMetricValue(const MetricValue &lhs, const MetricValue &rhs) {
  if (const auto *lhsInteger = std::get_if<IntegerValue>(&lhs)) {
    const auto &rhsInteger = std::get<IntegerValue>(rhs);
    if (lhsInteger->value() == rhsInteger.value())
      return 0;
    return lhsInteger->value() < rhsInteger.value() ? -1 : 1;
  }
  return compareDecimalValue(std::get<DecimalValue>(lhs),
                             std::get<DecimalValue>(rhs));
}

bool isZero(const MetricValue &value) {
  if (const auto *integer = std::get_if<IntegerValue>(&value))
    return integer->value() == 0;
  return std::get<DecimalValue>(value).coefficient() == 0;
}

bool isNegative(const MetricValue &value) {
  if (const auto *integer = std::get_if<IntegerValue>(&value))
    return integer->value() < 0;
  return std::get<DecimalValue>(value).coefficient() < 0;
}

llvm::Error validateValue(const MetricDescriptor &descriptor,
                          const MetricValue &value) {
  const bool isInteger = std::holds_alternative<IntegerValue>(value);
  if (descriptor.valueKind == MetricValueKind::Integer && !isInteger)
    return evaluationError(descriptor.spelling + " requires integer values");
  if (descriptor.valueKind == MetricValueKind::Decimal && isInteger)
    return evaluationError(descriptor.spelling + " requires decimal values");
  if (isNegative(value))
    return evaluationError(descriptor.spelling +
                           " requires non-negative values");
  if (descriptor.valueDomain == MetricValueDomain::Positive && isZero(value))
    return evaluationError(descriptor.spelling + " requires positive values");
  return llvm::Error::success();
}

llvm::Error validateOrderedValues(const MetricDescriptor &descriptor,
                                  const MetricValue &lower,
                                  const MetricValue &upper) {
  if (llvm::Error error = validateValue(descriptor, lower))
    return error;
  if (llvm::Error error = validateValue(descriptor, upper))
    return error;
  if (compareMetricValue(lower, upper) > 0)
    return evaluationError(
        "metric observation lower bound exceeds upper bound");
  return llvm::Error::success();
}

bool metricQueryLess(const MetricQuery &lhs, const MetricQuery &rhs) {
  const llvm::StringRef lhsSpelling = toString(lhs.metric);
  const llvm::StringRef rhsSpelling = toString(rhs.metric);
  if (lhsSpelling != rhsSpelling)
    return lhsSpelling < rhsSpelling;
  return canonicalScopeKey(lhs.scope) < canonicalScopeKey(rhs.scope);
}

void writeMetricValue(llvm::json::OStream &json, const MetricValue &value) {
  json.object([&] {
    if (const auto *integer = std::get_if<IntegerValue>(&value)) {
      json.attribute("kind", "integer");
      json.attribute("value", integer->value());
      return;
    }
    const DecimalValue decimal = std::get<DecimalValue>(value);
    json.attribute("kind", "decimal");
    json.attribute("coefficient", decimal.coefficient());
    json.attribute("base10_exponent", decimal.base10Exponent());
  });
}

void writeObservation(llvm::json::OStream &json,
                      const MetricObservationValue &observation) {
  json.object([&] {
    if (const auto *point = std::get_if<PointObservation>(&observation)) {
      json.attribute("form", "point");
      json.attributeBegin("value");
      writeMetricValue(json, point->value);
      json.attributeEnd();
      return;
    }
    if (const auto *interval = std::get_if<IntervalObservation>(&observation)) {
      json.attribute("form", "interval");
      json.attributeBegin("lower");
      writeMetricValue(json, interval->lower);
      json.attributeEnd();
      json.attributeBegin("upper");
      writeMetricValue(json, interval->upper);
      json.attributeEnd();
      return;
    }
    if (const auto *censored = std::get_if<CensoredObservation>(&observation)) {
      json.attribute("form", "censored");
      if (censored->lower) {
        json.attributeBegin("lower");
        writeMetricValue(json, *censored->lower);
        json.attributeEnd();
      }
      if (censored->upper) {
        json.attributeBegin("upper");
        writeMetricValue(json, *censored->upper);
        json.attributeEnd();
      }
      json.attribute("reason", toString(censored->reason));
      return;
    }
    const auto &notApplicable = std::get<NotApplicableObservation>(observation);
    json.attribute("form", "not_applicable");
    json.attribute("reason", toString(notApplicable.reason));
  });
}

llvm::Expected<MetricValue> parseMetricValue(const llvm::json::Object &object,
                                             llvm::StringRef context) {
  auto kind = requireString(object, "kind", context);
  if (!kind)
    return kind.takeError();
  if (*kind == "integer") {
    if (llvm::Error error =
            rejectUnknownFields(object, context, {"kind", "value"}))
      return std::move(error);
    auto value = requireInteger(object, "value", context);
    if (!value)
      return value.takeError();
    return MetricValue{IntegerValue(*value)};
  }
  if (*kind == "decimal") {
    if (llvm::Error error = rejectUnknownFields(
            object, context, {"kind", "coefficient", "base10_exponent"}))
      return std::move(error);
    auto coefficient = requireInteger(object, "coefficient", context);
    if (!coefficient)
      return coefficient.takeError();
    auto exponent = requireInteger(object, "base10_exponent", context);
    if (!exponent)
      return exponent.takeError();
    auto decimal = DecimalValue::get(*coefficient, *exponent);
    if (!decimal)
      return decimal.takeError();
    if (decimal->coefficient() != *coefficient ||
        decimal->base10Exponent() != *exponent)
      return evaluationError("decimal is not canonical");
    return MetricValue{*decimal};
  }
  return evaluationError(context + " has unknown value kind '" + *kind + "'");
}

llvm::Expected<MetricValue>
parseMetricValueField(const llvm::json::Object &object, llvm::StringRef key,
                      llvm::StringRef context) {
  auto valueObject = requireObject(object, key, context);
  if (!valueObject)
    return valueObject.takeError();
  std::string valueContext = (context + " field '" + key + "'").str();
  return parseMetricValue(**valueObject, valueContext);
}

llvm::Expected<EvaluationScope> parseScopeField(const llvm::json::Object &root,
                                                MetricKind metric,
                                                llvm::StringRef context) {
  auto scopeObject = requireObject(root, "scope", context);
  if (!scopeObject)
    return scopeObject.takeError();
  return parseEvaluationScopeJson(**scopeObject,
                                  metricDescriptor(metric).scopeForms);
}

llvm::Expected<MetricObservationValue>
parseObservationValue(const llvm::json::Object &object) {
  auto formSpelling = requireString(object, "form", "metric observation");
  if (!formSpelling)
    return formSpelling.takeError();
  auto form = parseObservationForm(*formSpelling);
  if (!form)
    return form.takeError();

  switch (*form) {
  case ObservationForm::Point: {
    if (llvm::Error error = rejectUnknownFields(object, "metric observation",
                                                {"form", "value"}))
      return std::move(error);
    auto value = parseMetricValueField(object, "value", "metric observation");
    if (!value)
      return value.takeError();
    return MetricObservationValue{PointObservation{std::move(*value)}};
  }
  case ObservationForm::Interval: {
    if (llvm::Error error = rejectUnknownFields(object, "metric observation",
                                                {"form", "lower", "upper"}))
      return std::move(error);
    auto lower = parseMetricValueField(object, "lower", "metric observation");
    if (!lower)
      return lower.takeError();
    auto upper = parseMetricValueField(object, "upper", "metric observation");
    if (!upper)
      return upper.takeError();
    return MetricObservationValue{
        IntervalObservation{std::move(*lower), std::move(*upper)}};
  }
  case ObservationForm::Censored: {
    if (llvm::Error error = rejectUnknownFields(
            object, "metric observation", {"form", "lower", "upper", "reason"}))
      return std::move(error);
    std::optional<MetricValue> lower;
    if (object.get("lower")) {
      auto value = parseMetricValueField(object, "lower", "metric observation");
      if (!value)
        return value.takeError();
      lower = std::move(*value);
    }
    std::optional<MetricValue> upper;
    if (object.get("upper")) {
      auto value = parseMetricValueField(object, "upper", "metric observation");
      if (!value)
        return value.takeError();
      upper = std::move(*value);
    }
    auto reasonSpelling = requireString(object, "reason", "metric observation");
    if (!reasonSpelling)
      return reasonSpelling.takeError();
    auto reason = parseCensoredReason(*reasonSpelling);
    if (!reason)
      return reason.takeError();
    return MetricObservationValue{
        CensoredObservation{std::move(lower), std::move(upper), *reason}};
  }
  case ObservationForm::NotApplicable: {
    if (llvm::Error error = rejectUnknownFields(object, "metric observation",
                                                {"form", "reason"}))
      return std::move(error);
    auto reasonSpelling = requireString(object, "reason", "metric observation");
    if (!reasonSpelling)
      return reasonSpelling.takeError();
    auto reason = parseNotApplicableReason(*reasonSpelling);
    if (!reason)
      return reason.takeError();
    return MetricObservationValue{NotApplicableObservation{*reason}};
  }
  }
  llvm_unreachable("unhandled ObservationForm");
}

} // namespace

bool MetricDescriptor::permitsObservationForm(ObservationForm form) const {
  return (permittedObservationForms & observationFormBit(form)) != 0;
}

ConditionApplicability MetricDescriptor::requestConditionApplicability() const {
  return ConditionApplicability{ConditionLocation::MetricRequest, spelling,
                                permittedRequestConditions};
}

llvm::ArrayRef<MetricDescriptor> allMetricDescriptors() {
  return metricDescriptors;
}

const MetricDescriptor &metricDescriptor(MetricKind kind) {
  for (const MetricDescriptor &descriptor : metricDescriptors)
    if (descriptor.kind == kind)
      return descriptor;
  llvm_unreachable("unknown MetricKind");
}

llvm::StringRef toString(MetricKind kind) {
  return metricDescriptor(kind).spelling;
}

llvm::StringRef toString(ObservationForm form) {
  switch (form) {
  case ObservationForm::Point:
    return "point";
  case ObservationForm::Interval:
    return "interval";
  case ObservationForm::Censored:
    return "censored";
  case ObservationForm::NotApplicable:
    return "not_applicable";
  }
  llvm_unreachable("unknown ObservationForm");
}

llvm::StringRef toString(UncertaintyKind kind) {
  switch (kind) {
  case UncertaintyKind::ExactWithinModel:
    return "exact_within_model";
  case UncertaintyKind::Bounded:
    return "bounded";
  case UncertaintyKind::Statistical:
    return "statistical";
  case UncertaintyKind::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown UncertaintyKind");
}

llvm::StringRef toString(CensoredReason reason) {
  switch (reason) {
  case CensoredReason::SubjectDidNotComplete:
    return "subject_did_not_complete";
  }
  llvm_unreachable("unknown CensoredReason");
}

llvm::StringRef toString(NotApplicableReason reason) {
  switch (reason) {
  case NotApplicableReason::UndefinedForSubject:
    return "undefined_for_subject";
  }
  llvm_unreachable("unknown NotApplicableReason");
}

llvm::Expected<MetricKind> parseMetricKind(llvm::StringRef spelling) {
  for (const MetricDescriptor &descriptor : metricDescriptors)
    if (descriptor.spelling == spelling)
      return descriptor.kind;
  return unknownEnum<MetricKind>("MetricKind", spelling);
}

llvm::Expected<ObservationForm> parseObservationForm(llvm::StringRef spelling) {
  if (spelling == "point")
    return ObservationForm::Point;
  if (spelling == "interval")
    return ObservationForm::Interval;
  if (spelling == "censored")
    return ObservationForm::Censored;
  if (spelling == "not_applicable")
    return ObservationForm::NotApplicable;
  return unknownEnum<ObservationForm>("ObservationForm", spelling);
}

llvm::Expected<UncertaintyKind> parseUncertaintyKind(llvm::StringRef spelling) {
  if (spelling == "exact_within_model")
    return UncertaintyKind::ExactWithinModel;
  if (spelling == "bounded")
    return UncertaintyKind::Bounded;
  if (spelling == "statistical")
    return UncertaintyKind::Statistical;
  if (spelling == "unknown")
    return UncertaintyKind::Unknown;
  return unknownEnum<UncertaintyKind>("UncertaintyKind", spelling);
}

llvm::Expected<CensoredReason> parseCensoredReason(llvm::StringRef spelling) {
  if (spelling == "subject_did_not_complete")
    return CensoredReason::SubjectDidNotComplete;
  return unknownEnum<CensoredReason>("CensoredReason", spelling);
}

llvm::Expected<NotApplicableReason>
parseNotApplicableReason(llvm::StringRef spelling) {
  if (spelling == "undefined_for_subject")
    return NotApplicableReason::UndefinedForSubject;
  return unknownEnum<NotApplicableReason>("NotApplicableReason", spelling);
}

llvm::Error validateMetricQuery(const MetricQuery &query) {
  return validateEvaluationScopeForm(metricDescriptor(query.metric).scopeForms,
                                     query.scope);
}

llvm::Expected<std::vector<MetricQuery>>
canonicalizeMetricQueries(llvm::ArrayRef<MetricQuery> queries) {
  std::vector<MetricQuery> canonical(queries.begin(), queries.end());
  for (const MetricQuery &query : canonical)
    if (llvm::Error error = validateMetricQuery(query))
      return std::move(error);

  std::sort(canonical.begin(), canonical.end(), metricQueryLess);
  for (std::size_t index = 1; index < canonical.size(); ++index)
    if (canonical[index - 1] == canonical[index])
      return evaluationError("duplicate metric query for '" +
                             toString(canonical[index].metric) + "'");
  return canonical;
}

llvm::Expected<std::string> serializeMetricQuery(const MetricQuery &query) {
  if (llvm::Error error = validateMetricQuery(query))
    return std::move(error);

  llvm::SmallString<256> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", metricQuerySchemaIdentity);
    json.attribute("schema_version",
                   formatSchemaVersion(metricQuerySchemaVersion));
    json.attribute("metric", toString(query.metric));
    json.attributeBegin("scope");
    writeEvaluationScopeJson(json, query.scope);
    json.attributeEnd();
  });
  return output.str().str();
}

llvm::Expected<MetricQuery> parseMetricQuery(llvm::StringRef json) {
  auto value = llvm::json::parse(json);
  if (!value)
    return value.takeError();
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return evaluationError("evaluation.metric_query root must be an object");
  if (llvm::Error error =
          rejectUnknownFields(*root, "evaluation.metric_query root",
                              {"schema", "schema_version", "metric", "scope"}))
    return std::move(error);

  auto schema = requireString(*root, "schema", "evaluation.metric_query root");
  if (!schema)
    return schema.takeError();
  if (*schema != metricQuerySchemaIdentity)
    return evaluationError("unsupported metric query schema '" + *schema + "'");
  auto version =
      requireString(*root, "schema_version", "evaluation.metric_query root");
  if (!version)
    return version.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  if (*parsedVersion != metricQuerySchemaVersion)
    return evaluationError("unsupported evaluation.metric_query version '" +
                           *version + "'");

  auto metricSpelling =
      requireString(*root, "metric", "evaluation.metric_query root");
  if (!metricSpelling)
    return metricSpelling.takeError();
  auto metric = parseMetricKind(*metricSpelling);
  if (!metric)
    return metric.takeError();

  auto scope = parseScopeField(*root, *metric, "evaluation.metric_query root");
  if (!scope)
    return scope.takeError();

  MetricQuery query{*metric, std::move(*scope)};
  if (llvm::Error error = validateMetricQuery(query))
    return std::move(error);
  auto canonical = serializeMetricQuery(query);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != json)
    return evaluationError("metric query JSON is not canonical");
  return query;
}

ObservationForm observationForm(const MetricObservation &observation) {
  if (std::holds_alternative<PointObservation>(observation.observation))
    return ObservationForm::Point;
  if (std::holds_alternative<IntervalObservation>(observation.observation))
    return ObservationForm::Interval;
  if (std::holds_alternative<CensoredObservation>(observation.observation))
    return ObservationForm::Censored;
  return ObservationForm::NotApplicable;
}

llvm::Error validateMetricObservation(const MetricObservation &observation) {
  const MetricDescriptor &descriptor = metricDescriptor(observation.metric);
  const ObservationForm form = observationForm(observation);
  if (!descriptor.permitsObservationForm(form))
    return evaluationError(descriptor.spelling + " does not permit " +
                           toString(form) + " observations");

  if (llvm::Error error =
          validateEvaluationScopeForm(descriptor.scopeForms, observation.scope))
    return error;

  if (const auto *point =
          std::get_if<PointObservation>(&observation.observation))
    return validateValue(descriptor, point->value);

  if (const auto *interval =
          std::get_if<IntervalObservation>(&observation.observation))
    return validateOrderedValues(descriptor, interval->lower, interval->upper);

  if (const auto *censored =
          std::get_if<CensoredObservation>(&observation.observation)) {
    if (!descriptor.censoredReasonPolicy ||
        descriptor.censoredReasonPolicy->reason != censored->reason)
      return evaluationError(descriptor.spelling +
                             " does not permit censored reason '" +
                             toString(censored->reason) + "'");
    const CensoredReasonPolicy &policy = *descriptor.censoredReasonPolicy;
    if (policy.requiresLowerBound && !censored->lower)
      return evaluationError(toString(censored->reason) +
                             " requires a lower bound");
    if (!policy.permitsUpperBound && censored->upper)
      return evaluationError(toString(censored->reason) +
                             " does not permit an upper bound");
    if (censored->lower && censored->upper)
      return validateOrderedValues(descriptor, *censored->lower,
                                   *censored->upper);
    if (censored->lower)
      return validateValue(descriptor, *censored->lower);
    if (censored->upper)
      return validateValue(descriptor, *censored->upper);
    return evaluationError("censored observation requires at least one bound");
  }

  if (observation.uncertainty != UncertaintyKind::Unknown)
    return evaluationError("not_applicable requires unknown uncertainty");
  return llvm::Error::success();
}

llvm::Expected<std::string>
serializeMetricObservation(const MetricObservation &observation) {
  if (llvm::Error error = validateMetricObservation(observation))
    return std::move(error);

  llvm::SmallString<512> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", metricSchemaIdentity);
    json.attribute("schema_version", formatSchemaVersion(metricSchemaVersion));
    json.attribute("metric", toString(observation.metric));
    json.attributeBegin("scope");
    writeEvaluationScopeJson(json, observation.scope);
    json.attributeEnd();
    json.attribute("uncertainty", toString(observation.uncertainty));
    json.attributeBegin("observation");
    writeObservation(json, observation.observation);
    json.attributeEnd();
  });
  return output.str().str();
}

llvm::Expected<MetricObservation> parseMetricObservation(llvm::StringRef json) {
  auto value = llvm::json::parse(json);
  if (!value)
    return value.takeError();
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return evaluationError("evaluation.metric root must be an object");
  if (llvm::Error error =
          rejectUnknownFields(*root, "evaluation.metric root",
                              {"schema", "schema_version", "metric", "scope",
                               "uncertainty", "observation"}))
    return std::move(error);

  auto schema = requireString(*root, "schema", "evaluation.metric root");
  if (!schema)
    return schema.takeError();
  if (*schema != metricSchemaIdentity)
    return evaluationError("unsupported metric schema '" + *schema + "'");
  auto version =
      requireString(*root, "schema_version", "evaluation.metric root");
  if (!version)
    return version.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  if (*parsedVersion != metricSchemaVersion)
    return evaluationError("unsupported evaluation.metric version '" +
                           *version + "'");

  auto metricSpelling =
      requireString(*root, "metric", "evaluation.metric root");
  if (!metricSpelling)
    return metricSpelling.takeError();
  auto metric = parseMetricKind(*metricSpelling);
  if (!metric)
    return metric.takeError();

  auto scope = parseScopeField(*root, *metric, "evaluation.metric root");
  if (!scope)
    return scope.takeError();

  auto uncertaintySpelling =
      requireString(*root, "uncertainty", "evaluation.metric root");
  if (!uncertaintySpelling)
    return uncertaintySpelling.takeError();
  auto uncertainty = parseUncertaintyKind(*uncertaintySpelling);
  if (!uncertainty)
    return uncertainty.takeError();

  auto observationObject =
      requireObject(*root, "observation", "evaluation.metric root");
  if (!observationObject)
    return observationObject.takeError();
  auto observationValue = parseObservationValue(**observationObject);
  if (!observationValue)
    return observationValue.takeError();

  MetricObservation observation{*metric, std::move(*scope), *uncertainty,
                                std::move(*observationValue)};
  if (llvm::Error error = validateMetricObservation(observation))
    return std::move(error);
  auto canonical = serializeMetricObservation(observation);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != json)
    return evaluationError("metric JSON is not canonical");
  return observation;
}

} // namespace loom::evaluation
