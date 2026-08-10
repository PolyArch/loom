#include "Evaluation/Metric.h"
#include "Evaluation/CaseText.h"
#include "Evaluation/MetricText.h"
#include "Evaluation/ProductionRegistry.h"
#include "QueryText.h"

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

constexpr llvm::StringLiteral metricQuerySchemaIdentity =
    "evaluation.metric_query";
constexpr SchemaVersion metricQuerySchemaVersion{1, 0};

constexpr std::uint8_t allObservationForms = allObservationFormsMask();

constexpr std::uint8_t nonCensoredObservationForms =
    observationFormMask(ObservationForm::Point) |
    observationFormMask(ObservationForm::Interval) |
    observationFormMask(ObservationForm::NotApplicable);

constexpr CensoredReasonPolicy subjectDidNotCompletePolicy{
    CensoredReason::SubjectDidNotComplete, true, false};

const ScopeFormDescriptor cycleBasedWholeCaseScopeForms[] = {
    {ScopeFormRef(0),
     "the entire exact Evaluation case",
     {},
     WholeExactCaseScope{},
     nullptr,
     ReferenceCycleRequirement::ExactCaseUniqueReferenceCycle},
};

const ScopeFormDescriptor runtimeWholeCaseScopeForms[] = {
    {ScopeFormRef(0),
     "the entire exact Evaluation case",
     {},
     WholeExactCaseScope{},
     nullptr,
     ReferenceCycleRequirement::NotRequired},
};

const ClosedDecimalIntervalMetricDomain predictionErrorDomain{
    llvm::cantFail(DecimalValue::get(0, 0)),
    llvm::cantFail(DecimalValue::get(2, 0))};

ConditionApplicabilityPattern quantilePattern(EvaluationCaseKind caseKind) {
  return ConditionApplicabilityPattern{
      EvaluationConditionKind::Quantile,
      {llvm::cantFail(EvaluationCaseSignatureRef::get(evaluationSchemaVersion(),
                                                      caseKind)),
       {}}};
}

llvm::ArrayRef<ConditionApplicabilityPattern> fpaQuantilePatterns() {
  static const std::array<ConditionApplicabilityPattern, 1> patterns = {
      quantilePattern(builtinEvaluationCaseKind(
          BuiltinEvaluationCase::FpaModelParameterCalibration))};
  return patterns;
}

llvm::ArrayRef<ConditionApplicabilityPattern> runtimeQuantilePatterns() {
  static const std::array<ConditionApplicabilityPattern, 1> patterns = {
      quantilePattern(builtinEvaluationCaseKind(
          BuiltinEvaluationCase::SystemRuntimeModelParameterCalibration))};
  return patterns;
}

const std::array<MetricDescriptor, 13> metricDescriptors = {{
    {MetricKind::CycleCount,
     "cycle_count",
     "Number of subject clock cycles required by the observed work.",
     MetricValueKind::Integer,
     MetricDimension::Cycle,
     "cycle",
     NonNegativeMetricDomain{},
     cycleBasedWholeCaseScopeForms,
     {},
     allObservationForms,
     subjectDidNotCompletePolicy},
    {MetricKind::ClockPeriod,
     "clock_period",
     "Duration of one clock cycle for the evaluated operating condition.",
     MetricValueKind::Decimal,
     MetricDimension::Time,
     "second",
     PositiveMetricDomain{},
     cycleBasedWholeCaseScopeForms,
     {},
     nonCensoredObservationForms,
     std::nullopt},
    {MetricKind::Runtime,
     "runtime",
     "Elapsed physical time for the observed work.",
     MetricValueKind::Decimal,
     MetricDimension::Time,
     "second",
     NonNegativeMetricDomain{},
     runtimeWholeCaseScopeForms,
     {},
     allObservationForms,
     subjectDidNotCompletePolicy},
    {MetricKind::LimitingClockFrequency,
     "limiting_clock_frequency",
     "Maximum common clock frequency permitted by the limiting synchronous "
     "domain of the exact evaluated case.",
     MetricValueKind::Decimal,
     MetricDimension::Frequency,
     "hertz",
     PositiveMetricDomain{},
     runtimeWholeCaseScopeForms,
     {},
     nonCensoredObservationForms,
     std::nullopt},
    {MetricKind::TotalArea,
     "total_area",
     "Total physical implementation footprint, including cells, macros, "
     "and allocated routing area.",
     MetricValueKind::Decimal,
     MetricDimension::Area,
     "square_meter",
     NonNegativeMetricDomain{},
     runtimeWholeCaseScopeForms,
     {},
     nonCensoredObservationForms,
     std::nullopt},
    {MetricKind::DynamicPower,
     "dynamic_power",
     "Average dynamic power under the exact evaluated workload, activity, "
     "and operating conditions.",
     MetricValueKind::Decimal,
     MetricDimension::Power,
     "watt",
     NonNegativeMetricDomain{},
     runtimeWholeCaseScopeForms,
     {},
     nonCensoredObservationForms,
     std::nullopt},
    {MetricKind::LeakagePower,
     "leakage_power",
     "Average static leakage power under the exact evaluated operating "
     "conditions.",
     MetricValueKind::Decimal,
     MetricDimension::Power,
     "watt",
     NonNegativeMetricDomain{},
     runtimeWholeCaseScopeForms,
     {},
     nonCensoredObservationForms,
     std::nullopt},
    {MetricKind::MaximumVoltageDrop,
     "maximum_voltage_drop",
     "Greatest nonnegative applied-to-delivered supply-voltage difference "
     "over the complete analyzed power network of the exact evaluated case.",
     MetricValueKind::Decimal,
     MetricDimension::Voltage,
     "volt",
     NonNegativeMetricDomain{},
     runtimeWholeCaseScopeForms,
     {},
     nonCensoredObservationForms,
     std::nullopt},
    {MetricKind::LimitingClockFrequencyPredictionError,
     "limiting_clock_frequency_prediction_error",
     "Symmetric relative error of a limiting-clock-frequency prediction.",
     MetricValueKind::Decimal, MetricDimension::Dimensionless, "one",
     predictionErrorDomain, runtimeWholeCaseScopeForms, fpaQuantilePatterns(),
     nonCensoredObservationForms, std::nullopt, fpaQuantilePatterns()},
    {MetricKind::TotalAreaPredictionError, "total_area_prediction_error",
     "Symmetric relative error of a total-area prediction.",
     MetricValueKind::Decimal, MetricDimension::Dimensionless, "one",
     predictionErrorDomain, runtimeWholeCaseScopeForms, fpaQuantilePatterns(),
     nonCensoredObservationForms, std::nullopt, fpaQuantilePatterns()},
    {MetricKind::DynamicPowerPredictionError, "dynamic_power_prediction_error",
     "Symmetric relative error of a dynamic-power prediction.",
     MetricValueKind::Decimal, MetricDimension::Dimensionless, "one",
     predictionErrorDomain, runtimeWholeCaseScopeForms, fpaQuantilePatterns(),
     nonCensoredObservationForms, std::nullopt, fpaQuantilePatterns()},
    {MetricKind::LeakagePowerPredictionError, "leakage_power_prediction_error",
     "Symmetric relative error of a leakage-power prediction.",
     MetricValueKind::Decimal, MetricDimension::Dimensionless, "one",
     predictionErrorDomain, runtimeWholeCaseScopeForms, fpaQuantilePatterns(),
     nonCensoredObservationForms, std::nullopt, fpaQuantilePatterns()},
    {MetricKind::RuntimePredictionError, "runtime_prediction_error",
     "Symmetric relative error of a whole-system runtime prediction.",
     MetricValueKind::Decimal, MetricDimension::Dimensionless, "one",
     predictionErrorDomain, runtimeWholeCaseScopeForms,
     runtimeQuantilePatterns(), nonCensoredObservationForms, std::nullopt,
     runtimeQuantilePatterns()},
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
  if (std::holds_alternative<NonNegativeMetricDomain>(descriptor.valueDomain)) {
    if (isNegative(value))
      return evaluationError(descriptor.spelling +
                             " requires non-negative values");
    return llvm::Error::success();
  }
  if (std::holds_alternative<PositiveMetricDomain>(descriptor.valueDomain)) {
    if (isNegative(value) || isZero(value))
      return evaluationError(descriptor.spelling + " requires positive values");
    return llvm::Error::success();
  }

  if (isInteger)
    return evaluationError(descriptor.spelling +
                           " requires decimal interval values");
  const auto &domain =
      std::get<ClosedDecimalIntervalMetricDomain>(descriptor.valueDomain);
  const MetricValue lower = domain.lower;
  const MetricValue upper = domain.upper;
  if (compareMetricValue(value, lower) < 0 ||
      compareMetricValue(value, upper) > 0)
    return evaluationError(descriptor.spelling +
                           " is outside its closed decimal interval");
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
  if (lhs.metric != rhs.metric)
    return lhs.metric < rhs.metric;
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
  return (permittedObservationForms & observationFormMask(form)) != 0;
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
  case UncertaintyKind::Unquantified:
    return "unquantified";
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
  if (spelling == "unquantified")
    return UncertaintyKind::Unquantified;
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

llvm::Error validateMetricScopeAdmissibility(
    MetricKind metric, ScopeFormRef form,
    const EvaluationCaseSignatureDescriptor &caseSignature) {
  const ScopeFormDescriptor *scope =
      findScopeForm(metricDescriptor(metric).scopeForms, form);
  if (!scope)
    return evaluationError("unknown scope form ordinal " +
                           std::to_string(form.ordinal()));
  switch (scope->referenceCycleRequirement) {
  case ReferenceCycleRequirement::NotRequired:
    return llvm::Error::success();
  case ReferenceCycleRequirement::ExactCaseUniqueReferenceCycle:
    if (std::holds_alternative<AbsentReferenceCycle>(
            caseSignature.wholeCaseCycleBasis))
      return evaluationError("case signature '" + caseSignature.spelling +
                             "' requires a unique whole-case reference cycle "
                             "for metric '" +
                             toString(metric) + "'");
    return llvm::Error::success();
  }
  llvm_unreachable("unknown ReferenceCycleRequirement");
}

llvm::Expected<std::optional<ReferenceCycleBasis>>
validateMetricScopeAdmissibility(MetricKind metric, ScopeFormRef form,
                                 const EvaluationCase &evaluationCase,
                                 const CaseArtifactResolution &resolution,
                                 const ArtifactStore &artifactStore,
                                 const BlobStore &blobStore) {
  const EvaluationCaseSignatureDescriptor *signature =
      evaluationCase.signature().descriptor();
  if (!signature)
    return evaluationError("the EvaluationCase signature is unresolved");
  if (llvm::Error error =
          validateMetricScopeAdmissibility(metric, form, *signature))
    return std::move(error);
  const ScopeFormDescriptor *scope =
      findScopeForm(metricDescriptor(metric).scopeForms, form);
  if (scope->referenceCycleRequirement ==
      ReferenceCycleRequirement::NotRequired)
    return std::nullopt;
  // ExactCaseUniqueReferenceCycle: the case-signature registry resolves the
  // basis once and the metric registry returns the validated basis.
  auto basis = resolveReferenceCycleBasis(evaluationCase, resolution,
                                          artifactStore, blobStore);
  if (!basis)
    return basis.takeError();
  return std::optional<ReferenceCycleBasis>{std::move(*basis)};
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
    detail::writeMetricQueryPayload(json, query);
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

  const llvm::StringRef envelopeFields[] = {"schema", "schema_version"};
  auto query = detail::parseMetricQueryPayload(
      *root, "evaluation.metric_query root", envelopeFields);
  if (!query)
    return query.takeError();
  if (llvm::Error error = validateMetricQuery(*query))
    return std::move(error);
  auto canonical = serializeMetricQuery(*query);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != json)
    return evaluationError("metric query JSON is not canonical");
  return *query;
}

ObservationForm observationForm(const MetricObservationValue &observation) {
  if (std::holds_alternative<PointObservation>(observation))
    return ObservationForm::Point;
  if (std::holds_alternative<IntervalObservation>(observation))
    return ObservationForm::Interval;
  if (std::holds_alternative<CensoredObservation>(observation))
    return ObservationForm::Censored;
  return ObservationForm::NotApplicable;
}

llvm::Error
validateMetricObservationValue(MetricKind metric, UncertaintyKind uncertainty,
                               const MetricObservationValue &observation) {
  const MetricDescriptor &descriptor = metricDescriptor(metric);
  const ObservationForm form = observationForm(observation);
  if (!descriptor.permitsObservationForm(form))
    return evaluationError(descriptor.spelling + " does not permit " +
                           toString(form) + " observations");

  if (const auto *point = std::get_if<PointObservation>(&observation))
    return validateValue(descriptor, point->value);

  if (const auto *interval = std::get_if<IntervalObservation>(&observation))
    return validateOrderedValues(descriptor, interval->lower, interval->upper);

  if (const auto *censored = std::get_if<CensoredObservation>(&observation)) {
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

  if (uncertainty != UncertaintyKind::Unquantified)
    return evaluationError("not_applicable requires unquantified uncertainty");
  return llvm::Error::success();
}

void writeMetricObservationValueJson(
    llvm::json::OStream &json, const MetricObservationValue &observation) {
  writeObservation(json, observation);
}

llvm::Expected<MetricObservationValue>
parseMetricObservationValueJson(const llvm::json::Object &object) {
  return parseObservationValue(object);
}

} // namespace loom::evaluation
