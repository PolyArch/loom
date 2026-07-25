#include "Evaluation/Evidence.h"

#include "CanonicalSupport.h"
#include "Evaluation/CaseText.h"
#include "Evaluation/MetricText.h"

#include "Common/ArtifactText.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::evaluationError;
using detail::rejectUnknownFields;
using detail::requireArray;
using detail::requireObject;
using detail::requireString;
using detail::requireUnsigned;

llvm::Expected<std::uint32_t> requireOrdinal(const llvm::json::Object &object,
                                             llvm::StringRef key,
                                             llvm::StringRef context) {
  auto value = requireUnsigned(object, key, context);
  if (!value)
    return value.takeError();
  if (*value > std::numeric_limits<std::uint32_t>::max())
    return evaluationError(context + " field '" + key + "' is out of range");
  return static_cast<std::uint32_t>(*value);
}

void writeOutputBindings(llvm::json::OStream &json,
                         llvm::ArrayRef<ModelOutputBinding> bindings) {
  json.array([&] {
    for (const ModelOutputBinding &binding : bindings) {
      json.object([&] {
        json.attribute("slot", binding.slot.ordinal());
        json.attributeArray("artifacts", [&] {
          for (const ArtifactRootReference &artifact : binding.artifacts)
            writeArtifactRootReferenceJson(json, artifact);
        });
      });
    }
  });
}

llvm::Expected<std::vector<ModelOutputBinding>>
parseOutputBindings(const llvm::json::Object &root) {
  auto array =
      requireArray(root, "output_bindings", "evaluation.evidence root");
  if (!array)
    return array.takeError();
  std::vector<ModelOutputBinding> bindings;
  bindings.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return evaluationError("model output binding must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *object, "model output binding", {"slot", "artifacts"}))
      return std::move(error);
    auto slot = requireOrdinal(*object, "slot", "model output binding");
    if (!slot)
      return slot.takeError();
    auto artifacts =
        requireArray(*object, "artifacts", "model output binding");
    if (!artifacts)
      return artifacts.takeError();
    ModelOutputBinding binding{ModelOutputSlotRef(*slot), {}};
    binding.artifacts.reserve((*artifacts)->size());
    for (const llvm::json::Value &artifactValue : **artifacts) {
      const llvm::json::Object *artifact = artifactValue.getAsObject();
      if (!artifact)
        return evaluationError("model output Artifact must be an object");
      auto reference = parseArtifactRootReferenceJson(*artifact);
      if (!reference)
        return reference.takeError();
      binding.artifacts.push_back(std::move(*reference));
    }
    bindings.push_back(std::move(binding));
  }
  return bindings;
}

void writeMetricResult(llvm::json::OStream &json, const MetricResult &result) {
  json.object([&] {
    json.attribute("uncertainty", toString(result.uncertainty));
    json.attributeBegin("observation");
    writeMetricObservationValueJson(json, result.observation);
    json.attributeEnd();
    json.attributeArray("calibration_input_slots", [&] {
      for (ModelInputSlotRef slot : result.calibrationInputSlots)
        json.value(slot.ordinal());
    });
  });
}

llvm::Expected<MetricResult>
parseMetricResult(const llvm::json::Value &value) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return evaluationError("metric result must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object, "metric result",
          {"uncertainty", "observation", "calibration_input_slots"}))
    return std::move(error);
  auto uncertaintySpelling =
      requireString(*object, "uncertainty", "metric result");
  if (!uncertaintySpelling)
    return uncertaintySpelling.takeError();
  auto uncertainty = parseUncertaintyKind(*uncertaintySpelling);
  if (!uncertainty)
    return uncertainty.takeError();
  auto observationObject =
      requireObject(*object, "observation", "metric result");
  if (!observationObject)
    return observationObject.takeError();
  auto observation =
      parseMetricObservationValueJson(**observationObject);
  if (!observation)
    return observation.takeError();
  auto slots =
      requireArray(*object, "calibration_input_slots", "metric result");
  if (!slots)
    return slots.takeError();
  std::vector<ModelInputSlotRef> calibrationSlots;
  calibrationSlots.reserve((*slots)->size());
  for (const llvm::json::Value &slotValue : **slots) {
    auto slot = slotValue.getAsUINT64();
    if (!slot || *slot > std::numeric_limits<std::uint32_t>::max())
      return evaluationError(
          "metric calibration input slot must be a uint32 ordinal");
    calibrationSlots.emplace_back(static_cast<std::uint32_t>(*slot));
  }
  return MetricResult{*uncertainty, std::move(*observation),
                      std::move(calibrationSlots)};
}

void writeFindingOccurrence(llvm::json::OStream &json,
                            const FindingOccurrence &occurrence) {
  json.value(occurrence.canonicalHex());
}

llvm::Expected<FindingOccurrence>
parseFindingOccurrence(const llvm::json::Value &value,
                       const FindingOccurrenceCodec &codec,
                       const FindingOccurrenceContext &context) {
  auto payload = value.getAsString();
  if (!payload)
    return evaluationError("finding occurrence must be a hexadecimal string");
  auto bytes = parseArtifactLocalPayloadHex(*payload);
  if (!bytes)
    return bytes.takeError();
  return FindingOccurrence::decode(codec, *bytes, context);
}

void writeFindingResult(llvm::json::OStream &json,
                        const FindingResult &result) {
  json.object([&] {
    if (std::holds_alternative<AbsentFinding>(result.result)) {
      json.attribute("state", "absent");
      return;
    }
    if (const auto *present = std::get_if<PresentFinding>(&result.result)) {
      json.attribute("state", "present");
      json.attributeArray("occurrences", [&] {
        for (const FindingOccurrence &occurrence : present->occurrences)
          writeFindingOccurrence(json, occurrence);
      });
      return;
    }
    const auto notApplicable =
        std::get<NotApplicableFinding>(result.result);
    json.attribute("state", "not_applicable");
    json.attribute("reason", toString(notApplicable.reason));
  });
}

llvm::Expected<FindingResult>
parseFindingResult(const llvm::json::Value &value,
                   const EvaluationRequest &request,
                   FindingRequestOrdinal ordinal,
                   llvm::ArrayRef<ModelOutputBinding> outputBindings,
                   const CaseArtifactResolution &resolution,
                   const ArtifactStore &artifactStore) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return evaluationError("finding result must be an object");
  auto state = requireString(*object, "state", "finding result");
  if (!state)
    return state.takeError();
  if (*state == "absent") {
    if (llvm::Error error = rejectUnknownFields(
            *object, "absent finding result", {"state"}))
      return std::move(error);
    return FindingResult{AbsentFinding{}};
  }
  if (*state == "present") {
    if (llvm::Error error = rejectUnknownFields(
            *object, "present finding result",
            {"state", "occurrences"}))
      return std::move(error);
    auto array =
        requireArray(*object, "occurrences", "present finding result");
    if (!array)
      return array.takeError();
    PresentFinding present;
    present.occurrences.reserve((*array)->size());
    const FindingRequest *requestItem = request.resolve(ordinal);
    if (!requestItem)
      return evaluationError("finding result ordinal is out of range");
    const FindingDescriptor *descriptor =
        findFindingDescriptor(requestItem->query().kind);
    if (!descriptor)
      return evaluationError("finding result kind is unregistered");
    if (llvm::Error error = requireFindingOccurrenceOwner(*descriptor))
      return std::move(error);
    const FindingOccurrenceContext context(
        request, ordinal, outputBindings, resolution, artifactStore);
    for (const llvm::json::Value &occurrenceValue : **array) {
      auto occurrence = parseFindingOccurrence(
          occurrenceValue, descriptor->occurrenceCodec, context);
      if (!occurrence)
        return occurrence.takeError();
      present.occurrences.push_back(std::move(*occurrence));
    }
    return FindingResult{std::move(present)};
  }
  if (*state != "not_applicable")
    return evaluationError("unknown finding result state '" + *state + "'");
  if (llvm::Error error = rejectUnknownFields(
          *object, "not-applicable finding result",
          {"state", "reason"}))
    return std::move(error);
  auto reasonSpelling =
      requireString(*object, "reason", "not-applicable finding result");
  if (!reasonSpelling)
    return reasonSpelling.takeError();
  auto reason = parseNotApplicableReason(*reasonSpelling);
  if (!reason)
    return reason.takeError();
  return FindingResult{NotApplicableFinding{*reason}};
}

void writeOutcome(llvm::json::OStream &json,
                  const EvaluationEvidenceOutcome &outcome) {
  json.object([&] {
    if (const auto *completed = std::get_if<CompletedEvidence>(&outcome)) {
      json.attribute("kind", "completed");
      json.attributeArray("metric_results", [&] {
        for (const MetricResult &result : completed->metricResults)
          writeMetricResult(json, result);
      });
      json.attributeArray("finding_results", [&] {
        for (const FindingResult &result : completed->findingResults)
          writeFindingResult(json, result);
      });
      return;
    }
    if (const auto *unsupported =
            std::get_if<UnsupportedEvidence>(&outcome)) {
      json.attribute("kind", "unsupported");
      json.attribute("reason", toString(unsupported->reason));
      return;
    }
    if (const auto *failed =
            std::get_if<ExecutionFailedEvidence>(&outcome)) {
      json.attribute("kind", "execution_failed");
      json.attribute("reason", toString(failed->reason));
      return;
    }
    const auto cancelled = std::get<CancelledOrTimeoutEvidence>(outcome);
    json.attribute("kind", "cancelled_or_timeout");
    json.attribute("reason", toString(cancelled.reason));
  });
}

llvm::Expected<EvaluationEvidenceOutcome>
parseOutcome(const llvm::json::Object &root, const EvaluationRequest &request,
             llvm::ArrayRef<ModelOutputBinding> outputBindings,
             const CaseArtifactResolution &resolution,
             const ArtifactStore &artifactStore) {
  auto object = requireObject(root, "outcome", "evaluation.evidence root");
  if (!object)
    return object.takeError();
  auto kind = requireString(**object, "kind", "Evidence outcome");
  if (!kind)
    return kind.takeError();
  if (*kind == "completed") {
    if (llvm::Error error = rejectUnknownFields(
            **object, "completed Evidence outcome",
            {"kind", "metric_results", "finding_results"}))
      return std::move(error);
    auto metrics =
        requireArray(**object, "metric_results", "completed Evidence outcome");
    if (!metrics)
      return metrics.takeError();
    CompletedEvidence completed;
    completed.metricResults.reserve((*metrics)->size());
    for (const llvm::json::Value &metricValue : **metrics) {
      auto metric = parseMetricResult(metricValue);
      if (!metric)
        return metric.takeError();
      completed.metricResults.push_back(std::move(*metric));
    }
    auto findings = requireArray(**object, "finding_results",
                                 "completed Evidence outcome");
    if (!findings)
      return findings.takeError();
    completed.findingResults.reserve((*findings)->size());
    for (std::size_t index = 0; index < (*findings)->size(); ++index) {
      auto finding = parseFindingResult(
          (**findings)[index], request, FindingRequestOrdinal(index),
          outputBindings, resolution, artifactStore);
      if (!finding)
        return finding.takeError();
      completed.findingResults.push_back(std::move(*finding));
    }
    return EvaluationEvidenceOutcome{std::move(completed)};
  }

  if (llvm::Error error = rejectUnknownFields(
          **object, "noncompleted Evidence outcome", {"kind", "reason"}))
    return std::move(error);
  auto reasonSpelling =
      requireString(**object, "reason", "noncompleted Evidence outcome");
  if (!reasonSpelling)
    return reasonSpelling.takeError();
  auto reason = parseOutcomeReason(*reasonSpelling);
  if (!reason)
    return reason.takeError();
  if (*kind == "unsupported")
    return EvaluationEvidenceOutcome{UnsupportedEvidence{*reason}};
  if (*kind == "execution_failed")
    return EvaluationEvidenceOutcome{ExecutionFailedEvidence{*reason}};
  if (*kind == "cancelled_or_timeout")
    return EvaluationEvidenceOutcome{CancelledOrTimeoutEvidence{*reason}};
  return evaluationError("unknown Evidence outcome kind '" + *kind + "'");
}

} // namespace

std::string serializeEvaluationEvidence(const EvaluationEvidence &evidence) {
  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", EvaluationEvidence::artifactSchema.identity);
    json.attribute(
        "schema_version",
        formatSchemaVersion(EvaluationEvidence::artifactSchema.version));
    json.attributeBegin("request_ref");
    writeArtifactRootReferenceJson(json, evidence.requestRef());
    json.attributeEnd();
    json.attributeBegin("output_bindings");
    writeOutputBindings(json, evidence.outputBindings());
    json.attributeEnd();
    json.attributeBegin("outcome");
    writeOutcome(json, evidence.outcome());
    json.attributeEnd();
  });
  return output.str().str();
}

llvm::Expected<EvaluationEvidence>
parseEvaluationEvidence(llvm::StringRef jsonText,
                        const CaseArtifactResolution &resolution,
                        const ArtifactStore &artifactStore) {
  auto value = llvm::json::parse(jsonText);
  if (!value)
    return value.takeError();
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return evaluationError("evaluation.evidence root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "evaluation.evidence root",
          {"schema", "schema_version", "request_ref", "output_bindings",
           "outcome"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "evaluation.evidence root");
  if (!schema)
    return schema.takeError();
  if (*schema != EvaluationEvidence::artifactSchema.identity)
    return evaluationError("unsupported EvaluationEvidence schema '" +
                           *schema + "'");
  auto versionSpelling =
      requireString(*root, "schema_version", "evaluation.evidence root");
  if (!versionSpelling)
    return versionSpelling.takeError();
  auto version = parseSchemaVersion(*versionSpelling);
  if (!version)
    return version.takeError();
  if (*version != EvaluationEvidence::artifactSchema.version)
    return evaluationError("unsupported evaluation.evidence version '" +
                           *versionSpelling + "'");

  auto requestObject =
      requireObject(*root, "request_ref", "evaluation.evidence root");
  if (!requestObject)
    return requestObject.takeError();
  auto requestRef = parseArtifactRootReferenceJson(**requestObject);
  if (!requestRef)
    return requestRef.takeError();
  auto request = importEvaluationRequest(*requestRef, resolution, artifactStore);
  if (!request)
    return request.takeError();
  auto outputs = parseOutputBindings(*root);
  if (!outputs)
    return outputs.takeError();
  auto outcome =
      parseOutcome(*root, *request, *outputs, resolution, artifactStore);
  if (!outcome)
    return outcome.takeError();
  auto evidence = EvaluationEvidence::get(
      *request, std::move(*outputs), std::move(*outcome), resolution,
      artifactStore);
  if (!evidence)
    return evidence.takeError();
  if (evidence->requestRef() != *requestRef)
    return evaluationError("EvaluationEvidence request reference changed");
  if (serializeEvaluationEvidence(*evidence) != jsonText)
    return evaluationError("EvaluationEvidence JSON is not canonical");
  return evidence;
}

} // namespace loom::evaluation
