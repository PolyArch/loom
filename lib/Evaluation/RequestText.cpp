#include "Evaluation/Request.h"

#include "CanonicalSupport.h"
#include "Evaluation/CaseText.h"
#include "Evaluation/ConditionText.h"
#include "QueryText.h"

#include "Common/ArtifactText.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <optional>
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

void writeOptionalRootReference(
    llvm::json::OStream &json,
    const std::optional<ArtifactRootReference> &reference) {
  if (!reference) {
    json.value(nullptr);
    return;
  }
  writeArtifactRootReferenceJson(json, *reference);
}

llvm::Expected<std::optional<ArtifactRootReference>>
parseOptionalRootReference(const llvm::json::Object &object,
                           llvm::StringRef key, llvm::StringRef context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return evaluationError(context + " field '" + key + "' is required");
  if (value->getAsNull())
    return std::optional<ArtifactRootReference>{};
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return evaluationError(context + " field '" + key +
                           "' must be null or an object");
  auto parsed = parseArtifactRootReferenceJson(*root);
  if (!parsed)
    return parsed.takeError();
  return std::optional<ArtifactRootReference>{std::move(*parsed)};
}

void writeConditions(llvm::json::OStream &json,
                     llvm::ArrayRef<EvaluationCondition> conditions) {
  json.array([&] {
    for (const EvaluationCondition &condition : conditions)
      writeEvaluationConditionJson(json, condition);
  });
}

llvm::Expected<std::vector<EvaluationCondition>>
parseConditions(const llvm::json::Object &object, llvm::StringRef key,
                llvm::StringRef context) {
  auto array = requireArray(object, key, context);
  if (!array)
    return array.takeError();
  std::vector<EvaluationCondition> conditions;
  conditions.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    auto condition = parseEvaluationConditionJson(value);
    if (!condition)
      return condition.takeError();
    conditions.push_back(std::move(*condition));
  }
  return conditions;
}

void writeSubjectBindings(llvm::json::OStream &json,
                          const EvaluationSubjectBindings &bindings) {
  json.array([&] {
    for (const CaseRoleBinding &binding : bindings.roleBindings()) {
      json.object([&] {
        json.attribute("role", binding.role.ordinal());
        json.attributeArray("subjects", [&] {
          for (const ArtifactRootReference &subject : binding.subjects)
            writeArtifactRootReferenceJson(json, subject);
        });
      });
    }
  });
}

llvm::Expected<EvaluationSubjectBindings>
parseSubjectBindings(const llvm::json::Object &root) {
  auto array =
      requireArray(root, "subject_bindings", "evaluation.request root");
  if (!array)
    return array.takeError();
  std::vector<CaseRoleBinding> bindings;
  bindings.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return evaluationError("subject binding must be an object");
    if (llvm::Error error = rejectUnknownFields(*object, "subject binding",
                                                {"role", "subjects"}))
      return std::move(error);
    auto role = requireOrdinal(*object, "role", "subject binding");
    if (!role)
      return role.takeError();
    auto subjects = requireArray(*object, "subjects", "subject binding");
    if (!subjects)
      return subjects.takeError();
    CaseRoleBinding binding{CaseSubjectRoleRef(*role), {}};
    binding.subjects.reserve((*subjects)->size());
    for (const llvm::json::Value &subjectValue : **subjects) {
      const llvm::json::Object *subject = subjectValue.getAsObject();
      if (!subject)
        return evaluationError("bound subject must be an object");
      auto reference = parseArtifactRootReferenceJson(*subject);
      if (!reference)
        return reference.takeError();
      binding.subjects.push_back(std::move(*reference));
    }
    bindings.push_back(std::move(binding));
  }
  return EvaluationSubjectBindings::get(std::move(bindings));
}

void writeModelBinding(llvm::json::OStream &json,
                       const ResolvedModelBinding &binding) {
  json.object([&] {
    json.attributeBegin("descriptor_ref");
    json.object([&] {
      json.attribute("schema_major",
                     binding.descriptorRef().schemaVersion().major);
      json.attribute("schema_minor",
                     binding.descriptorRef().schemaVersion().minor);
      json.attribute("model_kind",
                     binding.descriptorRef().modelKind().ordinal());
    });
    json.attributeEnd();
    json.attributeArray("input_bindings", [&] {
      for (const ModelInputBinding &input : binding.inputBindings()) {
        json.object([&] {
          json.attribute("slot", input.slot.ordinal());
          json.attributeArray("artifacts", [&] {
            for (const ArtifactRootReference &artifact : input.artifacts)
              writeArtifactRootReferenceJson(json, artifact);
          });
        });
      }
    });
    json.attributeObject("resolved_model_config", [&] {
      json.attribute("canonical_view_bytes",
                     formatArtifactLocalPayloadHex(
                         binding.resolvedModelConfig().canonicalViewBytes()));
      json.attribute(
          "component_view_digest",
          formatComponentViewDigestHex(binding.resolvedModelConfig().digest()));
    });
  });
}

llvm::Expected<ResolvedModelBinding>
parseModelBinding(const llvm::json::Object &root) {
  auto object = requireObject(root, "model_binding", "evaluation.request root");
  if (!object)
    return object.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **object, "model binding",
          {"descriptor_ref", "input_bindings", "resolved_model_config"}))
    return std::move(error);
  auto descriptorObject =
      requireObject(**object, "descriptor_ref", "model binding");
  if (!descriptorObject)
    return descriptorObject.takeError();
  if (llvm::Error error =
          rejectUnknownFields(**descriptorObject, "model descriptor reference",
                              {"schema_major", "schema_minor", "model_kind"}))
    return std::move(error);
  auto schemaMajor = requireUnsigned(**descriptorObject, "schema_major",
                                     "model descriptor reference");
  if (!schemaMajor)
    return schemaMajor.takeError();
  auto schemaMinor = requireUnsigned(**descriptorObject, "schema_minor",
                                     "model descriptor reference");
  if (!schemaMinor)
    return schemaMinor.takeError();
  if (*schemaMajor > std::numeric_limits<std::uint32_t>::max() ||
      *schemaMinor > std::numeric_limits<std::uint32_t>::max())
    return evaluationError(
        "model descriptor schema version components must be uint32");
  auto modelKind = requireOrdinal(**descriptorObject, "model_kind",
                                  "model descriptor reference");
  if (!modelKind)
    return modelKind.takeError();
  auto descriptorRef = EvaluationModelDescriptorRef::get(
      SchemaVersion{static_cast<std::uint32_t>(*schemaMajor),
                    static_cast<std::uint32_t>(*schemaMinor)},
      EvaluationModelKind(*modelKind));
  if (!descriptorRef)
    return descriptorRef.takeError();

  auto inputArray = requireArray(**object, "input_bindings", "model binding");
  if (!inputArray)
    return inputArray.takeError();
  std::vector<ModelInputBinding> inputs;
  inputs.reserve((*inputArray)->size());
  for (const llvm::json::Value &inputValue : **inputArray) {
    const llvm::json::Object *input = inputValue.getAsObject();
    if (!input)
      return evaluationError("model input binding must be an object");
    if (llvm::Error error = rejectUnknownFields(*input, "model input binding",
                                                {"slot", "artifacts"}))
      return std::move(error);
    auto slot = requireOrdinal(*input, "slot", "model input binding");
    if (!slot)
      return slot.takeError();
    auto artifacts = requireArray(*input, "artifacts", "model input binding");
    if (!artifacts)
      return artifacts.takeError();
    ModelInputBinding binding{ModelInputSlotRef(*slot), {}};
    binding.artifacts.reserve((*artifacts)->size());
    for (const llvm::json::Value &artifactValue : **artifacts) {
      const llvm::json::Object *artifact = artifactValue.getAsObject();
      if (!artifact)
        return evaluationError("model input artifact must be an object");
      auto reference = parseArtifactRootReferenceJson(*artifact);
      if (!reference)
        return reference.takeError();
      binding.artifacts.push_back(std::move(*reference));
    }
    inputs.push_back(std::move(binding));
  }

  auto config =
      requireObject(**object, "resolved_model_config", "model binding");
  if (!config)
    return config.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **config, "resolved model config",
          {"canonical_view_bytes", "component_view_digest"}))
    return std::move(error);
  auto viewBytes =
      requireString(**config, "canonical_view_bytes", "resolved model config");
  if (!viewBytes)
    return viewBytes.takeError();
  auto canonicalViewBytes = parseArtifactLocalPayloadHex(*viewBytes);
  if (!canonicalViewBytes)
    return canonicalViewBytes.takeError();
  auto digestSpelling =
      requireString(**config, "component_view_digest", "resolved model config");
  if (!digestSpelling)
    return digestSpelling.takeError();
  auto digest = parseComponentViewDigestHex(*digestSpelling);
  if (!digest)
    return digest.takeError();
  return ResolvedModelBinding::adopt(*descriptorRef, std::move(inputs),
                                     std::move(*canonicalViewBytes), *digest);
}

llvm::Expected<std::vector<MetricRequest>>
parseMetricRequests(const llvm::json::Object &root,
                    const EvaluationCase &evaluationCase,
                    const CaseArtifactResolution &resolution,
                    const ArtifactStore &artifactStore) {
  auto array = requireArray(root, "metric_requests", "evaluation.request root");
  if (!array)
    return array.takeError();
  std::vector<MetricRequest> requests;
  requests.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return evaluationError("metric request must be an object");
    if (llvm::Error error = rejectUnknownFields(*object, "metric request",
                                                {"query", "conditions"}))
      return std::move(error);
    auto queryObject = requireObject(*object, "query", "metric request");
    if (!queryObject)
      return queryObject.takeError();
    auto query =
        detail::parseMetricQueryPayload(**queryObject, "metric request query");
    if (!query)
      return query.takeError();
    auto conditions = parseConditions(*object, "conditions", "metric request");
    if (!conditions)
      return conditions.takeError();
    auto request =
        MetricRequest::get(std::move(*query), *conditions, evaluationCase,
                           resolution, artifactStore);
    if (!request)
      return request.takeError();
    requests.push_back(std::move(*request));
  }
  return requests;
}

llvm::Expected<std::vector<FindingRequest>>
parseFindingRequests(const llvm::json::Object &root,
                     const EvaluationCase &evaluationCase,
                     const CaseArtifactResolution &resolution,
                     const ArtifactStore &artifactStore) {
  auto array =
      requireArray(root, "finding_requests", "evaluation.request root");
  if (!array)
    return array.takeError();
  std::vector<FindingRequest> requests;
  requests.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return evaluationError("finding request must be an object");
    if (llvm::Error error = rejectUnknownFields(*object, "finding request",
                                                {"query", "conditions"}))
      return std::move(error);
    auto queryObject = requireObject(*object, "query", "finding request");
    if (!queryObject)
      return queryObject.takeError();
    auto query = detail::parseFindingQueryPayload(**queryObject,
                                                  "finding request query");
    if (!query)
      return query.takeError();
    auto conditions = parseConditions(*object, "conditions", "finding request");
    if (!conditions)
      return conditions.takeError();
    auto request =
        FindingRequest::get(std::move(*query), *conditions, evaluationCase,
                            resolution, artifactStore);
    if (!request)
      return request.takeError();
    requests.push_back(std::move(*request));
  }
  return requests;
}

} // namespace

std::string serializeResolvedModelBinding(const ResolvedModelBinding &binding) {
  llvm::SmallString<512> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attributeBegin("model_binding");
    writeModelBinding(json, binding);
    json.attributeEnd();
  });
  return output.str().str();
}

llvm::Expected<ResolvedModelBinding>
parseResolvedModelBinding(llvm::StringRef jsonText) {
  auto value = llvm::json::parse(jsonText);
  if (!value)
    return value.takeError();
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return evaluationError("resolved model binding root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "resolved model binding root", {"model_binding"}))
    return std::move(error);
  auto binding = parseModelBinding(*root);
  if (!binding)
    return binding.takeError();
  if (serializeResolvedModelBinding(*binding) != jsonText)
    return evaluationError("resolved model binding JSON is not canonical");
  return binding;
}

std::string
serializeEvaluationConditions(llvm::ArrayRef<EvaluationCondition> conditions) {
  llvm::SmallString<512> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attributeBegin("conditions");
    writeConditions(json, conditions);
    json.attributeEnd();
  });
  return output.str().str();
}

llvm::Expected<std::vector<EvaluationCondition>>
parseEvaluationConditions(llvm::StringRef jsonText) {
  auto value = llvm::json::parse(jsonText);
  if (!value)
    return value.takeError();
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return evaluationError("evaluation conditions root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "evaluation conditions root", {"conditions"}))
    return std::move(error);
  auto conditions =
      parseConditions(*root, "conditions", "evaluation conditions root");
  if (!conditions)
    return conditions.takeError();
  if (serializeEvaluationConditions(*conditions) != jsonText)
    return evaluationError("evaluation conditions JSON is not canonical");
  return conditions;
}

std::string serializeEvaluationRequest(const EvaluationRequest &request) {
  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", EvaluationRequest::artifactSchema.identity);
    json.attribute(
        "schema_version",
        formatSchemaVersion(EvaluationRequest::artifactSchema.version));
    json.attributeBegin("subject_bindings");
    writeSubjectBindings(json, request.subjectBindings());
    json.attributeEnd();
    json.attributeBegin("workload_ref");
    writeOptionalRootReference(json, request.workload());
    json.attributeEnd();
    json.attributeBegin("runtime_input_ref");
    writeOptionalRootReference(json, request.runtimeInput());
    json.attributeEnd();
    json.attributeBegin("base_conditions");
    writeConditions(json, request.baseConditions());
    json.attributeEnd();
    json.attributeArray("metric_requests", [&] {
      for (const MetricRequest &metric : request.metricRequests()) {
        json.object([&] {
          json.attributeBegin("query");
          json.object(
              [&] { detail::writeMetricQueryPayload(json, metric.query()); });
          json.attributeEnd();
          json.attributeBegin("conditions");
          writeConditions(json, metric.conditions());
          json.attributeEnd();
        });
      }
    });
    json.attributeArray("finding_requests", [&] {
      for (const FindingRequest &finding : request.findingRequests()) {
        json.object([&] {
          json.attributeBegin("query");
          json.object(
              [&] { detail::writeFindingQueryPayload(json, finding.query()); });
          json.attributeEnd();
          json.attributeBegin("conditions");
          writeConditions(json, finding.conditions());
          json.attributeEnd();
        });
      }
    });
    json.attributeBegin("model_binding");
    writeModelBinding(json, request.modelBinding());
    json.attributeEnd();
    json.attribute("replicate_index", request.replicateIndex());
  });
  return output.str().str();
}

llvm::Expected<EvaluationRequest>
parseEvaluationRequest(llvm::StringRef jsonText,
                       const CaseArtifactResolution &resolution,
                       const ArtifactStore &artifactStore) {
  auto value = llvm::json::parse(jsonText);
  if (!value)
    return value.takeError();
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return evaluationError("evaluation.request root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "evaluation.request root",
          {"schema", "schema_version", "subject_bindings", "workload_ref",
           "runtime_input_ref", "base_conditions", "metric_requests",
           "finding_requests", "model_binding", "replicate_index"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "evaluation.request root");
  if (!schema)
    return schema.takeError();
  if (*schema != EvaluationRequest::artifactSchema.identity)
    return evaluationError("unsupported EvaluationRequest schema '" + *schema +
                           "'");
  auto versionSpelling =
      requireString(*root, "schema_version", "evaluation.request root");
  if (!versionSpelling)
    return versionSpelling.takeError();
  auto version = parseSchemaVersion(*versionSpelling);
  if (!version)
    return version.takeError();
  if (*version != EvaluationRequest::artifactSchema.version)
    return evaluationError("unsupported evaluation.request version '" +
                           *versionSpelling + "'");

  auto modelBinding = parseModelBinding(*root);
  if (!modelBinding)
    return modelBinding.takeError();
  const EvaluationModelDescriptor *descriptor =
      modelBinding->descriptorRef().descriptor();
  if (!descriptor)
    return evaluationError("EvaluationRequest model descriptor is unresolved");
  auto subjectBindings = parseSubjectBindings(*root);
  if (!subjectBindings)
    return subjectBindings.takeError();
  auto workload = parseOptionalRootReference(*root, "workload_ref",
                                             "evaluation.request root");
  if (!workload)
    return workload.takeError();
  auto runtimeInput = parseOptionalRootReference(*root, "runtime_input_ref",
                                                 "evaluation.request root");
  if (!runtimeInput)
    return runtimeInput.takeError();
  auto baseConditions =
      parseConditions(*root, "base_conditions", "evaluation.request root");
  if (!baseConditions)
    return baseConditions.takeError();
  auto evaluationCase = EvaluationCase::get(
      descriptor->caseSignature, *subjectBindings, *workload, *runtimeInput,
      *baseConditions, resolution, artifactStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto metrics =
      parseMetricRequests(*root, *evaluationCase, resolution, artifactStore);
  if (!metrics)
    return metrics.takeError();
  auto findings =
      parseFindingRequests(*root, *evaluationCase, resolution, artifactStore);
  if (!findings)
    return findings.takeError();
  auto replicate =
      requireUnsigned(*root, "replicate_index", "evaluation.request root");
  if (!replicate)
    return replicate.takeError();

  auto request = EvaluationRequest::get(*evaluationCase, *metrics, *findings,
                                        std::move(*modelBinding), *replicate,
                                        resolution, artifactStore);
  if (!request)
    return request.takeError();
  if (serializeEvaluationRequest(*request) != jsonText)
    return evaluationError("EvaluationRequest JSON is not canonical");
  return request;
}

} // namespace loom::evaluation
