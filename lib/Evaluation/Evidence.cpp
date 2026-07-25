#include "Evaluation/Evidence.h"

#include "CanonicalSupport.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::evaluationError;

FindingResultForm findingResultForm(const FindingResultValue &result) {
  if (std::holds_alternative<AbsentFinding>(result))
    return FindingResultForm::Absent;
  if (std::holds_alternative<PresentFinding>(result))
    return FindingResultForm::Present;
  return FindingResultForm::NotApplicable;
}

const ModelOutputBinding *
findOutputBinding(llvm::ArrayRef<ModelOutputBinding> bindings,
                  ModelOutputSlotRef slot) {
  for (const ModelOutputBinding &binding : bindings)
    if (binding.slot == slot)
      return &binding;
  return nullptr;
}

llvm::Error requireAvailable(const ArtifactRootReference &reference,
                             llvm::StringRef owner,
                             const ArtifactStore &artifactStore) {
  auto bytes = artifactStore.get(reference);
  if (!bytes)
    return llvm::joinErrors(evaluationError(owner + " is unresolved"),
                            bytes.takeError());
  return llvm::Error::success();
}

llvm::Expected<std::vector<ModelOutputBinding>> canonicalizeOutputBindings(
    const EvaluationRequest &request, EvidenceOutcomeKind outcome,
    std::vector<ModelOutputBinding> bindings,
    const ArtifactStore &artifactStore) {
  const EvaluationModelDescriptor *descriptor =
      resolveEvaluationModelDescriptor(request);
  if (!descriptor)
    return evaluationError("Evidence Request model descriptor is unresolved");

  std::sort(bindings.begin(), bindings.end(),
            [](const ModelOutputBinding &lhs, const ModelOutputBinding &rhs) {
              return lhs.slot < rhs.slot;
            });
  for (std::size_t index = 1; index < bindings.size(); ++index)
    if (bindings[index - 1].slot == bindings[index].slot)
      return evaluationError("duplicate model output-slot binding");
  if (bindings.size() != descriptor->outputSlots.size())
    return evaluationError(
        "Evidence output bindings are not total over descriptor output slots");

  for (std::size_t index = 0; index < bindings.size(); ++index) {
    ModelOutputBinding &binding = bindings[index];
    const ModelOutputSlotDescriptor &slot = descriptor->outputSlots[index];
    if (binding.slot != slot.slot)
      return evaluationError("Evidence output binding has a foreign slot "
                             "ordinal");
    std::sort(binding.artifacts.begin(), binding.artifacts.end(),
              artifactRootReferenceLess);
    for (std::size_t artifact = 1; artifact < binding.artifacts.size();
         ++artifact)
      if (binding.artifacts[artifact - 1] == binding.artifacts[artifact])
        return evaluationError("duplicate Artifact in model output binding");
    if (llvm::Error error = validateArtifactCollectionCardinality(
            slot.cardinality(outcome), binding.artifacts.size(),
            slot.semanticRole))
      return std::move(error);
    for (const ArtifactRootReference &artifact : binding.artifacts) {
      if (artifact.schemaIdentity != slot.schema->identity ||
          artifact.schemaVersion != slot.schema->version)
        return evaluationError("model output slot '" + slot.semanticRole +
                               "' rejects Artifact schema '" +
                               artifact.schemaIdentity + "'");
      if (llvm::Error error = requireAvailable(
              artifact, "model output reference", artifactStore))
        return std::move(error);
    }
  }
  return bindings;
}

llvm::Error validateMetricResult(
    MetricResult &result, MetricRequestOrdinal ordinal,
    const EvaluationRequest &request,
    const EvaluationModelDescriptor &descriptor) {
  const MetricRequest *requestItem = request.resolve(ordinal);
  const MetricCapability *capability =
      descriptor.findMetricCapability(requestItem->query().metric);
  if (!capability)
    return evaluationError("model does not support the metric result");
  const ObservationForm form = observationForm(result.observation);
  if ((capability->permittedObservationForms & observationFormMask(form)) == 0)
    return evaluationError("model does not permit the metric result form");
  if (llvm::Error error = validateMetricObservationValue(
          requestItem->query().metric, result.uncertainty,
          result.observation))
    return error;

  std::sort(result.calibrationInputSlots.begin(),
            result.calibrationInputSlots.end());
  for (std::size_t index = 0; index < result.calibrationInputSlots.size();
       ++index) {
    const ModelInputSlotRef slot = result.calibrationInputSlots[index];
    if (index != 0 && result.calibrationInputSlots[index - 1] == slot)
      return evaluationError("duplicate metric calibration input slot");
    if (!descriptor.findInputSlot(slot) ||
        !request.modelBinding().findInputBinding(slot))
      return evaluationError("metric result references a foreign calibration "
                             "input slot");
  }
  return llvm::Error::success();
}

llvm::Error validateFindingResult(
    FindingResult &result, FindingRequestOrdinal ordinal,
    const EvaluationRequest &request,
    const EvaluationModelDescriptor &model,
    llvm::ArrayRef<ModelOutputBinding> outputBindings,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore) {
  const FindingRequest *requestItem = request.resolve(ordinal);
  const FindingCapability *capability =
      model.findFindingCapability(requestItem->query().kind);
  if (!capability)
    return evaluationError("model does not support the finding result");
  const FindingResultForm form = findingResultForm(result.result);
  if ((capability->permittedResultForms & findingResultFormMask(form)) == 0)
    return evaluationError("model does not permit the finding result form");

  auto *present = std::get_if<PresentFinding>(&result.result);
  if (!present)
    return llvm::Error::success();
  if (present->occurrences.empty())
    return evaluationError("present finding requires at least one occurrence");

  const FindingDescriptor *finding =
      findFindingDescriptor(requestItem->query().kind);
  if (!finding)
    return evaluationError("finding result references an unregistered kind");
  if (finding->terminalWitnessSchema && present->occurrences.size() != 1)
    return evaluationError(
        "terminal finding Present requires exactly one occurrence");
  if (llvm::Error error = requireFindingOccurrenceOwner(*finding))
    return error;
  const FindingOccurrenceContext context(request, ordinal, outputBindings,
                                         resolution, artifactStore);
  for (FindingOccurrence &occurrence : present->occurrences)
    if (llvm::Error error =
            occurrence.canonicalize(finding->occurrenceCodec, context))
      return error;
  std::sort(present->occurrences.begin(), present->occurrences.end());
  for (std::size_t index = 1; index < present->occurrences.size(); ++index)
    if (present->occurrences[index - 1] == present->occurrences[index])
      return evaluationError("duplicate finding occurrence");
  return llvm::Error::success();
}

llvm::Expected<CompletedEvidence> canonicalizeCompleted(
    CompletedEvidence completed, const EvaluationRequest &request,
    const EvaluationModelDescriptor &descriptor,
    llvm::ArrayRef<ModelOutputBinding> outputBindings,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore) {
  if (completed.metricResults.size() != request.metricRequests().size())
    return evaluationError(
        "completed Evidence metric results are not total over metric requests");
  for (std::size_t index = 0; index < completed.metricResults.size(); ++index) {
    if (llvm::Error error = validateMetricResult(
            completed.metricResults[index], MetricRequestOrdinal(index),
            request, descriptor))
      return std::move(error);
  }

  if (completed.findingResults.size() != request.findingRequests().size())
    return evaluationError("completed Evidence finding results are not total "
                           "over finding requests");
  for (std::size_t index = 0; index < completed.findingResults.size(); ++index) {
    if (llvm::Error error = validateFindingResult(
            completed.findingResults[index], FindingRequestOrdinal(index),
            request, descriptor, outputBindings, resolution, artifactStore))
      return std::move(error);
  }
  return completed;
}

llvm::Error validateOutcomeReason(EvidenceOutcomeKind outcome,
                                  OutcomeReason reason) {
  bool valid = false;
  switch (outcome) {
  case EvidenceOutcomeKind::Completed:
    break;
  case EvidenceOutcomeKind::Unsupported:
    valid = reason == OutcomeReason::RuntimeCapabilityUnavailable;
    break;
  case EvidenceOutcomeKind::ExecutionFailed:
    valid = reason == OutcomeReason::ToolFailure ||
            reason == OutcomeReason::AdapterFailure ||
            reason == OutcomeReason::InfrastructureFailure;
    break;
  case EvidenceOutcomeKind::CancelledOrTimeout:
    valid = reason == OutcomeReason::ExternalCancellation ||
            reason == OutcomeReason::ExecutionLimitReached;
    break;
  }
  if (!valid)
    return evaluationError("OutcomeReason is invalid for Evidence outcome '" +
                           toString(outcome) + "'");
  return llvm::Error::success();
}

} // namespace

const ArtifactRootReference *
FindingOccurrenceContext::resolveOutput(ModelOutputSlotRef slot,
                                        std::uint64_t ordinal) const {
  const ModelOutputBinding *output = findOutputBinding(outputBindings_, slot);
  if (!output || ordinal >= output->artifacts.size())
    return nullptr;
  return &output->artifacts[ordinal];
}

llvm::Expected<FindingOccurrence>
FindingOccurrence::decode(const FindingOccurrenceCodec &codec,
                          llvm::ArrayRef<std::uint8_t> canonicalPayload,
                          const FindingOccurrenceContext &context) {
  auto occurrence = codec.decode(canonicalPayload);
  if (!occurrence)
    return occurrence.takeError();
  if (!*occurrence)
    return evaluationError(
        "finding occurrence decoder returned no owner-typed value");
  if (llvm::Error error = codec.validate(*occurrence, context))
    return std::move(error);
  auto reencoded = codec.encode(*occurrence);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalPayload)
    return evaluationError("finding occurrence payload is not canonical");
  return FindingOccurrence(std::move(*occurrence),
                           std::vector<std::uint8_t>(canonicalPayload.begin(),
                                                     canonicalPayload.end()));
}

llvm::Error
FindingOccurrence::canonicalize(const FindingOccurrenceCodec &codec,
                                const FindingOccurrenceContext &context) {
  if (!occurrence_)
    return evaluationError("finding occurrence has no owner-typed value");
  if (llvm::Error error = codec.validate(occurrence_, context))
    return error;
  auto encoded = codec.encode(occurrence_);
  if (!encoded)
    return encoded.takeError();
  if (!canonicalPayload_.empty() && canonicalPayload_ != *encoded)
    return evaluationError("finding occurrence payload is not canonical");
  canonicalPayload_ = std::move(*encoded);
  return llvm::Error::success();
}

std::string FindingOccurrence::canonicalHex() const {
  return formatArtifactLocalPayloadHex(canonicalPayload_);
}

const ArtifactSchemaDescriptor EvaluationEvidence::artifactSchema{
    "evaluation.evidence", {1, 0}};

llvm::StringRef toString(OutcomeReason reason) {
  switch (reason) {
  case OutcomeReason::RuntimeCapabilityUnavailable:
    return "runtime_capability_unavailable";
  case OutcomeReason::ToolFailure:
    return "tool_failure";
  case OutcomeReason::AdapterFailure:
    return "adapter_failure";
  case OutcomeReason::InfrastructureFailure:
    return "infrastructure_failure";
  case OutcomeReason::ExternalCancellation:
    return "external_cancellation";
  case OutcomeReason::ExecutionLimitReached:
    return "execution_limit_reached";
  }
  llvm_unreachable("unknown OutcomeReason");
}

llvm::Expected<OutcomeReason> parseOutcomeReason(llvm::StringRef spelling) {
  if (spelling == "runtime_capability_unavailable")
    return OutcomeReason::RuntimeCapabilityUnavailable;
  if (spelling == "tool_failure")
    return OutcomeReason::ToolFailure;
  if (spelling == "adapter_failure")
    return OutcomeReason::AdapterFailure;
  if (spelling == "infrastructure_failure")
    return OutcomeReason::InfrastructureFailure;
  if (spelling == "external_cancellation")
    return OutcomeReason::ExternalCancellation;
  if (spelling == "execution_limit_reached")
    return OutcomeReason::ExecutionLimitReached;
  return evaluationError("unknown OutcomeReason '" + spelling + "'");
}

EvidenceOutcomeKind outcomeKind(const EvaluationEvidenceOutcome &outcome) {
  return static_cast<EvidenceOutcomeKind>(outcome.index());
}

llvm::Expected<EvaluationEvidence>
EvaluationEvidence::get(
    const EvaluationRequest &request,
    std::vector<ModelOutputBinding> outputBindings,
    EvaluationEvidenceOutcome outcome,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore) {
  const ArtifactRootReference requestRef = evaluationRequestReference(request);
  if (llvm::Error error =
          requireAvailable(requestRef, "EvaluationRequest reference",
                           artifactStore))
    return std::move(error);
  const EvaluationModelDescriptor *descriptor =
      resolveEvaluationModelDescriptor(request);
  if (!descriptor)
    return evaluationError("Evidence Request model descriptor is unresolved");

  const EvidenceOutcomeKind kind = evaluation::outcomeKind(outcome);
  auto canonicalOutputs = canonicalizeOutputBindings(
      request, kind, std::move(outputBindings), artifactStore);
  if (!canonicalOutputs)
    return canonicalOutputs.takeError();

  if (auto *completed = std::get_if<CompletedEvidence>(&outcome)) {
    auto canonical = canonicalizeCompleted(
        std::move(*completed), request, *descriptor, *canonicalOutputs,
        resolution, artifactStore);
    if (!canonical)
      return canonical.takeError();
    outcome = std::move(*canonical);
  } else if (const auto *unsupported =
                 std::get_if<UnsupportedEvidence>(&outcome)) {
    if (llvm::Error error =
            validateOutcomeReason(kind, unsupported->reason))
      return std::move(error);
  } else if (const auto *failed =
                 std::get_if<ExecutionFailedEvidence>(&outcome)) {
    if (llvm::Error error = validateOutcomeReason(kind, failed->reason))
      return std::move(error);
  } else {
    const auto &cancelled = std::get<CancelledOrTimeoutEvidence>(outcome);
    if (llvm::Error error = validateOutcomeReason(kind, cancelled.reason))
      return std::move(error);
  }

  return EvaluationEvidence(requestRef, std::move(*canonicalOutputs),
                            std::move(outcome));
}

CanonicalSemanticBytes
canonicalEvaluationEvidenceBytes(const EvaluationEvidence &evidence) {
  const std::string json = serializeEvaluationEvidence(evidence);
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));
}

ArtifactIdentity evaluationEvidenceIdentity(const EvaluationEvidence &evidence) {
  return finalizeArtifactIdentity(EvaluationEvidence::artifactSchema,
                                  canonicalEvaluationEvidenceBytes(evidence));
}

ArtifactRootReference
evaluationEvidenceReference(const EvaluationEvidence &evidence) {
  return ArtifactRootReference{
      EvaluationEvidence::artifactSchema.identity.str(),
      EvaluationEvidence::artifactSchema.version,
      evaluationEvidenceIdentity(evidence)};
}

llvm::Expected<ArtifactRootReference>
publishEvaluationEvidence(const EvaluationEvidence &evidence,
                          const ArtifactStore &artifactStore) {
  auto identity = artifactStore.put(EvaluationEvidence::artifactSchema,
                                    canonicalEvaluationEvidenceBytes(evidence));
  if (!identity)
    return identity.takeError();
  if (*identity != evaluationEvidenceIdentity(evidence))
    return evaluationError(
        "ArtifactStore returned a foreign EvaluationEvidence identity");
  return ArtifactRootReference{
      EvaluationEvidence::artifactSchema.identity.str(),
      EvaluationEvidence::artifactSchema.version, std::move(*identity)};
}

llvm::Expected<EvaluationEvidence>
importEvaluationEvidence(const ArtifactRootReference &reference,
                         const CaseArtifactResolution &resolution,
                         const ArtifactStore &artifactStore) {
  if (reference.schemaIdentity != EvaluationEvidence::artifactSchema.identity ||
      reference.schemaVersion != EvaluationEvidence::artifactSchema.version)
    return evaluationError("foreign EvaluationEvidence reference schema");
  auto bytes = artifactStore.get(EvaluationEvidence::artifactSchema,
                                 reference.artifact);
  if (!bytes)
    return bytes.takeError();
  const llvm::ArrayRef<std::uint8_t> payload = bytes->bytes();
  llvm::StringRef json(reinterpret_cast<const char *>(payload.data()),
                       payload.size());
  auto evidence = parseEvaluationEvidence(json, resolution, artifactStore);
  if (!evidence)
    return evidence.takeError();
  if (evaluationEvidenceIdentity(*evidence) != reference.artifact)
    return evaluationError("stale EvaluationEvidence reference identity");
  return evidence;
}

} // namespace loom::evaluation
