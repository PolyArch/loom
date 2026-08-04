#include "DSE/StructuredEvaluationAcquisition.h"

#include "Common/ArtifactStore.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_evaluation_acquisition.config.1.0";

enum InputSlot : std::uint32_t {
  CandidateInput,
  FabricInput,
  WorkloadInput,
  RuntimeInput,
  InputSlotCount,
};

constexpr std::array<PromotionAcquisitionInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {PromotionAcquisitionInputSlotRef(CandidateInput), "candidates",
         PlanValueRole::CandidateSet,
         &frontend::structuredProgramArtifactSchema,
         PlanValueCardinality::NonEmptySet},
        {PromotionAcquisitionInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {PromotionAcquisitionInputSlotRef(WorkloadInput), "workload",
         PlanValueRole::CandidateSet, &sim::simulationWorkloadSchema,
         PlanValueCardinality::ExactlyOne},
        {PromotionAcquisitionInputSlotRef(RuntimeInput), "runtime_input",
         PlanValueRole::CandidateSet, &sim::simulationRuntimeInputSchema,
         PlanValueCardinality::ExactlyOne},
    }};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_evaluation_acquisition_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

llvm::Expected<std::uint32_t> readU32(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - offset < 4)
    return invalid("truncated u32 field");
  std::uint32_t value = 0;
  for (unsigned ordinal = 0; ordinal != 4; ++ordinal)
    value = (value << 8) | bytes[offset++];
  return value;
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - offset < 8)
    return invalid("truncated u64 field");
  std::uint64_t value = 0;
  for (unsigned ordinal = 0; ordinal != 8; ++ordinal)
    value = (value << 8) | bytes[offset++];
  return value;
}

llvm::Expected<std::vector<EvidenceObligationTemplateRef>>
canonicalRefs(llvm::ArrayRef<EvidenceObligationTemplateRef> references) {
  std::vector<EvidenceObligationTemplateRef> canonical(references.begin(),
                                                       references.end());
  llvm::sort(canonical, [](EvidenceObligationTemplateRef lhs,
                           EvidenceObligationTemplateRef rhs) {
    return lhs.ordinal() < rhs.ordinal();
  });
  if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end())
    return invalid("Evidence obligation set contains a duplicate reference");
  return canonical;
}

std::vector<std::uint8_t>
encodeConfig(llvm::ArrayRef<EvidenceObligationTemplateRef> references) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(8 + references.size() * 4);
  appendU64(bytes, references.size());
  for (EvidenceObligationTemplateRef reference : references)
    appendU32(bytes, reference.ordinal());
  return bytes;
}

llvm::Expected<std::vector<EvidenceObligationTemplateRef>>
decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  std::size_t offset = 0;
  auto count = readU64(bytes, offset);
  if (!count)
    return count.takeError();
  if (*count > (bytes.size() - offset) / 4 ||
      *count > std::numeric_limits<std::size_t>::max())
    return invalid("Evidence obligation count exceeds remaining bytes");
  std::vector<EvidenceObligationTemplateRef> references;
  references.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t index = 0; index != *count; ++index) {
    auto ordinal = readU32(bytes, offset);
    if (!ordinal)
      return ordinal.takeError();
    references.emplace_back(*ordinal);
  }
  if (offset != bytes.size())
    return invalid("config has trailing bytes");
  auto canonical = canonicalRefs(references);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != references)
    return invalid("Evidence obligation references are not canonical");
  return references;
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedStructuredEvaluationAcquisitionConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Expected<std::vector<EvidenceObligationTemplateRef>>
resolveEvidenceObligations(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeConfig(bytes);
}

const PromotionAcquisitionDescriptor descriptor{
    structuredEvaluationPromotionAcquisitionKind,
    "compiler.structured_evaluation",
    "loom.compiler.structured_evaluation.acquisition.v1",
    inputSlots,
    PromotionAcquisitionInputSlotRef(CandidateInput),
    evaluation::CaseSubjectRoleRef(0),
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    &resolveEvidenceObligations,
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<PromotionAcquisitionInputBinding> bindings,
            InputSlot slot) {
  return bindings[slot].artifacts.front();
}

llvm::Expected<PromotionAcquisitionResolutionOutcome>
resolveCases(const ResolvedPromotionAcquisitionBinding &,
             llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings,
             llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
             const ArtifactStore &store) {
  auto invocation =
      evaluation::models::prepareStructuredFabricAnalyticInvocation(
          inputBindings[CandidateInput].artifacts,
          singleInput(inputBindings, FabricInput),
          singleInput(inputBindings, WorkloadInput),
          singleInput(inputBindings, RuntimeInput), store);
  if (!invocation)
    return invocation.takeError();
  auto resolution = std::make_shared<const evaluation::CaseArtifactResolution>(
      invocation->caseResolution());

  const evaluation::EvaluationModelDescriptorRef analytic =
      evaluation::models::structuredFabricAnalyticModelDescriptorRef();
  const evaluation::EvaluationModelDescriptorRef functional =
      evaluation::models::structuredProgramFunctionalModelDescriptorRef();
  std::vector<ResolvedPromotionEvidenceAcquisitionTask> resolved;
  resolved.reserve(tasks.size());
  for (const PromotionEvidenceAcquisitionTask &task : tasks) {
    if (!task.obligation)
      return invalid("task has no Evidence obligation");
    const auto model = task.obligation->modelBinding().descriptorRef();
    if (model != analytic && model != functional)
      return invalid("task references a non-Structured Evaluation model");
    if (!llvm::binary_search(inputBindings[CandidateInput].artifacts,
                             task.candidate, artifactRootReferenceLess))
      return invalid("task candidate is outside the bound candidate set");
    resolved.push_back({0, resolution});
  }
  return PromotionAcquisitionResolutionOutcome{
      CompletedPromotionAcquisitionResolution{std::move(resolved)}};
}

const PromotionAcquisitionProvider provider{descriptor.reference(),
                                            &resolveCases};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedStructuredEvaluationAcquisitionConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedStructuredEvaluationAcquisitionConfigView>
projectResolvedStructuredEvaluationAcquisitionConfigView(
    llvm::ArrayRef<EvidenceObligationTemplateRef> evidenceObligations) {
  auto canonical = canonicalRefs(evidenceObligations);
  if (!canonical)
    return canonical.takeError();
  std::vector<std::uint8_t> bytes = encodeConfig(*canonical);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedStructuredEvaluationAcquisitionConfigView(
      std::move(*canonical), std::move(bytes), std::move(*digest));
}

llvm::Expected<ResolvedStructuredEvaluationAcquisitionConfigView>
adoptResolvedStructuredEvaluationAcquisitionConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto references = decodeConfig(canonicalViewBytes);
  if (!references)
    return references.takeError();
  std::vector<std::uint8_t> reencoded = encodeConfig(*references);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("decoded config does not re-encode to the source bytes");
  return ResolvedStructuredEvaluationAcquisitionConfigView(
      std::move(*references), std::move(reencoded), digest);
}

const PromotionAcquisitionDescriptor &
structuredEvaluationPromotionAcquisitionDescriptor() {
  return descriptor;
}

llvm::Error registerStructuredEvaluationPromotionAcquisition() {
  if (llvm::Error error =
          evaluation::models::registerStructuredFabricAnalyticModel())
    return error;
  if (llvm::Error error =
          evaluation::models::registerStructuredProgramFunctionalModel())
    return error;
  if (evaluation::models::structuredFabricAnalyticCandidateRole() !=
      evaluation::models::structuredProgramFunctionalCandidateRole())
    return invalid("Structured models disagree on the candidate role");
  if (evaluation::models::structuredFabricAnalyticCandidateRole() !=
      descriptor.candidateRole)
    return invalid("descriptor candidate role differs from the model owner");
  if (llvm::Error error = registerPromotionAcquisitionDescriptor(descriptor))
    return error;
  return registerPromotionAcquisitionProvider(provider);
}

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindStructuredEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput) {
  if (llvm::Error error = registerStructuredEvaluationPromotionAcquisition())
    return std::move(error);
  std::vector<ArtifactRootReference> candidates(structuredPrograms.begin(),
                                                structuredPrograms.end());
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  std::vector<PromotionAcquisitionInputBinding> bindings = {
      {PromotionAcquisitionInputSlotRef(CandidateInput), std::move(candidates)},
      {PromotionAcquisitionInputSlotRef(FabricInput), {fabricReference}},
      {PromotionAcquisitionInputSlotRef(WorkloadInput), {workload}},
      {PromotionAcquisitionInputSlotRef(RuntimeInput), {runtimeInput}},
  };
  if (llvm::Error error = validatePromotionAcquisitionInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedPromotionAcquisitionBinding>
resolveStructuredEvaluationPromotionAcquisitionBinding(
    const ResolvedStructuredEvaluationAcquisitionConfigView &config) {
  if (llvm::Error error = registerStructuredEvaluationPromotionAcquisition())
    return std::move(error);
  return ResolvedPromotionAcquisitionBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

llvm::Expected<EvidenceObligationTemplate>
prepareStructuredFabricAnalyticEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store) {
  if (llvm::Error error = registerStructuredEvaluationPromotionAcquisition())
    return std::move(error);
  auto prepared = evaluation::models::prepareStructuredFabricEvaluation(
      prototypeCandidate, fabricReference, workload, runtimeInput, config,
      store);
  if (!prepared)
    return prepared.takeError();
  return EvidenceObligationTemplate::get(
      prepared->request, prepared->candidateRole,
      {{evaluation::models::structuredFabricAnalyticFabricRole(),
        EvidenceAcquisitionInputSlotRef(FabricInput)}});
}

llvm::Expected<EvidenceObligationTemplate>
prepareStructuredProgramFunctionalEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store) {
  if (llvm::Error error = registerStructuredEvaluationPromotionAcquisition())
    return std::move(error);
  auto prepared =
      evaluation::models::prepareStructuredProgramFunctionalEvaluation(
          prototypeCandidate, workload, runtimeInput, config, store);
  if (!prepared)
    return prepared.takeError();
  return EvidenceObligationTemplate::get(prepared->request,
                                         prepared->candidateRole, {});
}

} // namespace loom::dse
