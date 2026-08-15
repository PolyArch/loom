#include "DSE/StructuredExecutionShapeCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactStore.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StructuredExecutionShape.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_execution_shape_generator.config.1.0";

enum InputSlot : std::uint32_t {
  StructuredProgramsInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
         "structured_program", PlanValueRole::CandidateSet,
         &frontend::structuredProgramArtifactSchema,
         PlanValueCardinality::FiniteSet},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "structured_program",
     PlanValueRole::CandidateSet, &frontend::structuredProgramArtifactSchema,
     PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "execution_shape_decision"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "structured_execution_shape_generator_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedStructuredExecutionShapeGeneratorConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Error
validateDecisionPayload(llvm::ArrayRef<std::uint8_t> bytes,
                        llvm::ArrayRef<ArtifactRootReference> parents,
                        const ArtifactStore &store) {
  auto adopted = frontend::adoptStructuredExecutionShapeDecision(bytes);
  if (!adopted)
    return adopted.takeError();
  if (parents.size() != 1 ||
      parents.front().schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      parents.front().schemaVersion !=
          frontend::structuredProgramArtifactSchema.version)
    return invalid(
        "execution-shape decision does not have one exact Structured parent");
  auto parent = frontend::importStructuredProgram(parents.front(), store);
  if (!parent)
    return parent.takeError();
  auto domain = frontend::enumerateStructuredExecutionShapeDecisions(*parent);
  if (!domain)
    return domain.takeError();
  if (!llvm::is_contained(*domain, *adopted))
    return invalid(
        "execution-shape decision is outside the exact parent decision domain");
  return llvm::Error::success();
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    frontend::structuredExecutionShapeDecisionSchemaBytes(),
    validateDecisionPayload};

const CandidateGeneratorDescriptor descriptor{
    structuredExecutionShapeCandidateGeneratorKind,
    "compiler.structured_execution_shape",
    "loom.compiler.structured_execution_shape.generator.v2",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &lineageContract,
    ProviderForm::InProcess,
};

llvm::Error recordStructuredCandidate(
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    std::optional<frontend::StructuredExecutionShapeDecision> decision,
    frontend::MaterializedStructuredOwnershipCandidate candidate,
    const ArtifactStore &store) {
  StructuredOwnershipInvocation *invocation =
      detail::StructuredOwnershipInvocationAccess::current();
  if (!invocation)
    return llvm::Error::success();
  return detail::StructuredOwnershipInvocationAccess::
      recordExecutionShapeCandidate(*invocation, parent, child, decision,
                                    std::move(candidate), store);
}

llvm::Expected<frontend::MaterializedStructuredOwnershipCandidate>
cloneParentState(StructuredOwnershipInvocation *invocation,
                 const ArtifactRootReference &reference,
                 const ArtifactStore &store) {
  if (invocation)
    return detail::StructuredOwnershipInvocationAccess::
        clonePreClosureCandidate(*invocation, reference);
  auto clone = frontend::importStructuredProgram(reference, store);
  if (!clone)
    return clone.takeError();
  return frontend::MaterializedStructuredOwnershipCandidate{
      std::move(*clone), std::nullopt, {}, {}};
}

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &blobs,
               const CandidateGeneratorInvocationView &) {
  auto config = adoptResolvedStructuredExecutionShapeGeneratorConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();
  StructuredOwnershipInvocation *invocation =
      detail::StructuredOwnershipInvocationAccess::current();

  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
  std::uint64_t decisionAttempts = 0;
  for (const ArtifactRootReference &reference :
       inputBindings[StructuredProgramsInput].artifacts) {
    auto parent = cloneParentState(invocation, reference, store);
    if (!parent)
      return parent.takeError();
    auto decisions = frontend::enumerateStructuredExecutionShapeDecisions(
        parent->structuredProgram);
    if (!decisions)
      return decisions.takeError();
    if (decisions->size() >
        std::numeric_limits<std::uint64_t>::max() - decisionAttempts)
      return invalid("execution-shape accounting overflows u64");
    decisionAttempts += decisions->size();

    if (decisions->empty()) {
      if (llvm::Error error = recordStructuredCandidate(
              reference, reference, std::nullopt, std::move(*parent), store))
        return std::move(error);
      outputs.push_back(reference);
      continue;
    }

    for (auto item : llvm::enumerate(*decisions)) {
      const frontend::StructuredExecutionShapeDecision &decision = item.value();
      llvm::Expected<frontend::MaterializedStructuredOwnershipCandidate>
          candidate = item.index() == 0
                          ? std::move(parent)
                          : cloneParentState(invocation, reference, store);
      if (!candidate)
        return candidate.takeError();
      auto child = frontend::materializeStructuredExecutionShapeDecision(
          std::move(*candidate), decision);
      if (!child)
        return child.takeError();
      auto published =
          frontend::publishStructuredProgram(child->structuredProgram, store);
      if (!published)
        return published.takeError();
      if (llvm::Error error = recordStructuredCandidate(
              reference, *published, decision, std::move(*child), store))
        return std::move(error);
      auto ownerPayload =
          frontend::encodeStructuredExecutionShapeDecision(decision);
      if (!ownerPayload)
        return ownerPayload.takeError();
      lineageEdges.push_back(CandidateGeneratorLineageEdge{
          CandidateGeneratorLineageEdgeKind::CandidateDecision,
          CandidateGeneratorOutputSlotRef(0),
          *published,
          {reference},
          std::move(*ownerPayload)});
      outputs.push_back(std::move(*published));
    }
  }
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineageEdges)},
      {{CandidateGeneratorWorkUnitRef(0), decisionAttempts, decisionAttempts}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedStructuredExecutionShapeGeneratorConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedStructuredExecutionShapeGeneratorConfigView>
projectResolvedStructuredExecutionShapeGeneratorConfigView() {
  std::vector<std::uint8_t> bytes;
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedStructuredExecutionShapeGeneratorConfigView(
      std::move(bytes), std::move(*digest));
}

llvm::Expected<ResolvedStructuredExecutionShapeGeneratorConfigView>
adoptResolvedStructuredExecutionShapeGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (!canonicalViewBytes.empty())
    return invalid("schema-1.0 config must be empty");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  return ResolvedStructuredExecutionShapeGeneratorConfigView({}, digest);
}

const CandidateGeneratorDescriptor &
structuredExecutionShapeCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerStructuredExecutionShapeCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredExecutionShapeCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms) {
  if (llvm::Error error = registerStructuredExecutionShapeCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
       structuredPrograms.vec()},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveStructuredExecutionShapeCandidateGeneratorBinding(
    const ResolvedStructuredExecutionShapeGeneratorConfigView &config) {
  if (llvm::Error error = registerStructuredExecutionShapeCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
