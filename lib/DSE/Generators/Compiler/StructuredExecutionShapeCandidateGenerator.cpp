#include "DSE/StructuredExecutionShapeCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StructuredExecutionShape.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_execution_shape_generator.config.1.0";

enum InputSlot : std::uint32_t {
  StructuredProgramsInput,
  FabricInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
         "structured_program", PlanValueRole::CandidateSet,
         &frontend::structuredProgramArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
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

const CandidateGeneratorDescriptor descriptor{
    structuredExecutionShapeCandidateGeneratorKind,
    "compiler.structured_execution_shape",
    "loom.compiler.structured_execution_shape.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    {},
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<CandidateGeneratorInputBinding> bindings,
            InputSlot slot) {
  return bindings[slot].artifacts.front();
}

llvm::Expected<std::optional<frontend::MaterializedOwnershipCandidate>>
finalizeCandidate(
    frontend::MaterializedStructuredOwnershipCandidate candidate,
    const fabric::FinalizedFabricRoot &fabric,
    const lowering::CanonicalDataflowLoweringOptions &loweringOptions) {
  auto finalized = frontend::finalizeSpatialOwnershipCandidate(
      std::move(candidate), fabric, loweringOptions);
  if (finalized)
    return std::optional<frontend::MaterializedOwnershipCandidate>(
        std::move(*finalized));

  bool rejected = false;
  llvm::Error unhandled = llvm::handleErrors(
      finalized.takeError(),
      [&](const frontend::SpatialOwnershipCandidateRejection &) {
        rejected = true;
      });
  if (unhandled)
    return std::move(unhandled);
  if (!rejected)
    return invalid("candidate failed without a classified rejection");
  return std::optional<frontend::MaterializedOwnershipCandidate>{};
}

llvm::Error recordFinalizedCandidate(
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    std::optional<frontend::StructuredExecutionShapeDecision> decision,
    frontend::MaterializedOwnershipCandidate candidate,
    const ArtifactStore &store) {
  StructuredOwnershipInvocation *invocation =
      detail::StructuredOwnershipInvocationAccess::current();
  if (!invocation)
    return llvm::Error::success();
  lowering::ProjectedCanonicalDataflow projected{
      std::move(candidate.canonicalDataflow),
      std::move(candidate.spatialGraphs)};
  frontend::MaterializedStructuredOwnershipCandidate structured{
      std::move(candidate.structuredProgram),
      std::move(candidate.blockActivityLineage),
      std::move(candidate.sourceProvenance)};
  return detail::StructuredOwnershipInvocationAccess::
      recordExecutionShapeCandidate(*invocation, parent, child, decision,
                                    std::move(structured), std::move(projected),
                                    store);
}

llvm::Expected<frontend::MaterializedStructuredOwnershipCandidate>
cloneParentState(StructuredOwnershipInvocation *invocation,
                 const ArtifactRootReference &reference,
                 const ArtifactStore &store) {
  if (invocation)
    return detail::StructuredOwnershipInvocationAccess::cloneOwnershipCandidate(
        *invocation, reference);
  auto clone = frontend::importStructuredProgram(reference, store);
  if (!clone)
    return clone.takeError();
  return frontend::MaterializedStructuredOwnershipCandidate{
      std::move(*clone), {}, {}};
}

llvm::Expected<CandidateGeneratorInvocationOutcome>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store) {
  auto config = adoptResolvedStructuredExecutionShapeGeneratorConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();
  StructuredOwnershipInvocation *invocation =
      detail::StructuredOwnershipInvocationAccess::current();
  std::optional<fabric::FinalizedFabricRoot> importedFabric;
  const fabric::FinalizedFabricRoot *exactFabric = nullptr;
  if (invocation) {
    exactFabric =
        &detail::StructuredOwnershipInvocationAccess::fabric(*invocation);
    if (singleInput(inputBindings, FabricInput) != exactFabric->reference())
      return invalid("Fabric input differs from the bound invocation");
  } else {
    auto imported = fabric::importEntireFabricRoot(
        singleInput(inputBindings, FabricInput), store);
    if (!imported)
      return imported.takeError();
    importedFabric.emplace(std::move(*imported));
    exactFabric = &*importedFabric;
  }
  const lowering::CanonicalDataflowLoweringOptions loweringOptions =
      invocation ? detail::StructuredOwnershipInvocationAccess::loweringOptions(
                       *invocation)
                 : lowering::CanonicalDataflowLoweringOptions{};

  std::vector<ArtifactRootReference> outputs;
  for (const ArtifactRootReference &reference :
       inputBindings[StructuredProgramsInput].artifacts) {
    auto parent = cloneParentState(invocation, reference, store);
    if (!parent)
      return parent.takeError();
    auto decisions = frontend::enumerateStructuredExecutionShapeDecisions(
        parent->structuredProgram);
    if (!decisions)
      return decisions.takeError();

    if (decisions->empty()) {
      auto finalized =
          finalizeCandidate(std::move(*parent), *exactFabric, loweringOptions);
      if (!finalized)
        return finalized.takeError();
      if (!*finalized)
        continue;
      if (llvm::Error error =
              recordFinalizedCandidate(reference, reference, std::nullopt,
                                       std::move(**finalized), store))
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
      auto finalized =
          finalizeCandidate(std::move(*child), *exactFabric, loweringOptions);
      if (!finalized)
        return finalized.takeError();
      if (!*finalized)
        continue;
      auto published = frontend::publishStructuredProgram(
          (*finalized)->structuredProgram, store);
      if (!published)
        return published.takeError();
      if (llvm::Error error = recordFinalizedCandidate(
              reference, *published, decision, std::move(**finalized), store))
        return std::move(error);
      outputs.push_back(std::move(*published));
    }
  }
  return CandidateGeneratorInvocationOutcome{
      CompletedCandidateGeneratorInvocation{{
          {CandidateGeneratorOutputSlotRef(0), std::move(outputs)},
      }}};
}

const CandidateGeneratorProvider provider{descriptor.reference(),
                                          invokeProvider};

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
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error = registerStructuredExecutionShapeCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
       structuredPrograms.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
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
