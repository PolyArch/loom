#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.dataflow_rewrite_generator.config.1.0";

enum InputSlot : std::uint32_t {
  CanonicalDataflowProgramsInput,
  FabricInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(CanonicalDataflowProgramsInput),
         "canonical_dataflow", PlanValueRole::CandidateSet,
         &dataflow::canonicalDataflowSchema, PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "canonical_dataflow",
      PlanValueRole::CandidateSet, &dataflow::canonicalDataflowSchema,
      PlanValueCardinality::FiniteSet}}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "typed_rewrite_kind"},
}};

constexpr std::array<dataflow::DataflowRewriteKind, 3> rewriteKinds = {{
    dataflow::DataflowRewriteKind::PackUnpackRoundTripEliminate,
    dataflow::DataflowRewriteKind::ParallelizeSerializeRoundTripEliminate,
    dataflow::DataflowRewriteKind::ActivationPreservingConstantFold,
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_rewrite_generator_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedDataflowRewriteGeneratorConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    dataflowRewriteCandidateGeneratorKind,
    "compiler.dataflow_rewrite",
    "loom.compiler.dataflow_rewrite.generator.v1",
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

llvm::Expected<bool>
isAdmitted(const frontend::FabricCapabilityIndex &capabilities,
           const dataflow::CanonicalDataflowArtifact &program) {
  auto miss = capabilities.firstInadmissibleActor(program);
  if (!miss)
    return miss.takeError();
  return !miss->has_value();
}

llvm::Expected<CandidateGeneratorInvocationOutcome>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store) {
  auto config = adoptResolvedDataflowRewriteGeneratorConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();

  const ArtifactRootReference &fabricReference =
      singleInput(inputBindings, FabricInput);
  std::optional<fabric::FinalizedFabricRoot> importedFabric;
  const fabric::FinalizedFabricRoot *fabricRoot = nullptr;
  if (StructuredOwnershipInvocation *invocation =
          detail::StructuredOwnershipInvocationAccess::current()) {
    const fabric::FinalizedFabricRoot &bound =
        detail::StructuredOwnershipInvocationAccess::fabric(*invocation);
    if (bound.reference() != fabricReference)
      return invalid("bound Fabric differs from the active invocation");
    fabricRoot = &bound;
  } else {
    auto imported = fabric::importEntireFabricRoot(fabricReference, store);
    if (!imported)
      return imported.takeError();
    importedFabric.emplace(std::move(*imported));
    fabricRoot = &*importedFabric;
  }
  frontend::FabricCapabilityIndex capabilities(fabricRoot->view());

  std::vector<ArtifactRootReference> outputs;
  for (const ArtifactRootReference &reference :
       inputBindings[CanonicalDataflowProgramsInput].artifacts) {
    auto parent = dataflow::importCanonicalDataflow(reference, store);
    if (!parent)
      return parent.takeError();

    auto parentAdmitted = isAdmitted(capabilities, *parent);
    if (!parentAdmitted)
      return parentAdmitted.takeError();
    if (*parentAdmitted)
      outputs.push_back(reference);

    for (dataflow::DataflowRewriteKind kind : rewriteKinds) {
      auto child = dataflow::materializeDataflowRewrite(*parent, kind);
      if (!child)
        return child.takeError();
      if (!*child)
        continue;

      auto childAdmitted = isAdmitted(capabilities, **child);
      if (!childAdmitted)
        return childAdmitted.takeError();
      if (!*childAdmitted)
        continue;

      auto published = dataflow::publishCanonicalDataflow(**child, store);
      if (!published)
        return published.takeError();
      if (StructuredOwnershipInvocation *invocation =
              detail::StructuredOwnershipInvocationAccess::current())
        if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
                recordDataflowRewriteCandidate(*invocation, reference,
                                               *published, kind, store))
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
resolvedDataflowRewriteGeneratorConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedDataflowRewriteGeneratorConfigView>
projectResolvedDataflowRewriteGeneratorConfigView() {
  std::vector<std::uint8_t> bytes;
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedDataflowRewriteGeneratorConfigView(std::move(bytes),
                                                    std::move(*digest));
}

llvm::Expected<ResolvedDataflowRewriteGeneratorConfigView>
adoptResolvedDataflowRewriteGeneratorConfigView(
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
  return ResolvedDataflowRewriteGeneratorConfigView({}, digest);
}

const CandidateGeneratorDescriptor &
dataflowRewriteCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerDataflowRewriteCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindDataflowRewriteCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error = registerDataflowRewriteCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(CanonicalDataflowProgramsInput),
       canonicalDataflowPrograms.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveDataflowRewriteCandidateGeneratorBinding(
    const ResolvedDataflowRewriteGeneratorConfigView &config) {
  if (llvm::Error error = registerDataflowRewriteCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
