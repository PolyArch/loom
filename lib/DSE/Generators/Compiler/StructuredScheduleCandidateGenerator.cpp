#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/StructuredSchedule.h"

#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_schedule_generator.config.1.0";

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

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 2> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "loop_scope"},
    {CandidateGeneratorWorkUnitRef(1), "schedule_decision"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_schedule_generator_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

std::vector<std::uint8_t> encodeConfig(std::uint64_t limit) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(8);
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(limit >> shift));
  bytes.push_back(static_cast<std::uint8_t>(limit));
  return bytes;
}

llvm::Expected<std::uint64_t> decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() < 8)
    return invalid("truncated scope expansion limit");
  std::uint64_t limit = 0;
  for (std::uint8_t byte : bytes.take_front(8))
    limit = (limit << 8) | byte;
  if (bytes.size() != 8)
    return invalid("config has trailing bytes");
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  return limit;
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedStructuredScheduleGeneratorConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Error
validateDecisionPayload(llvm::ArrayRef<std::uint8_t> bytes,
                        llvm::ArrayRef<ArtifactRootReference> parents,
                        const ArtifactStore &store) {
  auto adopted = frontend::adoptStructuredScheduleDecision(bytes);
  if (!adopted)
    return adopted.takeError();
  if (parents.size() != 1 ||
      parents.front().schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      parents.front().schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      adopted->loop.parent != parents.front().artifact)
    return invalid("schedule decision does not belong to its exact parent");
  auto parent = frontend::importStructuredProgram(parents.front(), store);
  if (!parent)
    return parent.takeError();
  auto view = parent->view();
  if (!view)
    return view.takeError();
  auto loop = view->resolve(adopted->loop);
  if (!loop)
    return loop.takeError();
  if (!llvm::isa_and_nonnull<mlir::scf::ForOp>(loop->operation))
    return invalid("schedule decision does not reference an exact SCF loop");
  return llvm::Error::success();
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    frontend::structuredScheduleDecisionSchemaBytes(), validateDecisionPayload};

const CandidateGeneratorDescriptor descriptor{
    structuredScheduleCandidateGeneratorKind,
    "compiler.structured_schedule",
    "loom.compiler.structured_schedule.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &lineageContract,
    ProviderForm::InProcess,
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<CandidateGeneratorInputBinding> bindings,
            InputSlot slot) {
  return bindings[slot].artifacts.front();
}

bool hasSelectedSpatialRegion(
    const frontend::StructuredProgramCandidate &candidate) {
  bool found = false;
  candidate.module().walk([&](loom::SpatialRegionOp) {
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

llvm::Expected<CandidateGeneratorProviderResult> invokeScheduleProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto config = adoptResolvedStructuredScheduleGeneratorConfigView(
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
  frontend::FabricCapabilityIndex capabilities(exactFabric->view());
  const lowering::CanonicalDataflowLoweringOptions loweringOptions =
      invocation ? detail::StructuredOwnershipInvocationAccess::loweringOptions(
                       *invocation)
                 : lowering::CanonicalDataflowLoweringOptions{};

  std::vector<ArtifactRootReference> outputs =
      inputBindings[StructuredProgramsInput].artifacts;
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
  std::uint64_t inspectedLoopScopes = 0;
  std::uint64_t decisionAttempts = 0;
  for (const ArtifactRootReference &reference :
       inputBindings[StructuredProgramsInput].artifacts) {
    auto parent = frontend::importStructuredProgram(reference, store);
    if (!parent)
      return parent.takeError();
    auto decisions = frontend::enumerateStructuredScheduleDecisions(
        *parent, *exactFabric, config->scopeExpansionLimit());
    if (!decisions)
      return decisions.takeError();
    if (decisions->inspectedLoopScopes >
        std::numeric_limits<std::uint64_t>::max() - inspectedLoopScopes)
      return invalid("loop-scope accounting overflows u64");
    inspectedLoopScopes += decisions->inspectedLoopScopes;
    if (decisions->decisions.size() >
        std::numeric_limits<std::uint64_t>::max() - decisionAttempts)
      return invalid("schedule-decision accounting overflows u64");
    decisionAttempts += decisions->decisions.size();
    outputs.reserve(outputs.size() + decisions->decisions.size());
    std::optional<frontend::StructuredEntityRef> trackedSpatialRegion;
    if (invocation) {
      auto tracked = detail::StructuredOwnershipInvocationAccess::
          ownedSpatialRegion(*invocation, reference);
      if (!tracked)
        return tracked.takeError();
      trackedSpatialRegion = *tracked;
    }
    for (const frontend::StructuredScheduleDecision &decision :
         decisions->decisions) {
      auto child = frontend::materializeStructuredScheduleDecision(
          *parent, decision, trackedSpatialRegion);
      if (!child)
        return child.takeError();
      std::optional<lowering::ProjectedCanonicalDataflow> projected;
      if (hasSelectedSpatialRegion(child->structuredProgram)) {
        auto lowered =
            lowering::lowerStructuredProgramToCanonicalDataflowWithProjection(
                child->structuredProgram, loweringOptions);
        if (!lowered)
          return lowered.takeError();
        auto miss = capabilities.firstInadmissibleActor(lowered->artifact);
        if (!miss)
          return miss.takeError();
        if (*miss)
          continue;
        projected.emplace(std::move(*lowered));
      }
      auto published =
          frontend::publishStructuredProgram(child->structuredProgram, store);
      if (!published)
        return published.takeError();
      if (invocation)
        if (!projected)
          return invalid(
              "central Schedule child has no selected Spatial projection");
      if (invocation)
        if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
                recordScheduleCandidate(*invocation, reference, *published,
                                        decision, std::move(*child),
                                        std::move(*projected), store))
          return std::move(error);
      auto ownerPayload = frontend::encodeStructuredScheduleDecision(decision);
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
      {{CandidateGeneratorWorkUnitRef(0), inspectedLoopScopes,
        inspectedLoopScopes},
       {CandidateGeneratorWorkUnitRef(1), decisionAttempts,
        decisionAttempts}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeScheduleProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedStructuredScheduleGeneratorConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedStructuredScheduleGeneratorConfigView>
projectResolvedStructuredScheduleGeneratorConfigView(
    const ResolvedConfig &config) {
  const std::uint64_t limit = config.dse.schedule.scopeExpansionLimit;
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  std::vector<std::uint8_t> bytes = encodeConfig(limit);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedStructuredScheduleGeneratorConfigView(limit, std::move(bytes),
                                                       std::move(*digest));
}

llvm::Expected<ResolvedStructuredScheduleGeneratorConfigView>
adoptResolvedStructuredScheduleGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto limit = decodeConfig(canonicalViewBytes);
  if (!limit)
    return limit.takeError();
  std::vector<std::uint8_t> reencoded = encodeConfig(*limit);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("decoded config does not re-encode to the source bytes");
  return ResolvedStructuredScheduleGeneratorConfigView(
      *limit, std::move(reencoded), digest);
}

const CandidateGeneratorDescriptor &
structuredScheduleCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerStructuredScheduleCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredScheduleCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error = registerStructuredScheduleCandidateGenerator())
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
resolveStructuredScheduleCandidateGeneratorBinding(
    const ResolvedStructuredScheduleGeneratorConfigView &config) {
  if (llvm::Error error = registerStructuredScheduleCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
