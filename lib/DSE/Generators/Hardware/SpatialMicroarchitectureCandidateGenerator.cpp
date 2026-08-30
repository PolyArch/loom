#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "HardwareTopologyQuality.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.spatial_microarchitecture_rewrite.config.2.3";

constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> inputSlots = {{
    {CandidateGeneratorInputSlotRef(0), "fabric_module_parent",
     PlanValueRole::CandidateSet, &loom::fabric::fabricArtifactSchema,
     PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "fabric_module_child",
     PlanValueRole::CandidateSet, &loom::fabric::fabricArtifactSchema,
     PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "decision_attempt"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "spatial_microarchitecture_generator_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

llvm::Error
validateDecisionAgainstParent(const SpatialMicroarchitectureDecision &decision,
                              const loom::fabric::FabricArtifactView &parent) {
  if (parent.rootKind() != loom::fabric::FabricRootKind::Module)
    return invalid("microarchitecture parent is not a finalized Module");
  return std::visit(
      [&](const auto &value) -> llvm::Error {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, ResizeInstructionStores>) {
          for (const ResizeInstructionStore &store : value.stores)
            if (llvm::Error error =
                    loom::fabric::validateFabricRef(parent, store.target))
              return error;
        } else if (llvm::Error error =
                       loom::fabric::validateFabricRef(parent, value.target)) {
          return error;
        }
        if constexpr (std::is_same_v<Value, ResizeInstructionStores>) {
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<
                                 Value, ChangeFifoQueueDiscipline>) {
          if (!::fabric::symbolizeFifoQueueDiscipline(
                  static_cast<std::uint32_t>(value.discipline)))
            return invalid(
                "FIFO queue discipline is outside its closed domain");
          const auto current = parent.fifoQueueDiscipline(value.target);
          if (!current)
            return invalid("queue-discipline change requires a FIFO");
          if (*current == value.discipline)
            return invalid("FIFO queue-discipline change is a no-op");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<
                                 Value, ChangeTemporalOperandBufferMode>) {
          const auto current = parent.peOperandBufferMode(value.target);
          if (!current)
            return invalid("operand-buffer mode change requires a temporal PE");
          if (*current == value.mode)
            return invalid("operand-buffer mode change is a no-op");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<
                                 Value, ResizeTemporalOperandBuffer>) {
          if (value.entriesPerAllocationUnit == 0)
            return invalid("operand-buffer entries must be positive");
          if (!parent.peOperandBufferMode(value.target))
            return invalid("operand-buffer resize requires a temporal PE");
          if (parent.peOperandBufferSize(value.target) ==
              value.entriesPerAllocationUnit)
            return invalid("operand-buffer resize is a no-op");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<Value, ResizeSwitchRouteTable>) {
          if (value.entries == 0)
            return invalid("switch route-table capacity must be positive");
          if (parent.switchSchedule(value.target) !=
              ::fabric::Schedule::Temporal)
            return invalid("route-table resize requires a Temporal switch");
          if (parent.switchRouteTableSize(value.target) == value.entries)
            return invalid("switch route-table resize is a no-op");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<Value, ChangePeKind> ||
                             std::is_same_v<Value, ChangeFuCapability> ||
                             std::is_same_v<
                                 Value, ChangeSwitchModeOrScheduleCapacity> ||
                             std::is_same_v<Value, ChangeMemoryOperationTable>)
          return loom::fabric::validateFabricRef(parent, value.prototype);
        else if constexpr (std::is_same_v<Value, ChangeFuInventory>) {
          if (value.prototypes.empty())
            return invalid("FU inventory replacement is empty");
          for (auto prototype : value.prototypes)
            if (llvm::Error error =
                    loom::fabric::validateFabricRef(parent, prototype))
              return error;
        }
        return llvm::Error::success();
      },
      decision);
}

llvm::Error applyDecision(loom::adg::SpatialCoreBuilder &builder,
                          const SpatialMicroarchitectureDecision &decision) {
  return std::visit(
      [&](const auto &value) -> llvm::Error {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, ResizeInstructionStores>) {
          for (const ResizeInstructionStore &store : value.stores)
            if (llvm::Error error = builder.resizeInstructionStore(
                    store.target, store.instructionCapacity))
              return error;
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<Value, ChangePeKind>)
          return builder.replacePeKind(value.target, value.prototype);
        else if constexpr (std::is_same_v<Value, ResizeInstructionStore>)
          return builder.resizeInstructionStore(value.target,
                                                value.instructionCapacity);
        else if constexpr (std::is_same_v<Value, ChangeFuInventory>)
          return builder.replaceFuInventory(value.target, value.prototypes);
        else if constexpr (std::is_same_v<Value, ChangeFuCapability>)
          return builder.replaceFuCapability(value.target, value.prototype);
        else if constexpr (std::is_same_v<Value,
                                          ChangeSwitchModeOrScheduleCapacity>)
          return builder.replaceSwitchModeOrScheduleCapacity(value.target,
                                                             value.prototype);
        else if constexpr (std::is_same_v<Value, ResizeSwitchRouteTable>)
          return builder.resizeSwitchRouteTable(value.target, value.entries);
        else if constexpr (std::is_same_v<Value, ResizeMemory>)
          return builder.resizeMemory(value.target, value.capacityBytes);
        else if constexpr (std::is_same_v<Value, ChangeMemoryOperationTable>)
          return builder.replaceMemoryOperationTable(value.target,
                                                     value.prototype);
        else if constexpr (std::is_same_v<Value, ResizeFifo>)
          return builder.resizeFifo(value.target, value.depth);
        else if constexpr (std::is_same_v<Value,
                                          ChangeFifoQueueDiscipline>)
          return builder.changeFifoQueueDiscipline(value.target,
                                                   value.discipline);
        else if constexpr (std::is_same_v<
                               Value, ChangeTemporalOperandBufferMode>)
          return builder.changeTemporalOperandBufferMode(value.target,
                                                         value.mode);
        else if constexpr (std::is_same_v<
                               Value, ResizeTemporalOperandBuffer>)
          return builder.resizeTemporalOperandBuffer(
              value.target, value.entriesPerAllocationUnit);
        else
          return builder.changeFifoBypassCapability(value.target,
                                                    value.bypassable);
      },
      decision);
}

bool isRejectedDraftError(llvm::Error error, std::string &unexpected) {
  std::string message = llvm::toString(std::move(error));
  if (message.find("fabric_artifact_invalid:") != std::string::npos ||
      message.find("adg_builder_invalid:") != std::string::npos ||
      message.find("fabric_module_domain_invalid:") != std::string::npos)
    return true;
  unexpected = std::move(message);
  return false;
}

llvm::Expected<std::optional<loom::fabric::FinalizedFabricModuleProjection>>
materializeChild(const loom::fabric::FinalizedFabricRoot &parent,
                 const SpatialMicroarchitectureDecision &decision,
                 const ArtifactStore &store) {
  loom::adg::DesignBuilder design(store);
  auto builder = design.deriveSpatialCore(parent);
  if (!builder)
    return builder.takeError();
  if (llvm::Error error = applyDecision(*builder, decision)) {
    std::string unexpected;
    if (isRejectedDraftError(std::move(error), unexpected))
      return std::optional<loom::fabric::FinalizedFabricModuleProjection>();
    return llvm::createStringError(llvm::inconvertibleErrorCode(), unexpected);
  }
  if (llvm::Error error = builder->closeDerived()) {
    std::string unexpected;
    if (isRejectedDraftError(std::move(error), unexpected))
      return std::optional<loom::fabric::FinalizedFabricModuleProjection>();
    return llvm::createStringError(llvm::inconvertibleErrorCode(), unexpected);
  }
  auto finalized =
      std::move(design).finalizeDerivedSpatialCoreWithCorrespondence();
  if (!finalized) {
    std::string unexpected;
    if (isRejectedDraftError(finalized.takeError(), unexpected))
      return std::optional<loom::fabric::FinalizedFabricModuleProjection>();
    return llvm::createStringError(llvm::inconvertibleErrorCode(), unexpected);
  }
  if (llvm::Error error =
          validateHardwareTopologyQuality(finalized->root.view()))
    return std::move(error);
  return std::optional<loom::fabric::FinalizedFabricModuleProjection>(
      std::move(*finalized));
}

llvm::Error validateLineagePayload(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &,
    llvm::ArrayRef<ArtifactRootReference> parents, const ArtifactStore &store) {
  auto decision = adoptSpatialMicroarchitectureDecision(bytes);
  if (!decision)
    return decision.takeError();
  if (parents.size() != 1 || parents.front() != decision->parent)
    return invalid("microarchitecture decision does not name its exact parent");
  auto parent = loom::fabric::importEntireFabricRoot(decision->parent, store);
  if (!parent)
    return parent.takeError();
  if (llvm::Error error =
          validateDecisionAgainstParent(decision->decision, parent->view()))
    return error;
  if (decision->entities.empty())
    return invalid("microarchitecture lineage has no Module correspondence");
  return llvm::Error::success();
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    spatialMicroarchitectureDecisionSchemaBytes(), validateLineagePayload};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedSpatialMicroarchitectureRewriteConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    spatialMicroarchitectureCandidateGeneratorKind,
    "spatial_microarchitecture_rewrite",
    "loom.spatial_microarchitecture_rewrite.generator.v4",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &lineageContract,
    ProviderForm::InProcess,
};

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &,
               const CandidateGeneratorInvocationView &) {
  auto config = adoptResolvedSpatialMicroarchitectureRewriteConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();
  const std::uint64_t decisionsPerParent = std::min<std::uint64_t>(
      config->decisions().size(), config->maxChildrenPerParent());
  if (inputBindings.front().artifacts.size() >
      std::numeric_limits<std::uint64_t>::max() / decisionsPerParent)
    return invalid("decision-attempt accounting overflows u64");
  const std::uint64_t attempts =
      inputBindings.front().artifacts.size() * decisionsPerParent;

  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  for (const ArtifactRootReference &parentReference :
       inputBindings.front().artifacts) {
    auto parent = loom::fabric::importEntireFabricRoot(parentReference, store);
    if (!parent)
      return parent.takeError();
    for (const SpatialMicroarchitectureDecision &decision :
         config->decisions().take_front(decisionsPerParent)) {
      if (llvm::Error error =
              validateDecisionAgainstParent(decision, parent->view()))
        return std::move(error);
      auto child = materializeChild(*parent, decision, store);
      if (!child)
        return child.takeError();
      if (!*child)
        continue;
      const ArtifactRootReference childReference = (*child)->root.reference();
      if (childReference == parentReference)
        continue;
      if (llvm::none_of(outputs, [&](const auto &existing) {
            return existing == childReference;
          }))
        outputs.push_back(childReference);
      lineage.push_back(CandidateGeneratorLineageEdge{
          CandidateGeneratorLineageEdgeKind::CandidateDecision,
          CandidateGeneratorOutputSlotRef(0),
          childReference,
          {parentReference},
          encodeSpatialMicroarchitectureDecision(parentReference, decision,
                                                 (*child)->entities)});
    }
  }
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)},
      {{CandidateGeneratorWorkUnitRef(0), attempts, attempts}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedSpatialMicroarchitectureRewriteConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedSpatialMicroarchitectureRewriteConfigView>
resolveSpatialMicroarchitectureRewriteConfig(
    llvm::ArrayRef<SpatialMicroarchitectureDecisionDomain> domains,
    std::uint64_t maxChildrenPerParent) {
  if (maxChildrenPerParent == 0)
    return invalid("max children per parent must be positive");
  auto decisions = expandSpatialMicroarchitectureDecisionDomains(domains);
  if (!decisions)
    return decisions.takeError();
  std::vector<std::uint8_t> bytes = encodeSpatialMicroarchitectureRewriteConfig(
      *decisions, maxChildrenPerParent);
  auto admitted = adoptSpatialMicroarchitectureRewriteConfig(bytes);
  if (!admitted)
    return admitted.takeError();
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedSpatialMicroarchitectureRewriteConfigView(
      std::move(admitted->first), admitted->second, std::move(bytes), *digest);
}

llvm::Expected<ResolvedSpatialMicroarchitectureRewriteConfigView>
adoptResolvedSpatialMicroarchitectureRewriteConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto decoded = adoptSpatialMicroarchitectureRewriteConfig(canonicalViewBytes);
  if (!decoded)
    return decoded.takeError();
  return ResolvedSpatialMicroarchitectureRewriteConfigView(
      std::move(decoded->first), decoded->second, canonicalViewBytes.vec(),
      digest);
}

const CandidateGeneratorDescriptor &
spatialMicroarchitectureCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerSpatialMicroarchitectureCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSpatialMicroarchitectureCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> parents) {
  if (llvm::Error error = registerSpatialMicroarchitectureCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(0), parents.vec()}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialMicroarchitectureCandidateGeneratorBinding(
    const ResolvedSpatialMicroarchitectureRewriteConfigView &config) {
  if (llvm::Error error = registerSpatialMicroarchitectureCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
