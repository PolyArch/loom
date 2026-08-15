#include "DSE/SystemCompositionCandidateGenerator.h"
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
    "loom.system_composition_rewrite.config.1.0";

enum InputSlot : std::uint32_t {
  SystemParentInput,
  AdmissibleModuleInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(SystemParentInput),
         "fabric_system_parent", PlanValueRole::CandidateSet,
         &loom::fabric::fabricArtifactSchema, PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(AdmissibleModuleInput),
         "admissible_fabric_module", PlanValueRole::CandidateSet,
         &loom::fabric::fabricArtifactSchema, PlanValueCardinality::FiniteSet},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "fabric_system_child",
     PlanValueRole::CandidateSet, &loom::fabric::fabricArtifactSchema,
     PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "decision_attempt"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_composition_generator_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

const loom::fabric::FinalizedFabricRoot *
findModule(llvm::ArrayRef<loom::fabric::FinalizedFabricRoot> modules,
           const ArtifactRootReference &reference) {
  for (const auto &module : modules)
    if (module.reference() == reference)
      return &module;
  return nullptr;
}

llvm::Error validateDecisionAgainstParent(
    const SystemCompositionDecision &decision,
    const loom::fabric::FabricArtifactView &parent,
    llvm::ArrayRef<loom::fabric::FinalizedFabricRoot> modules) {
  if (parent.rootKind() != loom::fabric::FabricRootKind::System)
    return invalid("System composition parent is not a finalized System");
  return std::visit(
      [&](const auto &value) -> llvm::Error {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, AddAccCore>) {
          if (llvm::Error error =
                  loom::fabric::validateFabricRef(parent, value.prototype))
            return error;
          if (!findModule(modules, value.module))
            return invalid("AddAccCore selects a Module outside the input set");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<Value, RemoveAccCore>) {
          return loom::fabric::validateFabricRef(parent, value.target);
        } else if constexpr (std::is_same_v<Value, ReplaceSpatialAttachment>) {
          if (llvm::Error error =
                  loom::fabric::validateFabricRef(parent, value.target))
            return error;
          if (!findModule(modules, value.module))
            return invalid(
                "Spatial attachment selects a Module outside the input set");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<Value,
                                            SelectInstructionCoreRealization> ||
                             std::is_same_v<Value, ChangeTransportResource>) {
          if (llvm::Error error =
                  loom::fabric::validateFabricRef(parent, value.target))
            return error;
          return loom::fabric::validateFabricRef(parent, value.prototype);
        } else if constexpr (std::is_same_v<Value, ChangeTransportConnection>) {
          if (llvm::Error error =
                  loom::fabric::validateFabricRef(parent, value.destination))
            return error;
          return loom::fabric::validateFabricRef(parent, value.source);
        } else {
          return std::visit(
              [&](const auto &attachment) -> llvm::Error {
                using Attachment = std::decay_t<decltype(attachment)>;
                if constexpr (std::is_same_v<Attachment,
                                             ChangeSpatialMemoryAttachment>) {
                  if (llvm::Error error = loom::fabric::validateFabricRef(
                          parent, attachment.spatialEndpoint))
                    return error;
                  return loom::fabric::validateFabricRef(
                      parent, attachment.serviceEndpoint);
                } else {
                  if (llvm::Error error = loom::fabric::validateFabricRef(
                          parent, attachment.destination))
                    return error;
                  return loom::fabric::validateFabricRef(parent,
                                                         attachment.source);
                }
              },
              value.value);
        }
      },
      decision);
}

llvm::Error
applyDecision(loom::adg::SystemBuilder &builder,
              const SystemCompositionDecision &decision,
              llvm::ArrayRef<loom::fabric::FinalizedFabricRoot> modules) {
  return std::visit(
      [&](const auto &value) -> llvm::Error {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, AddAccCore>) {
          const auto *module = findModule(modules, value.module);
          if (!module)
            return invalid("AddAccCore Module was not admitted");
          auto added =
              builder.addAccCoreFromPrototype(value.prototype, *module);
          if (!added)
            return added.takeError();
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<Value, RemoveAccCore>) {
          return builder.removeAccCore(value.target);
        } else if constexpr (std::is_same_v<Value, ReplaceSpatialAttachment>) {
          const auto *module = findModule(modules, value.module);
          if (!module)
            return invalid("replacement SpatialCore Module was not admitted");
          return builder.replaceSpatialAttachment(value.target, *module);
        } else if constexpr (std::is_same_v<Value,
                                            SelectInstructionCoreRealization>) {
          return builder.selectInstructionCoreRealization(value.target,
                                                          value.prototype);
        } else if constexpr (std::is_same_v<Value, ChangeTransportResource>) {
          return builder.replaceTransportResource(value.target,
                                                  value.prototype);
        } else if constexpr (std::is_same_v<Value, ChangeTransportConnection>) {
          return builder.replaceTransportConnection(value.destination,
                                                    value.source);
        } else {
          return std::visit(
              [&](const auto &attachment) -> llvm::Error {
                using Attachment = std::decay_t<decltype(attachment)>;
                if constexpr (std::is_same_v<Attachment,
                                             ChangeSpatialMemoryAttachment>)
                  return builder.replaceSpatialMemoryAttachment(
                      attachment.spatialEndpoint, attachment.serviceEndpoint);
                else
                  return builder.replaceMemoryServiceConnection(
                      attachment.destination, attachment.source);
              },
              value.value);
        }
      },
      decision);
}

bool isRejectedDraftError(llvm::Error error, std::string &unexpected) {
  std::string message = llvm::toString(std::move(error));
  const bool rejected =
      message.find("fabric_artifact_invalid:") != std::string::npos ||
      message.find("adg_builder_invalid:") != std::string::npos ||
      message.find("fabric_module_domain_invalid:") != std::string::npos;
  unexpected = std::move(message);
  if (rejected)
    return true;
  return false;
}

llvm::Expected<std::optional<loom::fabric::FinalizedFabricRoot>>
materializeChild(const loom::fabric::FinalizedFabricRoot &parent,
                 const SystemCompositionDecision &decision,
                 llvm::ArrayRef<loom::fabric::FinalizedFabricRoot> modules,
                 const ArtifactStore &store) {
  loom::adg::DesignBuilder design(store);
  auto builder = design.deriveSystem(parent, modules);
  if (!builder)
    return builder.takeError();
  if (llvm::Error error = applyDecision(*builder, decision, modules)) {
    std::string unexpected;
    if (isRejectedDraftError(std::move(error), unexpected))
      return std::optional<loom::fabric::FinalizedFabricRoot>();
    return llvm::createStringError(llvm::inconvertibleErrorCode(), unexpected);
  }
  if (llvm::Error error = builder->close()) {
    std::string unexpected;
    if (isRejectedDraftError(std::move(error), unexpected))
      return std::optional<loom::fabric::FinalizedFabricRoot>();
    return llvm::createStringError(llvm::inconvertibleErrorCode(), unexpected);
  }
  auto finalized = std::move(design).finalize();
  if (!finalized) {
    std::string unexpected;
    if (isRejectedDraftError(finalized.takeError(), unexpected))
      return std::optional<loom::fabric::FinalizedFabricRoot>();
    return llvm::createStringError(llvm::inconvertibleErrorCode(), unexpected);
  }
  if (finalized->roots().size() != 1)
    return invalid(
        "one System composition decision did not finalize one System child");
  if (llvm::Error error =
          validateHardwareTopologyQuality(finalized->roots().front().view()))
    return std::move(error);
  return std::optional<loom::fabric::FinalizedFabricRoot>(
      finalized->roots().front());
}

llvm::Error
validateLineagePayload(llvm::ArrayRef<std::uint8_t> bytes,
                       llvm::ArrayRef<ArtifactRootReference> parents,
                       const ArtifactStore &store) {
  auto decision = adoptSystemCompositionDecision(bytes);
  if (!decision)
    return decision.takeError();
  if (parents.size() != 1 || parents.front() != decision->parent)
    return invalid("System decision does not name its exact parent");
  auto parent = loom::fabric::importEntireFabricRoot(decision->parent, store);
  if (!parent)
    return parent.takeError();
  std::optional<ArtifactRootReference> selectedModule;
  std::visit(
      [&](const auto &value) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, AddAccCore> ||
                      std::is_same_v<Value, ReplaceSpatialAttachment>)
          selectedModule = value.module;
      },
      decision->decision);
  std::vector<loom::fabric::FinalizedFabricRoot> selectedModules;
  if (selectedModule) {
    auto module = loom::fabric::importEntireFabricRoot(*selectedModule, store);
    if (!module)
      return module.takeError();
    if (module->view().rootKind() != loom::fabric::FabricRootKind::Module)
      return invalid("System decision selects a non-Module root");
    selectedModules.push_back(std::move(*module));
  }
  return validateDecisionAgainstParent(decision->decision, parent->view(),
                                       selectedModules);
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    systemCompositionDecisionSchemaBytes(), validateLineagePayload};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedSystemCompositionRewriteConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    systemCompositionCandidateGeneratorKind,
    "system_composition_rewrite",
    "loom.system_composition_rewrite.generator.v1",
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
  auto config = adoptResolvedSystemCompositionRewriteConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();

  std::vector<loom::fabric::FinalizedFabricRoot> modules;
  modules.reserve(inputBindings[AdmissibleModuleInput].artifacts.size());
  for (const ArtifactRootReference &reference :
       inputBindings[AdmissibleModuleInput].artifacts) {
    auto module = loom::fabric::importEntireFabricRoot(reference, store);
    if (!module)
      return module.takeError();
    if (module->view().rootKind() != loom::fabric::FabricRootKind::Module)
      return invalid("admissible Module input contains a non-Module root");
    modules.push_back(std::move(*module));
  }

  const std::uint64_t decisionsPerParent = std::min<std::uint64_t>(
      config->decisions().size(), config->maxChildrenPerParent());
  if (inputBindings[SystemParentInput].artifacts.size() >
      std::numeric_limits<std::uint64_t>::max() / decisionsPerParent)
    return invalid("decision-attempt accounting overflows u64");
  const std::uint64_t attempts =
      inputBindings[SystemParentInput].artifacts.size() * decisionsPerParent;

  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  for (const ArtifactRootReference &parentReference :
       inputBindings[SystemParentInput].artifacts) {
    auto parent = loom::fabric::importEntireFabricRoot(parentReference, store);
    if (!parent)
      return parent.takeError();
    for (const SystemCompositionDecision &decision :
         config->decisions().take_front(decisionsPerParent)) {
      if (llvm::Error error =
              validateDecisionAgainstParent(decision, parent->view(), modules))
        return std::move(error);
      auto child = materializeChild(*parent, decision, modules, store);
      if (!child)
        return child.takeError();
      if (!*child)
        continue;
      const ArtifactRootReference childReference = (*child)->reference();
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
          encodeSystemCompositionDecision(parentReference, decision)});
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
resolvedSystemCompositionRewriteConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedSystemCompositionRewriteConfigView>
resolveSystemCompositionRewriteConfig(
    llvm::ArrayRef<SystemCompositionDecisionDomain> domains,
    std::uint64_t maxChildrenPerParent) {
  if (maxChildrenPerParent == 0)
    return invalid("max children per parent must be positive");
  auto decisions = expandSystemCompositionDecisionDomains(domains);
  if (!decisions)
    return decisions.takeError();
  std::vector<std::uint8_t> bytes =
      encodeSystemCompositionRewriteConfig(*decisions, maxChildrenPerParent);
  auto admitted = adoptSystemCompositionRewriteConfig(bytes);
  if (!admitted)
    return admitted.takeError();
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedSystemCompositionRewriteConfigView(
      std::move(admitted->first), admitted->second, std::move(bytes), *digest);
}

llvm::Expected<ResolvedSystemCompositionRewriteConfigView>
adoptResolvedSystemCompositionRewriteConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto decoded = adoptSystemCompositionRewriteConfig(canonicalViewBytes);
  if (!decoded)
    return decoded.takeError();
  return ResolvedSystemCompositionRewriteConfigView(
      std::move(decoded->first), decoded->second, canonicalViewBytes.vec(),
      digest);
}

const CandidateGeneratorDescriptor &
systemCompositionCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerSystemCompositionCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSystemCompositionCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> parents,
    llvm::ArrayRef<ArtifactRootReference> admissibleModules) {
  if (llvm::Error error = registerSystemCompositionCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(SystemParentInput), parents.vec()},
      {CandidateGeneratorInputSlotRef(AdmissibleModuleInput),
       admissibleModules.vec()}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSystemCompositionCandidateGeneratorBinding(
    const ResolvedSystemCompositionRewriteConfigView &config) {
  if (llvm::Error error = registerSystemCompositionCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
