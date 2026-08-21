#include "DSE/SpatialTopologyCandidateGenerator.h"
#include "HardwareTopologyQuality.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.spatial_topology_rewrite.config.1.0";

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
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_topology_generator_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

llvm::Error
validatePhysicalOwner(const loom::fabric::FabricArtifactView &parent,
                      const loom::fabric::FabricModulePhysicalOwnerRef &owner) {
  return std::visit(
      [&](const auto &reference) {
        return loom::fabric::validateFabricRef(parent, reference);
      },
      owner.payload());
}

llvm::Error
validateDecisionAgainstParent(const SpatialTopologyDecision &decision,
                              const loom::fabric::FabricArtifactView &parent) {
  if (parent.rootKind() != loom::fabric::FabricRootKind::Module)
    return invalid("topology parent is not a finalized Module");
  return std::visit(
      [&](const auto &value) -> llvm::Error {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, AddOccurrence>)
          return validatePhysicalOwner(parent, value.prototype);
        else if constexpr (std::is_same_v<Value, RemoveOccurrence>)
          return validatePhysicalOwner(parent, value.target);
        else if constexpr (std::is_same_v<Value, ReplacePointConnection>) {
          if (llvm::Error error =
                  loom::fabric::validateFabricRef(parent, value.destination))
            return error;
          return loom::fabric::validateFabricRef(parent, value.source);
        } else if constexpr (std::is_same_v<Value,
                                            AdjustParallelConnectionCount>) {
          for (const auto &connection : value.connections) {
            if (llvm::Error error = loom::fabric::validateFabricRef(
                    parent, connection.destination))
              return error;
            if (llvm::Error error =
                    loom::fabric::validateFabricRef(parent, connection.source))
              return error;
          }
          return llvm::Error::success();
        } else {
          for (const auto &source : value.value.outputSources)
            if (llvm::Error error =
                    loom::fabric::validateFabricRef(parent, source))
              return error;
          return llvm::Error::success();
        }
      },
      decision);
}

llvm::Error applyDecision(loom::adg::SpatialCoreBuilder &builder,
                          const SpatialTopologyDecision &decision) {
  return std::visit(
      [&](const auto &value) -> llvm::Error {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, AddOccurrence>)
          return builder.cloneOccurrence(value.prototype);
        else if constexpr (std::is_same_v<Value, RemoveOccurrence>)
          return builder.eraseOccurrence(value.target);
        else if constexpr (std::is_same_v<Value, ReplacePointConnection>)
          return builder.replacePointConnection(value.destination,
                                                value.source);
        else if constexpr (std::is_same_v<Value, AdjustParallelConnectionCount>)
          return builder.replaceParallelConnections(value.connections);
        else
          return builder.changeBoundaryInventory(value.value.inputCount,
                                                 value.value.outputSources);
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

llvm::Expected<std::optional<loom::fabric::FinalizedFabricRoot>>
materializeChild(const loom::fabric::FinalizedFabricRoot &parent,
                 const SpatialTopologyDecision &decision,
                 const ArtifactStore &store) {
  loom::adg::DesignBuilder design(store);
  auto builder = design.deriveSpatialCore(parent);
  if (!builder)
    return builder.takeError();
  if (llvm::Error error = applyDecision(*builder, decision)) {
    std::string unexpected;
    if (isRejectedDraftError(std::move(error), unexpected))
      return std::optional<loom::fabric::FinalizedFabricRoot>();
    return llvm::createStringError(llvm::inconvertibleErrorCode(), unexpected);
  }
  if (llvm::Error error = builder->closeDerived()) {
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
    return invalid("one topology decision did not finalize one Module child");
  if (llvm::Error error =
          validateHardwareTopologyQuality(finalized->roots().front().view()))
    return std::move(error);
  return std::optional<loom::fabric::FinalizedFabricRoot>(
      finalized->roots().front());
}

llvm::Error validateLineagePayload(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &,
    llvm::ArrayRef<ArtifactRootReference> parents, const ArtifactStore &store) {
  auto decision = adoptSpatialTopologyDecision(bytes);
  if (!decision)
    return decision.takeError();
  if (parents.size() != 1 || parents.front() != decision->parent)
    return invalid("topology decision does not name its exact parent");
  auto parent = loom::fabric::importEntireFabricRoot(decision->parent, store);
  if (!parent)
    return parent.takeError();
  return validateDecisionAgainstParent(decision->decision, parent->view());
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    spatialTopologyDecisionSchemaBytes(), validateLineagePayload};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedSpatialTopologyRewriteConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    spatialTopologyCandidateGeneratorKind,
    "spatial_topology_rewrite",
    "loom.spatial_topology_rewrite.generator.v1",
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
  auto config = adoptResolvedSpatialTopologyRewriteConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();
  const std::uint64_t decisionsPerParent = std::min<std::uint64_t>(
      config->decisions().size(), config->maxChildrenPerParent());
  if (!inputBindings.empty() &&
      inputBindings.front().artifacts.size() >
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
    for (const SpatialTopologyDecision &decision :
         config->decisions().take_front(decisionsPerParent)) {
      if (llvm::Error error =
              validateDecisionAgainstParent(decision, parent->view()))
        return std::move(error);
      auto child = materializeChild(*parent, decision, store);
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
          encodeSpatialTopologyDecision(parentReference, decision)});
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

llvm::ArrayRef<std::uint8_t> resolvedSpatialTopologyRewriteConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedSpatialTopologyRewriteConfigView>
resolveSpatialTopologyRewriteConfig(
    llvm::ArrayRef<SpatialTopologyDecisionDomain> domains,
    std::uint64_t maxChildrenPerParent) {
  if (maxChildrenPerParent == 0)
    return invalid("max children per parent must be positive");
  auto decisions = expandSpatialTopologyDecisionDomains(domains);
  if (!decisions)
    return decisions.takeError();
  std::vector<std::uint8_t> bytes =
      encodeSpatialTopologyRewriteConfig(*decisions, maxChildrenPerParent);
  auto admitted = adoptSpatialTopologyRewriteConfig(bytes);
  if (!admitted)
    return admitted.takeError();
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedSpatialTopologyRewriteConfigView(
      std::move(admitted->first), admitted->second, std::move(bytes), *digest);
}

llvm::Expected<ResolvedSpatialTopologyRewriteConfigView>
adoptResolvedSpatialTopologyRewriteConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto decoded = adoptSpatialTopologyRewriteConfig(canonicalViewBytes);
  if (!decoded)
    return decoded.takeError();
  return ResolvedSpatialTopologyRewriteConfigView(
      std::move(decoded->first), decoded->second, canonicalViewBytes.vec(),
      digest);
}

const CandidateGeneratorDescriptor &
spatialTopologyCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerSpatialTopologyCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSpatialTopologyCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> parents) {
  if (llvm::Error error = registerSpatialTopologyCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(0), parents.vec()}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialTopologyCandidateGeneratorBinding(
    const ResolvedSpatialTopologyRewriteConfigView &config) {
  if (llvm::Error error = registerSpatialTopologyCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
