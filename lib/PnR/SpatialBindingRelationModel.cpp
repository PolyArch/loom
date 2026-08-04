#include "SpatialBindingRelationModel.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <variant>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

using Projection = ::mapping::SpatialConstraintProjection;
using ProjectionKey = std::vector<std::uint8_t>;

llvm::Error invalid(Projection projection, const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid,
      ("invalid Spatial binding relation projection: " + message).str(),
      projection);
}

bool isComputeProjection(Projection projection) {
  switch (projection) {
  case Projection::ComputePlacement:
  case Projection::ComputeParentPe:
  case Projection::ComputeInstructionContext:
  case Projection::ComputeFuContext:
    return true;
  default:
    return false;
  }
}

bool isBindingProjection(Projection projection) {
  return isComputeProjection(projection) ||
         projection == Projection::MemoryPlacement ||
         projection == Projection::MemoryOperationPort ||
         projection == Projection::SpatialTransferAttachment;
}

bool isRouteProjection(Projection projection) {
  return projection == Projection::NetSelectedPhysicalTraversals ||
         projection == Projection::NetTraversalResourceStates;
}

llvm::Error infeasible(Projection projection, const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::ProvenInfeasible,
      ("infeasible Spatial binding relation projection: " + message).str(),
      projection);
}

void appendU32Be(ProjectionKey &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendComponent(ProjectionKey &bytes,
                     llvm::ArrayRef<std::uint8_t> component) {
  assert(component.size() <= std::numeric_limits<std::uint32_t>::max());
  appendU32Be(bytes, static_cast<std::uint32_t>(component.size()));
  bytes.insert(bytes.end(), component.begin(), component.end());
}

ProjectionKey
computeProjectionKey(Projection projection,
                     const FrozenSpatialRealizationIndex &realizations,
                     const SpatialComputeBindingChoice &choice) {
  const FrozenSpatialComputePlacement &placement =
      realizations.computePlacements()[choice.placement];
  const InstructionContextRef &context =
      realizations.computeInstructionContexts()[choice.instructionContext];
  switch (projection) {
  case Projection::ComputePlacement:
    return canonicalFabricBytes(placement.fu);
  case Projection::ComputeParentPe:
    return canonicalFabricBytes(placement.parentPe);
  case Projection::ComputeInstructionContext:
    return canonicalFabricBytes(context);
  case Projection::ComputeFuContext: {
    ProjectionKey key;
    const ProjectionKey fu = canonicalFabricBytes(placement.fu);
    const ProjectionKey instructionContext = canonicalFabricBytes(context);
    key.reserve(8 + fu.size() + instructionContext.size());
    appendComponent(key, fu);
    appendComponent(key, instructionContext);
    return key;
  }
  default:
    llvm_unreachable("non-compute projection in compute projection key");
  }
}

ProjectionKey
memoryProjectionKey(const FrozenSpatialRealizationIndex &realizations,
                    const SpatialMemoryBindingChoice &choice) {
  return canonicalFabricBytes(
      realizations.memoryPlacements()[choice.placement].memory);
}

ProjectionKey memoryOperationPortProjectionKey(
    const FrozenSpatialRealizationIndex &realizations,
    const SpatialMemoryBindingChoice &choice, std::uint64_t portOrdinal) {
  return canonicalFabricBytes(FabricMemoryOperationPortRef{
      realizations.memoryPlacements()[choice.placement].memory, portOrdinal});
}

llvm::Expected<ProjectionKey> actorKey(const ArtifactIdentity &dataflowIdentity,
                                       const dataflow::ActorRef &actor) {
  auto bytes = dataflow::encodeDataflowReference(dataflowIdentity, actor);
  if (!bytes)
    return invalid(Projection::MemoryOperationPort,
                   "cannot encode the memory actor: " +
                       llvm::toString(bytes.takeError()));
  return std::move(*bytes);
}

struct MemoryOperationPortDecision final {
  PnrIndex decision = 0;
  std::uint64_t portOrdinal = 0;
};

llvm::Expected<ProjectionKey>
transferTerminalKey(const ArtifactIdentity &dataflowIdentity,
                    const SpatialConstraintTransferTerminal &terminal) {
  auto producer =
      dataflow::encodeDataflowReference(dataflowIdentity, terminal.producer);
  if (!producer)
    return invalid(Projection::SpatialTransferAttachment,
                   "cannot encode the transfer producer: " +
                       llvm::toString(producer.takeError()));
  ProjectionKey key;
  appendComponent(key, *producer);
  key.push_back(terminal.consumer ? 1 : 0);
  if (terminal.consumer) {
    auto consumer =
        dataflow::encodeDataflowReference(dataflowIdentity, *terminal.consumer);
    if (!consumer)
      return invalid(Projection::SpatialTransferAttachment,
                     "cannot encode the transfer consumer: " +
                         llvm::toString(consumer.takeError()));
    appendComponent(key, *consumer);
  }
  return key;
}

llvm::Expected<std::vector<PnrIndex>> restrictedAttachmentEndpoints(
    const FrozenConstraintShard &shard,
    const SpatialConstraintTransferTerminal &subject,
    const std::map<ProjectionKey, PnrIndex> &endpointOrdinals) {
  const auto restricted = shard.restrictedDomain(subject);
  if (!restricted)
    return std::vector<PnrIndex>{};
  std::vector<PnrIndex> endpoints;
  endpoints.reserve(restricted->size());
  for (const SpatialConstraintDomainValue &value : *restricted) {
    const auto *endpoint = std::get_if<FabricTransportEndpointRef>(&value);
    if (!endpoint)
      return invalid(Projection::SpatialTransferAttachment,
                     "attachment restriction contains a non-endpoint value");
    const auto found = endpointOrdinals.find(canonicalFabricBytes(*endpoint));
    if (found == endpointOrdinals.end())
      return invalid(Projection::SpatialTransferAttachment,
                     "attachment restriction names a foreign endpoint");
    endpoints.push_back(found->second);
  }
  llvm::sort(endpoints);
  endpoints.erase(std::unique(endpoints.begin(), endpoints.end()),
                  endpoints.end());
  return endpoints;
}

llvm::Expected<PnrIndex> relationDecision(
    Projection projection, const SpatialConstraintSubject &subject,
    const llvm::DenseMap<std::uint64_t, PnrIndex> &computeDecisions,
    const llvm::DenseMap<std::uint64_t, PnrIndex> &memoryDecisions,
    const std::map<ProjectionKey, MemoryOperationPortDecision>
        &memoryOperationPortDecisions,
    const std::map<ProjectionKey, PnrIndex> &attachmentDecisions,
    const ArtifactIdentity &dataflowIdentity) {
  if (isComputeProjection(projection)) {
    const auto *compute = std::get_if<TechComputeRealizationRef>(&subject);
    if (!compute)
      return invalid(projection,
                     "compute projection has a non-compute subject");
    const auto found = computeDecisions.find(compute->entity);
    if (found == computeDecisions.end())
      return invalid(projection,
                     "compute projection names a foreign realization");
    return found->second;
  }
  if (projection == Projection::MemoryPlacement) {
    const auto *memory = std::get_if<TechMemoryRealizationRef>(&subject);
    if (!memory)
      return invalid(projection, "memory projection has a non-memory subject");
    const auto found = memoryDecisions.find(memory->entity);
    if (found == memoryDecisions.end())
      return invalid(projection,
                     "memory projection names a foreign realization");
    return found->second;
  }
  if (projection == Projection::MemoryOperationPort) {
    const auto *actor = std::get_if<dataflow::ActorRef>(&subject);
    if (!actor)
      return invalid(projection,
                     "memory operation-port projection has a non-actor "
                     "subject");
    auto key = actorKey(dataflowIdentity, *actor);
    if (!key)
      return key.takeError();
    const auto found = memoryOperationPortDecisions.find(*key);
    if (found == memoryOperationPortDecisions.end())
      return invalid(projection,
                     "memory operation-port projection names a foreign "
                     "memory actor");
    return found->second.decision;
  }
  if (projection == Projection::SpatialTransferAttachment) {
    const auto *terminal =
        std::get_if<SpatialConstraintTransferTerminal>(&subject);
    if (!terminal)
      return invalid(projection,
                     "attachment projection has a non-terminal subject");
    auto key = transferTerminalKey(dataflowIdentity, *terminal);
    if (!key)
      return key.takeError();
    const auto found = attachmentDecisions.find(*key);
    if (found == attachmentDecisions.end())
      return invalid(projection,
                     "attachment projection names a foreign terminal");
    return found->second;
  }
  llvm_unreachable("deferred projection requested a binding decision");
}

} // namespace

llvm::Expected<std::shared_ptr<const SpatialBindingRelationModel>>
SpatialBindingRelationModel::create(
    const ArtifactIdentity &dataflowIdentity,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenConstraintIndex &constraints,
    const FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialPortIndex &ports,
    const FrozenSpatialRoutingGraph &routing) {
  std::vector<PnrIndex> computeChoiceOffsets;
  std::vector<SpatialComputeBindingChoice> computeChoices;
  std::vector<PnrIndex> computeContextChoiceOrdinals(
      realizations.computeInstructionContexts().size(), getInvalidPnrIndex());
  std::vector<PnrIndex> memoryChoiceOffsets;
  std::vector<SpatialMemoryBindingChoice> memoryChoices;
  std::vector<PnrIndex> memoryPlacementChoiceOrdinals(
      realizations.memoryPlacements().size(), getInvalidPnrIndex());
  std::vector<PnrIndex> portAttachmentChoiceOffsets;
  std::vector<PnrIndex> graphBoundaryAttachmentChoiceOffsets;
  std::vector<PnrIndex> attachmentChoices;
  std::vector<PnrIndex> attachmentOptionChoiceOrdinals(
      ports.attachmentOptions().size(), getInvalidPnrIndex());
  std::vector<PnrIndex> decisionChoiceCounts;

  computeChoiceOffsets.reserve(realizations.computeRealizations().size() + 1);
  computeChoiceOffsets.push_back(0);
  decisionChoiceCounts.reserve(realizations.computeRealizations().size() +
                               realizations.memoryRealizations().size());
  for (const FrozenSpatialComputeRealization &realization :
       realizations.computeRealizations()) {
    const std::size_t begin = computeChoices.size();
    for (PnrIndex placement = realization.placementOffset;
         placement != realization.placementOffset + realization.placementCount;
         ++placement) {
      const FrozenSpatialComputePlacement &record =
          realizations.computePlacements()[placement];
      for (PnrIndex context = record.contextOffset;
           context != record.contextOffset + record.contextCount; ++context) {
        if (context >= computeContextChoiceOrdinals.size() ||
            computeContextChoiceOrdinals[context] != getInvalidPnrIndex())
          return invalid(Projection::ComputeInstructionContext,
                         "instruction context choice ownership is invalid");
        computeContextChoiceOrdinals[context] =
            static_cast<PnrIndex>(computeChoices.size() - begin);
        computeChoices.push_back({placement, context});
      }
    }
    const std::size_t count = computeChoices.size() - begin;
    if (count == 0 || computeChoices.size() > getPnrIndexMax())
      return invalid(Projection::ComputeFuContext,
                     "compute choice domain is empty or too large");
    decisionChoiceCounts.push_back(static_cast<PnrIndex>(count));
    computeChoiceOffsets.push_back(
        static_cast<PnrIndex>(computeChoices.size()));
  }

  memoryChoiceOffsets.reserve(realizations.memoryRealizations().size() + 1);
  memoryChoiceOffsets.push_back(0);
  for (const FrozenSpatialMemoryRealization &realization :
       realizations.memoryRealizations()) {
    const std::size_t begin = memoryChoices.size();
    for (PnrIndex placement = realization.placementOffset;
         placement != realization.placementOffset + realization.placementCount;
         ++placement) {
      if (placement >= memoryPlacementChoiceOrdinals.size() ||
          memoryPlacementChoiceOrdinals[placement] != getInvalidPnrIndex())
        return invalid(Projection::MemoryPlacement,
                       "memory placement choice ownership is invalid");
      memoryPlacementChoiceOrdinals[placement] =
          static_cast<PnrIndex>(memoryChoices.size() - begin);
      memoryChoices.push_back({placement});
    }
    const std::size_t count = memoryChoices.size() - begin;
    if (count == 0 || memoryChoices.size() > getPnrIndexMax())
      return invalid(Projection::MemoryPlacement,
                     "memory choice domain is empty or too large");
    decisionChoiceCounts.push_back(static_cast<PnrIndex>(count));
    memoryChoiceOffsets.push_back(static_cast<PnrIndex>(memoryChoices.size()));
  }

  const PnrIndex computeDecisionCount =
      static_cast<PnrIndex>(realizations.computeRealizations().size());
  const PnrIndex memoryDecisionCount =
      static_cast<PnrIndex>(realizations.memoryRealizations().size());
  const PnrIndex portDecisionOffset =
      computeDecisionCount + memoryDecisionCount;
  const PnrIndex graphBoundaryDecisionOffset =
      portDecisionOffset + static_cast<PnrIndex>(ports.portDemands().size());

  std::map<ProjectionKey, PnrIndex> endpointOrdinals;
  for (auto [ordinal, endpoint] : llvm::enumerate(routing.routingEndpoints())) {
    const bool inserted =
        endpointOrdinals
            .try_emplace(canonicalFabricBytes(endpoint.reference),
                         static_cast<PnrIndex>(ordinal))
            .second;
    if (!inserted)
      return invalid(Projection::SpatialTransferAttachment,
                     "routing endpoint reference is not unique");
  }

  std::map<ProjectionKey, PnrIndex> attachmentDecisions;
  std::vector<std::optional<SpatialConstraintTransferTerminal>>
      attachmentSubjects(ports.portDemands().size() +
                         ports.graphBoundaries().size());
  const auto rememberTerminal =
      [&](const SpatialConstraintTransferTerminal &terminal,
          FrozenSpatialTerminalBinding binding) -> llvm::Error {
    PnrIndex decision = 0;
    PnrIndex subjectOrdinal = 0;
    if (binding.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
      if (binding.index >= ports.portDemands().size())
        return invalid(Projection::SpatialTransferAttachment,
                       "terminal names a foreign PortDemand");
      decision = portDecisionOffset + binding.index;
      subjectOrdinal = binding.index;
    } else {
      if (binding.index >= ports.graphBoundaries().size())
        return invalid(Projection::SpatialTransferAttachment,
                       "terminal names a foreign graph boundary");
      decision = graphBoundaryDecisionOffset + binding.index;
      subjectOrdinal =
          static_cast<PnrIndex>(ports.portDemands().size()) + binding.index;
    }
    auto key = transferTerminalKey(dataflowIdentity, terminal);
    if (!key)
      return key.takeError();
    if (!attachmentDecisions.try_emplace(std::move(*key), decision).second)
      return invalid(Projection::SpatialTransferAttachment,
                     "transfer terminal ownership is not unique");
    if (attachmentSubjects[subjectOrdinal])
      return invalid(Projection::SpatialTransferAttachment,
                     "attachment decision owns multiple transfer terminals");
    attachmentSubjects[subjectOrdinal] = terminal;
    return llvm::Error::success();
  };
  if (transfers.logicalNetSourceBindings().size() !=
      transfers.logicalNets().size())
    return invalid(Projection::SpatialTransferAttachment,
                   "logical-net source binding index is incomplete");
  for (PnrIndex net = 0; net < transfers.logicalNets().size(); ++net) {
    const FrozenSpatialLogicalNet &record = transfers.logicalNets()[net];
    if (record.sinkOffset + record.sinkCount >
            transfers.logicalNetSinks().size() ||
        record.sinkOffset + record.sinkCount >
            transfers.logicalNetSinkBindings().size())
      return invalid(Projection::SpatialTransferAttachment,
                     "logical-net sink binding index is incomplete");
    if (llvm::Error error =
            rememberTerminal({record.producer, std::nullopt},
                             transfers.logicalNetSourceBindings()[net]))
      return std::move(error);
    for (PnrIndex sink = 0; sink < record.sinkCount; ++sink)
      if (llvm::Error error = rememberTerminal(
              {record.producer,
               transfers.logicalNetSinks()[record.sinkOffset + sink]},
              transfers.logicalNetSinkBindings()[record.sinkOffset + sink]))
        return std::move(error);
  }
  if (llvm::any_of(attachmentSubjects,
                   [](const auto &subject) { return !subject.has_value(); }))
    return invalid(Projection::SpatialTransferAttachment,
                   "an attachment decision has no exact transfer terminal");

  const FrozenConstraintShard &attachmentShard =
      constraints.shard(Projection::SpatialTransferAttachment);
  const auto appendAttachmentChoice = [&](PnrIndex option,
                                          PnrIndex localChoice) -> llvm::Error {
    if (option >= ports.attachmentOptions().size() ||
        attachmentOptionChoiceOrdinals[option] != getInvalidPnrIndex())
      return invalid(Projection::SpatialTransferAttachment,
                     "attachment option ownership is invalid");
    attachmentOptionChoiceOrdinals[option] = localChoice;
    attachmentChoices.push_back(option);
    return llvm::Error::success();
  };
  portAttachmentChoiceOffsets.reserve(ports.portDemands().size() + 1);
  portAttachmentChoiceOffsets.push_back(0);
  for (PnrIndex demand = 0; demand < ports.portDemands().size(); ++demand) {
    auto restricted = restrictedAttachmentEndpoints(
        attachmentShard, *attachmentSubjects[demand], endpointOrdinals);
    if (!restricted)
      return restricted.takeError();
    const bool hasRestriction =
        attachmentShard.restrictedDomain(*attachmentSubjects[demand])
            .has_value();
    const std::size_t begin = attachmentChoices.size();
    const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
    for (PnrIndex localDomain = 0; localDomain < record.placementDomainCount;
         ++localDomain) {
      const PnrIndex domainOrdinal = record.placementDomainOffset + localDomain;
      if (domainOrdinal >= ports.placementDomains().size())
        return invalid(Projection::SpatialTransferAttachment,
                       "PortDemand placement domain is out of range");
      const FrozenSpatialPortPlacementDomain &domain =
          ports.placementDomains()[domainOrdinal];
      for (PnrIndex local = 0; local < domain.attachmentOptionCount; ++local) {
        const PnrIndex option = domain.attachmentOptionOffset + local;
        if (option >= ports.attachmentOptions().size())
          return invalid(Projection::SpatialTransferAttachment,
                         "PortDemand attachment option is out of range");
        const auto &choice = ports.attachmentOptions()[option];
        if (choice.ownerKind !=
                FrozenSpatialAttachmentOwnerKind::PlacementDomain ||
            choice.owner != domainOrdinal)
          return invalid(Projection::SpatialTransferAttachment,
                         "PortDemand attachment owner is malformed");
        if (hasRestriction &&
            !std::binary_search(restricted->begin(), restricted->end(),
                                choice.endpoint))
          continue;
        if (llvm::Error error = appendAttachmentChoice(
                option,
                static_cast<PnrIndex>(attachmentChoices.size() - begin)))
          return std::move(error);
      }
    }
    const std::size_t count = attachmentChoices.size() - begin;
    if (count == 0)
      return infeasible(Projection::SpatialTransferAttachment,
                        "a PortDemand has no admissible attachment");
    if (count > getPnrIndexMax() || attachmentChoices.size() > getPnrIndexMax())
      return invalid(Projection::SpatialTransferAttachment,
                     "PortDemand attachment domain overflows PnrIndex");
    decisionChoiceCounts.push_back(static_cast<PnrIndex>(count));
    portAttachmentChoiceOffsets.push_back(
        static_cast<PnrIndex>(attachmentChoices.size()));
  }

  graphBoundaryAttachmentChoiceOffsets.reserve(ports.graphBoundaries().size() +
                                               1);
  graphBoundaryAttachmentChoiceOffsets.push_back(
      static_cast<PnrIndex>(attachmentChoices.size()));
  for (PnrIndex boundary = 0; boundary < ports.graphBoundaries().size();
       ++boundary) {
    const PnrIndex subjectOrdinal =
        static_cast<PnrIndex>(ports.portDemands().size()) + boundary;
    auto restricted = restrictedAttachmentEndpoints(
        attachmentShard, *attachmentSubjects[subjectOrdinal], endpointOrdinals);
    if (!restricted)
      return restricted.takeError();
    const bool hasRestriction =
        attachmentShard.restrictedDomain(*attachmentSubjects[subjectOrdinal])
            .has_value();
    const std::size_t begin = attachmentChoices.size();
    const FrozenSpatialGraphBoundary &record =
        ports.graphBoundaries()[boundary];
    for (PnrIndex local = 0; local < record.attachmentOptionCount; ++local) {
      const PnrIndex option = record.attachmentOptionOffset + local;
      if (option >= ports.attachmentOptions().size())
        return invalid(Projection::SpatialTransferAttachment,
                       "graph-boundary attachment option is out of range");
      const auto &choice = ports.attachmentOptions()[option];
      if (choice.ownerKind != FrozenSpatialAttachmentOwnerKind::GraphBoundary ||
          choice.owner != boundary)
        return invalid(Projection::SpatialTransferAttachment,
                       "graph-boundary attachment owner is malformed");
      if (hasRestriction &&
          !std::binary_search(restricted->begin(), restricted->end(),
                              choice.endpoint))
        continue;
      if (llvm::Error error = appendAttachmentChoice(
              option, static_cast<PnrIndex>(attachmentChoices.size() - begin)))
        return std::move(error);
    }
    const std::size_t count = attachmentChoices.size() - begin;
    if (count == 0)
      return infeasible(Projection::SpatialTransferAttachment,
                        "a graph boundary has no admissible attachment");
    if (count > getPnrIndexMax() || attachmentChoices.size() > getPnrIndexMax())
      return invalid(Projection::SpatialTransferAttachment,
                     "graph-boundary attachment domain overflows PnrIndex");
    decisionChoiceCounts.push_back(static_cast<PnrIndex>(count));
    graphBoundaryAttachmentChoiceOffsets.push_back(
        static_cast<PnrIndex>(attachmentChoices.size()));
  }

  llvm::DenseMap<std::uint64_t, PnrIndex> computeDecisions;
  llvm::DenseMap<std::uint64_t, PnrIndex> memoryDecisions;
  for (auto [ordinal, realization] :
       llvm::enumerate(realizations.computeRealizations())) {
    const bool inserted = computeDecisions
                              .try_emplace(realization.reference.entity,
                                           static_cast<PnrIndex>(ordinal))
                              .second;
    if (!inserted)
      return invalid(Projection::ComputePlacement,
                     "compute realization reference is not unique");
  }
  const PnrIndex memoryDecisionOffset = computeDecisionCount;
  for (auto [ordinal, realization] :
       llvm::enumerate(realizations.memoryRealizations())) {
    const bool inserted =
        memoryDecisions
            .try_emplace(realization.reference.entity,
                         memoryDecisionOffset + static_cast<PnrIndex>(ordinal))
            .second;
    if (!inserted)
      return invalid(Projection::MemoryPlacement,
                     "memory realization reference is not unique");
  }
  std::map<ProjectionKey, MemoryOperationPortDecision>
      memoryOperationPortDecisions;
  if (realizations.memoryActors().size() !=
      realizations.memoryActorRealizations().size())
    return invalid(Projection::MemoryOperationPort,
                   "memory actor ownership projection is incomplete");
  for (auto [actorOrdinal, actor] :
       llvm::enumerate(realizations.memoryActors())) {
    const PnrIndex realization =
        realizations.memoryActorRealizations()[actorOrdinal];
    if (realization >= memoryDecisionCount)
      return invalid(Projection::MemoryOperationPort,
                     "memory actor names a foreign realization");
    auto key = actorKey(dataflowIdentity, actor.actor);
    if (!key)
      return key.takeError();
    const bool inserted =
        memoryOperationPortDecisions
            .try_emplace(
                std::move(*key),
                MemoryOperationPortDecision{memoryDecisionOffset + realization,
                                            actor.operationPort.ordinal})
            .second;
    if (!inserted)
      return invalid(Projection::MemoryOperationPort,
                     "memory actor reference is not unique");
  }

  std::vector<InitializerRelationInput> relationInputs;
  std::vector<std::uint8_t> constraintRelations;
  std::optional<Projection> deferredProjection;
  for (std::size_t projectionOrdinal = 0;
       projectionOrdinal != FrozenConstraintIndex::projectionCount;
       ++projectionOrdinal) {
    const auto projection =
        ::mapping::symbolizeSpatialConstraintProjection(projectionOrdinal);
    if (!projection)
      return llvm::make_error<SpatialPnrFreezeFailure>(
          SpatialPnrFreezeFailureKind::Invalid,
          "Spatial constraint projection catalog is not dense");
    const FrozenConstraintShard &shard = constraints.shard(*projection);
    if (shard.equalityClasses().empty() && shard.disjointGroups().empty())
      continue;
    if (!isBindingProjection(*projection)) {
      if (isRouteProjection(*projection) ||
          *projection == Projection::NetAssignedTagValues ||
          *projection == Projection::MemoryBoundServices ||
          *projection == Projection::MemoryAddressRegion)
        continue;
      if (!deferredProjection)
        deferredProjection = *projection;
      continue;
    }

    const auto appendRelations =
        [&](llvm::ArrayRef<FrozenConstraintRelation> relations,
            InitializerRelationKind kind) -> llvm::Error {
      for (const FrozenConstraintRelation &relation : relations) {
        InitializerRelationInput input;
        input.kind = kind;
        std::vector<ProjectionKey> relationUniverse;
        std::vector<std::vector<ProjectionKey>> memberKeys;
        input.members.reserve(relation.memberCount);
        memberKeys.reserve(relation.memberCount);
        for (PnrIndex subjectOrdinal : shard.relationMembers().slice(
                 relation.memberOffset, relation.memberCount)) {
          if (subjectOrdinal >= shard.subjects().size())
            return invalid(*projection,
                           "relation contains an out-of-range subject");
          auto decision = relationDecision(
              *projection, shard.subjects()[subjectOrdinal], computeDecisions,
              memoryDecisions, memoryOperationPortDecisions,
              attachmentDecisions, dataflowIdentity);
          if (!decision)
            return decision.takeError();

          std::vector<ProjectionKey> keys;
          if (isComputeProjection(*projection)) {
            const PnrIndex realization = *decision;
            const auto choices =
                llvm::ArrayRef(computeChoices)
                    .slice(computeChoiceOffsets[realization],
                           computeChoiceOffsets[realization + 1] -
                               computeChoiceOffsets[realization]);
            keys.reserve(choices.size());
            for (const SpatialComputeBindingChoice &choice : choices)
              keys.push_back(
                  computeProjectionKey(*projection, realizations, choice));
          } else if (*projection == Projection::MemoryPlacement) {
            const PnrIndex realization = *decision - memoryDecisionOffset;
            const auto choices =
                llvm::ArrayRef(memoryChoices)
                    .slice(memoryChoiceOffsets[realization],
                           memoryChoiceOffsets[realization + 1] -
                               memoryChoiceOffsets[realization]);
            keys.reserve(choices.size());
            for (const SpatialMemoryBindingChoice &choice : choices)
              keys.push_back(memoryProjectionKey(realizations, choice));
          } else if (*projection == Projection::MemoryOperationPort) {
            const auto *actor = std::get_if<dataflow::ActorRef>(
                &shard.subjects()[subjectOrdinal]);
            assert(actor && "validated memory operation-port subject");
            auto key = actorKey(dataflowIdentity, *actor);
            if (!key)
              return key.takeError();
            const auto owner = memoryOperationPortDecisions.find(*key);
            assert(owner != memoryOperationPortDecisions.end());
            const PnrIndex realization = *decision - memoryDecisionOffset;
            const auto choices =
                llvm::ArrayRef(memoryChoices)
                    .slice(memoryChoiceOffsets[realization],
                           memoryChoiceOffsets[realization + 1] -
                               memoryChoiceOffsets[realization]);
            keys.reserve(choices.size());
            for (const SpatialMemoryBindingChoice &choice : choices)
              keys.push_back(memoryOperationPortProjectionKey(
                  realizations, choice, owner->second.portOrdinal));
          } else {
            llvm::ArrayRef<PnrIndex> choices;
            if (*decision < graphBoundaryDecisionOffset) {
              const PnrIndex demand = *decision - portDecisionOffset;
              choices = llvm::ArrayRef(attachmentChoices)
                            .slice(portAttachmentChoiceOffsets[demand],
                                   portAttachmentChoiceOffsets[demand + 1] -
                                       portAttachmentChoiceOffsets[demand]);
            } else {
              const PnrIndex boundary = *decision - graphBoundaryDecisionOffset;
              choices =
                  llvm::ArrayRef(attachmentChoices)
                      .slice(
                          graphBoundaryAttachmentChoiceOffsets[boundary],
                          graphBoundaryAttachmentChoiceOffsets[boundary + 1] -
                              graphBoundaryAttachmentChoiceOffsets[boundary]);
            }
            keys.reserve(choices.size());
            for (PnrIndex option : choices) {
              if (option >= ports.attachmentOptions().size() ||
                  ports.attachmentOptions()[option].endpoint >=
                      routing.routingEndpoints().size())
                return invalid(*projection,
                               "attachment relation choice is malformed");
              keys.push_back(canonicalFabricBytes(
                  routing
                      .routingEndpoints()[ports.attachmentOptions()[option]
                                              .endpoint]
                      .reference));
            }
          }
          relationUniverse.insert(relationUniverse.end(), keys.begin(),
                                  keys.end());
          memberKeys.push_back(std::move(keys));
          input.members.push_back({*decision, {}});
        }

        llvm::sort(relationUniverse);
        relationUniverse.erase(
            std::unique(relationUniverse.begin(), relationUniverse.end()),
            relationUniverse.end());
        if (relationUniverse.size() > getPnrIndexMax())
          return invalid(*projection,
                         "relation value domain overflows PnrIndex");
        for (std::size_t member = 0; member < input.members.size(); ++member) {
          input.members[member].projectedValues.reserve(
              memberKeys[member].size());
          for (const ProjectionKey &key : memberKeys[member]) {
            const auto found = llvm::lower_bound(relationUniverse, key);
            assert(found != relationUniverse.end() && *found == key);
            input.members[member].projectedValues.push_back(
                static_cast<PnrIndex>(found - relationUniverse.begin()));
          }
        }
        relationInputs.push_back(std::move(input));
        constraintRelations.push_back(1);
      }
      return llvm::Error::success();
    };

    if (llvm::Error error = appendRelations(shard.equalityClasses(),
                                            InitializerRelationKind::Equal))
      return std::move(error);
    if (llvm::Error error = appendRelations(shard.disjointGroups(),
                                            InitializerRelationKind::Disjoint))
      return std::move(error);
  }

  for (PnrIndex demand = 0; demand < ports.portDemands().size(); ++demand) {
    const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
    const PnrIndex rootDecision =
        record.kind == FrozenSpatialPortDemandKind::Compute
            ? record.realization
            : memoryDecisionOffset + record.realization;
    if (rootDecision >= portDecisionOffset)
      return invalid(Projection::SpatialTransferAttachment,
                     "PortDemand realization owner is out of range");
    InitializerRelationInput compatibility;
    compatibility.kind = InitializerRelationKind::Equal;
    InitializerRelationMemberInput root;
    root.decision = rootDecision;
    if (record.kind == FrozenSpatialPortDemandKind::Compute) {
      for (const SpatialComputeBindingChoice &choice :
           llvm::ArrayRef(computeChoices)
               .slice(computeChoiceOffsets[record.realization],
                      computeChoiceOffsets[record.realization + 1] -
                          computeChoiceOffsets[record.realization]))
        root.projectedValues.push_back(choice.placement);
    } else {
      for (const SpatialMemoryBindingChoice &choice :
           llvm::ArrayRef(memoryChoices)
               .slice(memoryChoiceOffsets[record.realization],
                      memoryChoiceOffsets[record.realization + 1] -
                          memoryChoiceOffsets[record.realization]))
        root.projectedValues.push_back(choice.placement);
    }
    InitializerRelationMemberInput attachment;
    attachment.decision = portDecisionOffset + demand;
    for (PnrIndex option :
         llvm::ArrayRef(attachmentChoices)
             .slice(portAttachmentChoiceOffsets[demand],
                    portAttachmentChoiceOffsets[demand + 1] -
                        portAttachmentChoiceOffsets[demand])) {
      const FrozenSpatialAttachmentOption &choice =
          ports.attachmentOptions()[option];
      if (choice.ownerKind !=
              FrozenSpatialAttachmentOwnerKind::PlacementDomain ||
          choice.owner >= ports.placementDomains().size())
        return invalid(Projection::SpatialTransferAttachment,
                       "attachment compatibility owner is malformed");
      attachment.projectedValues.push_back(
          ports.placementDomains()[choice.owner].placement);
    }
    compatibility.members.push_back(std::move(root));
    compatibility.members.push_back(std::move(attachment));
    relationInputs.push_back(std::move(compatibility));
    constraintRelations.push_back(0);
  }

  auto relations = InitializerRelationModel::create(
      std::move(decisionChoiceCounts), std::move(relationInputs));
  if (!relations)
    return relations.takeError();
  if (constraintRelations.size() != relations->relations().size())
    return invalid(Projection::SpatialTransferAttachment,
                   "relation ownership projection is incomplete");
  return std::shared_ptr<const SpatialBindingRelationModel>(
      new SpatialBindingRelationModel(
          std::move(*relations), std::move(computeChoiceOffsets),
          std::move(computeChoices), std::move(computeContextChoiceOrdinals),
          std::move(memoryChoiceOffsets), std::move(memoryChoices),
          std::move(memoryPlacementChoiceOrdinals),
          std::move(portAttachmentChoiceOffsets),
          std::move(graphBoundaryAttachmentChoiceOffsets),
          std::move(attachmentChoices),
          std::move(attachmentOptionChoiceOrdinals),
          std::move(constraintRelations), deferredProjection));
}

llvm::ArrayRef<SpatialComputeBindingChoice>
SpatialBindingRelationModel::computeChoices(PnrIndex realization) const {
  assert(realization + 1 < computeChoiceOffsets_.size());
  return llvm::ArrayRef(computeChoices_)
      .slice(computeChoiceOffsets_[realization],
             computeChoiceOffsets_[realization + 1] -
                 computeChoiceOffsets_[realization]);
}

llvm::ArrayRef<SpatialMemoryBindingChoice>
SpatialBindingRelationModel::memoryChoices(PnrIndex realization) const {
  assert(realization + 1 < memoryChoiceOffsets_.size());
  return llvm::ArrayRef(memoryChoices_)
      .slice(memoryChoiceOffsets_[realization],
             memoryChoiceOffsets_[realization + 1] -
                 memoryChoiceOffsets_[realization]);
}

std::optional<PnrIndex> SpatialBindingRelationModel::computeChoiceOrdinal(
    PnrIndex realization, PnrIndex placement,
    PnrIndex instructionContext) const {
  if (realization >= computeDecisionCount())
    return std::nullopt;
  const auto choices = computeChoices(realization);
  if (instructionContext >= computeContextChoiceOrdinals_.size())
    return std::nullopt;
  const PnrIndex local = computeContextChoiceOrdinals_[instructionContext];
  if (local >= choices.size())
    return std::nullopt;
  if (choices[local].placement != placement ||
      choices[local].instructionContext != instructionContext)
    return std::nullopt;
  return local;
}

std::optional<PnrIndex>
SpatialBindingRelationModel::memoryChoiceOrdinal(PnrIndex realization,
                                                 PnrIndex placement) const {
  if (realization + 1 >= memoryChoiceOffsets_.size())
    return std::nullopt;
  const auto choices = memoryChoices(realization);
  if (placement >= memoryPlacementChoiceOrdinals_.size())
    return std::nullopt;
  const PnrIndex local = memoryPlacementChoiceOrdinals_[placement];
  if (local >= choices.size())
    return std::nullopt;
  if (choices[local].placement != placement)
    return std::nullopt;
  return local;
}

llvm::ArrayRef<PnrIndex>
SpatialBindingRelationModel::portAttachmentChoices(PnrIndex demand) const {
  assert(demand + 1 < portAttachmentChoiceOffsets_.size());
  return llvm::ArrayRef(attachmentChoices_)
      .slice(portAttachmentChoiceOffsets_[demand],
             portAttachmentChoiceOffsets_[demand + 1] -
                 portAttachmentChoiceOffsets_[demand]);
}

llvm::ArrayRef<PnrIndex>
SpatialBindingRelationModel::graphBoundaryAttachmentChoices(
    PnrIndex boundary) const {
  assert(boundary + 1 < graphBoundaryAttachmentChoiceOffsets_.size());
  return llvm::ArrayRef(attachmentChoices_)
      .slice(graphBoundaryAttachmentChoiceOffsets_[boundary],
             graphBoundaryAttachmentChoiceOffsets_[boundary + 1] -
                 graphBoundaryAttachmentChoiceOffsets_[boundary]);
}

std::optional<PnrIndex>
SpatialBindingRelationModel::portAttachmentChoiceOrdinal(
    PnrIndex demand, PnrIndex option) const {
  if (demand >= portDecisionCount() ||
      option >= attachmentOptionChoiceOrdinals_.size())
    return std::nullopt;
  const PnrIndex local = attachmentOptionChoiceOrdinals_[option];
  const auto choices = portAttachmentChoices(demand);
  if (local >= choices.size() || choices[local] != option)
    return std::nullopt;
  return local;
}

std::optional<PnrIndex>
SpatialBindingRelationModel::graphBoundaryAttachmentChoiceOrdinal(
    PnrIndex boundary, PnrIndex option) const {
  if (boundary >= graphBoundaryDecisionCount() ||
      option >= attachmentOptionChoiceOrdinals_.size())
    return std::nullopt;
  const PnrIndex local = attachmentOptionChoiceOrdinals_[option];
  const auto choices = graphBoundaryAttachmentChoices(boundary);
  if (local >= choices.size() || choices[local] != option)
    return std::nullopt;
  return local;
}

llvm::ArrayRef<PnrIndex>
SpatialBindingRelationModel::decisionRelations(PnrIndex decision) const {
  assert(decision < decisionCount());
  const auto offsets = relations_.decisionRelationOffsets();
  return relations_.decisionRelations().slice(
      offsets[decision], offsets[decision + 1] - offsets[decision]);
}
