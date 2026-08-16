#include "SpatialPnrPortIndex.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialPnrProblem";
constexpr PnrCapacityContext demandIndexContext{
    frozenArtifact, "port_demands", "port_demands", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext demandOffsetContext{
    frozenArtifact, "realizations", "port_demands", PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext demandCountContext{
    frozenArtifact, "port_demands", "port_demands", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext boundaryIndexContext{
    frozenArtifact, "graph_boundaries", "graph_boundaries",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext domainIndexContext{
    frozenArtifact, "placement_domains", "placement_domains",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext domainOffsetContext{frozenArtifact, "port_demands",
                                                 "placement_domains",
                                                 PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext domainCountContext{
    frozenArtifact, "placement_domains", "placement_domains",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext optionIndexContext{
    frozenArtifact, "attachment_options", "attachment_options",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext optionOffsetContext{
    frozenArtifact, "attachment_domains", "attachment_options",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext optionCountContext{
    frozenArtifact, "attachment_options", "attachment_options",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext endpointIndexContext{
    frozenArtifact, "routing_endpoints", "routing_endpoints",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext traversalIndexContext{
    frozenArtifact, "traversals", "traversals", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext placementIndexContext{
    frozenArtifact, "placements", "placements", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext placementOffsetContext{
    frozenArtifact, "realizations", "placements", PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext placementCountContext{
    frozenArtifact, "placements", "placements", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext contextOffsetContext{
    frozenArtifact, "compute_placements", "instruction_contexts",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext contextCountContext{
    frozenArtifact, "instruction_contexts", "instruction_contexts",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext netIndexContext{
    frozenArtifact, "logical_nets", "logical_nets", PnrCapacityMeasure::Index};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

llvm::Error infeasible(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::ProvenInfeasible, message.str());
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

llvm::Error preflightAppend(PnrCapacityContext context, std::uint64_t current,
                            std::uint64_t added) {
  auto end = checkedPnrIndexAdd(context, current, added);
  if (!end)
    return end.takeError();
  return llvm::Error::success();
}

template <typename Ref>
llvm::Expected<std::vector<std::uint8_t>>
dataflowKey(const ArtifactIdentity &owner, const Ref &reference) {
  return ::dataflow::encodeDataflowReference(owner, reference);
}

struct ActorOwner final {
  FrozenSpatialPortDemandKind kind;
  PnrIndex realization;
  const TechComputeActorView *compute = nullptr;
  const TechMemoryActorView *memory = nullptr;
};

struct AttachmentDraft final {
  PnrIndex endpoint;
  std::optional<PnrIndex> localTraversal;
  SpatialDurableProgressBoundaryKind progressBoundary =
      SpatialDurableProgressBoundaryKind::None;
  std::optional<std::uint32_t> sharedOperandEnqueueUnit;

  friend bool operator<(const AttachmentDraft &lhs,
                        const AttachmentDraft &rhs) {
    return std::tie(lhs.endpoint, lhs.localTraversal, lhs.progressBoundary,
                    lhs.sharedOperandEnqueueUnit) <
           std::tie(rhs.endpoint, rhs.localTraversal, rhs.progressBoundary,
                    rhs.sharedOperandEnqueueUnit);
  }
  friend bool operator==(const AttachmentDraft &lhs,
                         const AttachmentDraft &rhs) {
    return lhs.endpoint == rhs.endpoint &&
           lhs.localTraversal == rhs.localTraversal &&
           lhs.progressBoundary == rhs.progressBoundary &&
           lhs.sharedOperandEnqueueUnit == rhs.sharedOperandEnqueueUnit;
  }
};

struct ComputeAttachmentClassKey final {
  FabricEntityId fu = 0;
  FabricEntityId parentPe = 0;
  ::fabric::Schedule schedule = ::fabric::Schedule::Spatial;
  FabricEntityId templateFu = 0;
  FabricPortDirection templateDirection = FabricPortDirection::Input;
  FabricOrdinal templateOrdinal = 0;
  FabricPortDirection terminalDirection = FabricPortDirection::Input;
  std::uint32_t payloadWidthBits = 0;

  friend bool operator<(const ComputeAttachmentClassKey &lhs,
                        const ComputeAttachmentClassKey &rhs) {
    return std::tie(lhs.fu, lhs.parentPe, lhs.schedule, lhs.templateFu,
                    lhs.templateDirection, lhs.templateOrdinal,
                    lhs.terminalDirection, lhs.payloadWidthBits) <
           std::tie(rhs.fu, rhs.parentPe, rhs.schedule, rhs.templateFu,
                    rhs.templateDirection, rhs.templateOrdinal,
                    rhs.terminalDirection, rhs.payloadWidthBits);
  }
};

llvm::Expected<std::optional<std::uint32_t>>
deriveSharedOperandEnqueueUnit(const FabricArtifactView &fabric,
                               const FrozenSpatialComputePlacement &placement,
                               FabricFuOccurrencePortRef concretePort) {
  if (placement.schedule != ::fabric::Schedule::Temporal ||
      concretePort.direction != FabricPortDirection::Input)
    return std::optional<std::uint32_t>();
  auto mode = fabric.peOperandBufferMode(placement.parentPe);
  auto schema = fabric.temporalPeConfigurationSchema(placement.parentPe);
  if (!mode || !schema)
    return invalid("Temporal PE attachment has no operand-buffer schema");
  std::vector<std::uint32_t> fuInputCounts;
  fuInputCounts.reserve(schema->layout().fus.size());
  std::optional<FabricOrdinal> fuOrdinal;
  for (auto [ordinal, fu] : llvm::enumerate(schema->layout().fus)) {
    fuInputCounts.push_back(fu.inputCount);
    if (fu.fu == concretePort.fu)
      fuOrdinal = static_cast<FabricOrdinal>(ordinal);
  }
  if (!fuOrdinal || concretePort.ordinal >= fuInputCounts[*fuOrdinal])
    return invalid("Temporal PE attachment has no concrete operand queue");
  auto contract = ::fabric::TemporalOperandBufferContract::create(
      ::fabric::TemporalOperandBufferDeclaration{
          placement.parentPe, schema->layout().contextCount, fuInputCounts,
          *mode, fabric.peOperandBufferSize(placement.parentPe)});
  if (!contract)
    return contract.takeError();

  std::optional<std::uint32_t> sharedUnit;
  std::uint32_t matchedContexts = 0;
  for (auto [queue, key] : llvm::enumerate(contract->logicalQueues())) {
    if (key.fuOccurrence != *fuOrdinal || key.fuInput != concretePort.ordinal)
      continue;
    const std::uint32_t unit =
        contract->allocationUnitOf(static_cast<std::uint32_t>(queue));
    if (!sharedUnit)
      sharedUnit = unit;
    else if (*sharedUnit != unit)
      return std::optional<std::uint32_t>();
    ++matchedContexts;
  }
  if (matchedContexts != schema->layout().contextCount || !sharedUnit)
    return invalid("Temporal PE operand queue projection is incomplete");
  return sharedUnit;
}

struct PlacementDomainDraft final {
  PnrIndex placement;
  const std::vector<AttachmentDraft> *sharedOptions = nullptr;
  std::vector<AttachmentDraft> ownedOptions;

  PlacementDomainDraft(PnrIndex placement,
                       const std::vector<AttachmentDraft> *options)
      : placement(placement), sharedOptions(options) {}

  PlacementDomainDraft(PnrIndex placement, std::vector<AttachmentDraft> options)
      : placement(placement), ownedOptions(std::move(options)) {}

  llvm::ArrayRef<AttachmentDraft> options() const {
    if (sharedOptions)
      return *sharedOptions;
    return ownedOptions;
  }
};

struct PortDemandDraft final {
  FrozenSpatialPortDemand frozen;
  std::vector<PlacementDomainDraft> domains;
};

struct GraphBoundaryDraft final {
  FrozenSpatialGraphBoundary frozen;
  std::vector<AttachmentDraft> options;
};

llvm::Expected<std::uint32_t> transportPayloadWidth(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::dataflow::CanonicalGraphProducerEndpointRef &endpoint) {
  auto type = dataflow.tokenType(endpoint);
  if (!type)
    return type.takeError();
  return dataflow.transportPayloadBitWidth(*type);
}

llvm::Expected<std::uint32_t> transportPayloadWidth(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::dataflow::CanonicalGraphConsumerEndpointRef &endpoint) {
  auto type = dataflow.tokenType(endpoint);
  if (!type)
    return type.takeError();
  return dataflow.transportPayloadBitWidth(*type);
}

llvm::Expected<FabricFuTemplatePortRef>
computeTemplateTerminal(const TechComputeRealizationView &realization,
                        const FrozenSpatialActorTerminalRef &terminal) {
  const ::dataflow::ActorRef actor =
      std::visit([](const auto &endpoint) { return endpoint.actor; }, terminal);
  const FabricPortDirection direction =
      std::holds_alternative<::dataflow::ActorTokenOperandRef>(terminal)
          ? FabricPortDirection::Input
          : FabricPortDirection::Output;
  const std::uint64_t ordinal = std::visit(
      [](const auto &endpoint) { return endpoint.ordinal; }, terminal);
  const TechComputeBoundaryView *selected = nullptr;
  for (const TechComputeBoundaryView &boundary : realization.boundaries) {
    if (boundary.actor != actor || boundary.direction != direction ||
        boundary.portOrdinal != ordinal)
      continue;
    if (selected)
      return invalid("compute PortDemand has duplicate boundary witnesses");
    selected = &boundary;
  }
  if (!selected)
    return invalid("compute PortDemand has no exact boundary witness");
  return selected->fabricPort;
}

llvm::Expected<FabricMemoryEngineTemplateEndpointRef>
memoryTemplateTerminal(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                       const TechMemoryActorView &binding,
                       const FrozenSpatialActorTerminalRef &terminal) {
  return std::visit(
      [&](const auto &typed) {
        return ::loom::mapping::resolveTechMemoryActorTerminal(dataflow,
                                                               binding, typed);
      },
      terminal);
}

FabricPortDirection directionOf(const FrozenSpatialActorTerminalRef &terminal) {
  return std::holds_alternative<::dataflow::ActorTokenOperandRef>(terminal)
             ? FabricPortDirection::Input
             : FabricPortDirection::Output;
}

FabricPortDirection
directionOf(const FrozenSpatialGraphBoundaryTerminalRef &terminal) {
  return std::holds_alternative<::dataflow::GraphIngressTokenRef>(terminal)
             ? FabricPortDirection::Input
             : FabricPortDirection::Output;
}

void canonicalizeOptions(std::vector<AttachmentDraft> &options) {
  llvm::sort(options);
  options.erase(std::unique(options.begin(), options.end()), options.end());
}

} // namespace

class loom::pnr::FrozenSpatialPortIndexBuilder final {
public:
  static llvm::Expected<FrozenSpatialPortIndex>
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping, const FabricArtifactView &fabric,
        FrozenSpatialRealizationIndex &realizations,
        FrozenSpatialTransferIndex &transfers,
        const FrozenSpatialRoutingGraph &routing) {
    FrozenSpatialPortIndex result;

    std::map<std::vector<std::uint8_t>, PnrIndex> endpointByRef;
    for (auto [ordinal, endpoint] :
         llvm::enumerate(routing.routingEndpoints())) {
      auto index = checked(endpointIndexContext, ordinal);
      if (!index)
        return index.takeError();
      if (!endpointByRef
               .emplace(canonicalFabricBytes(endpoint.reference), *index)
               .second)
        return invalid("routing endpoint inventory contains a duplicate");
    }
    std::map<std::vector<std::uint8_t>, PnrIndex> traversalByRef;
    for (auto [ordinal, traversal] : llvm::enumerate(routing.traversals())) {
      auto index = checked(traversalIndexContext, ordinal);
      if (!index)
        return index.takeError();
      if (!traversalByRef
               .emplace(canonicalFabricBytes(traversal.reference), *index)
               .second)
        return invalid("routing traversal inventory contains a duplicate");
    }

    std::map<std::uint64_t, ActorOwner> actorOwners;
    for (auto [realizationOrdinal, realization] :
         llvm::enumerate(techMapping.computeRealizations())) {
      auto realizationIndex = checked(demandIndexContext, realizationOrdinal);
      if (!realizationIndex)
        return realizationIndex.takeError();
      for (const TechComputeActorView &actor : realization.actors)
        if (!actorOwners
                 .emplace(actor.actor.entity.value(),
                          ActorOwner{FrozenSpatialPortDemandKind::Compute,
                                     *realizationIndex, &actor, nullptr})
                 .second)
          return invalid("an actor belongs to multiple Tech Realizations");
    }
    for (auto [realizationOrdinal, realization] :
         llvm::enumerate(techMapping.memoryRealizations())) {
      auto realizationIndex = checked(demandIndexContext, realizationOrdinal);
      if (!realizationIndex)
        return realizationIndex.takeError();
      for (const TechMemoryActorView &actor : realization.actors)
        if (!actorOwners
                 .emplace(actor.actor.entity.value(),
                          ActorOwner{FrozenSpatialPortDemandKind::Memory,
                                     *realizationIndex, nullptr, &actor})
                 .second)
          return invalid("an actor belongs to multiple Tech Realizations");
    }

    std::vector<PortDemandDraft> demands;
    std::vector<GraphBoundaryDraft> graphBoundaries;
    std::map<std::vector<std::uint8_t>, PnrIndex> demandByTerminal;
    std::map<std::vector<std::uint8_t>, PnrIndex> boundaryByTerminal;

    const auto addActorDemand = [&](FrozenSpatialActorTerminalRef terminal,
                                    PnrIndex logicalNet,
                                    std::vector<std::uint8_t> key)
        -> llvm::Expected<FrozenSpatialTerminalBinding> {
      auto existing = demandByTerminal.find(key);
      if (existing != demandByTerminal.end()) {
        if (demands[existing->second].frozen.logicalNet != logicalNet)
          return invalid("one actor terminal belongs to multiple logical nets");
        return FrozenSpatialTerminalBinding{
            FrozenSpatialTerminalBindingKind::PortDemand, existing->second};
      }

      const ::dataflow::ActorRef actor = std::visit(
          [](const auto &endpoint) { return endpoint.actor; }, terminal);
      const auto owner = actorOwners.find(actor.entity.value());
      if (owner == actorOwners.end())
        return invalid("residual logical net names an unmapped actor terminal");

      FrozenSpatialTemplateTerminalRef templateTerminal;
      if (owner->second.kind == FrozenSpatialPortDemandKind::Compute) {
        auto endpoint = computeTemplateTerminal(
            techMapping.computeRealizations()[owner->second.realization],
            terminal);
        if (!endpoint)
          return endpoint.takeError();
        templateTerminal = *endpoint;
      } else {
        auto endpoint =
            memoryTemplateTerminal(dataflow, *owner->second.memory, terminal);
        if (!endpoint)
          return endpoint.takeError();
        templateTerminal = *endpoint;
      }

      llvm::Expected<std::uint32_t> width = std::visit(
          [&](const auto &endpoint) {
            using Endpoint = std::decay_t<decltype(endpoint)>;
            if constexpr (std::is_same_v<Endpoint,
                                         ::dataflow::ActorTokenResultRef>)
              return transportPayloadWidth(
                  dataflow,
                  ::dataflow::CanonicalGraphProducerEndpointRef{endpoint});
            else
              return transportPayloadWidth(
                  dataflow,
                  ::dataflow::CanonicalGraphConsumerEndpointRef{endpoint});
          },
          terminal);
      if (!width)
        return width.takeError();
      auto demandIndex = checked(demandIndexContext, demands.size());
      if (!demandIndex)
        return demandIndex.takeError();
      demands.push_back({FrozenSpatialPortDemand{
                             owner->second.kind, owner->second.realization,
                             std::move(terminal), std::move(templateTerminal),
                             *width, logicalNet, 0, 0},
                         {}});
      demandByTerminal.emplace(std::move(key), *demandIndex);
      return FrozenSpatialTerminalBinding{
          FrozenSpatialTerminalBindingKind::PortDemand, *demandIndex};
    };

    const auto addGraphBoundary =
        [&](FrozenSpatialGraphBoundaryTerminalRef terminal, PnrIndex logicalNet,
            std::vector<std::uint8_t> key, std::uint32_t payloadWidth)
        -> llvm::Expected<FrozenSpatialTerminalBinding> {
      auto existing = boundaryByTerminal.find(key);
      if (existing != boundaryByTerminal.end()) {
        if (graphBoundaries[existing->second].frozen.logicalNet != logicalNet)
          return invalid(
              "one graph boundary terminal belongs to multiple logical nets");
        return FrozenSpatialTerminalBinding{
            FrozenSpatialTerminalBindingKind::GraphBoundary, existing->second};
      }
      auto boundaryIndex =
          checked(boundaryIndexContext, graphBoundaries.size());
      if (!boundaryIndex)
        return boundaryIndex.takeError();
      graphBoundaries.push_back(
          {FrozenSpatialGraphBoundary{std::move(terminal), payloadWidth,
                                      logicalNet, 0, 0},
           {}});
      boundaryByTerminal.emplace(std::move(key), *boundaryIndex);
      return FrozenSpatialTerminalBinding{
          FrozenSpatialTerminalBindingKind::GraphBoundary, *boundaryIndex};
    };

    transfers.logicalNetSourceBindings_.assign(transfers.logicalNets_.size(),
                                               {});
    transfers.logicalNetSinkBindings_.assign(transfers.logicalNetSinks_.size(),
                                             {});
    for (auto [netOrdinal, net] : llvm::enumerate(transfers.logicalNets_)) {
      auto netIndex = checked(netIndexContext, netOrdinal);
      if (!netIndex)
        return netIndex.takeError();
      const auto &producer = net.producer;
      auto producerKey = dataflowKey(dataflow.identity(), producer);
      if (!producerKey)
        return producerKey.takeError();
      llvm::Expected<FrozenSpatialTerminalBinding> source =
          [&]() -> llvm::Expected<FrozenSpatialTerminalBinding> {
        if (const auto *actor =
                std::get_if<::dataflow::ActorTokenResultRef>(&producer))
          return addActorDemand(*actor, *netIndex, std::move(*producerKey));
        const auto &ingress =
            std::get<::dataflow::GraphIngressTokenRef>(producer);
        auto width = transportPayloadWidth(dataflow, producer);
        if (!width)
          return width.takeError();
        return addGraphBoundary(ingress, *netIndex, std::move(*producerKey),
                                *width);
      }();
      if (!source)
        return source.takeError();
      transfers.logicalNetSourceBindings_[netOrdinal] = *source;

      for (std::size_t sinkOrdinal = 0; sinkOrdinal < net.sinkCount;
           ++sinkOrdinal) {
        const std::size_t absoluteSink =
            static_cast<std::size_t>(net.sinkOffset) + sinkOrdinal;
        const auto &consumer = transfers.logicalNetSinks_[absoluteSink];
        auto consumerKey = dataflowKey(dataflow.identity(), consumer);
        if (!consumerKey)
          return consumerKey.takeError();
        llvm::Expected<FrozenSpatialTerminalBinding> sink =
            [&]() -> llvm::Expected<FrozenSpatialTerminalBinding> {
          if (const auto *actor =
                  std::get_if<::dataflow::ActorTokenOperandRef>(&consumer))
            return addActorDemand(*actor, *netIndex, std::move(*consumerKey));
          const auto &egress =
              std::get<::dataflow::GraphEgressTokenRef>(consumer);
          auto width = transportPayloadWidth(dataflow, consumer);
          if (!width)
            return width.takeError();
          return addGraphBoundary(egress, *netIndex, std::move(*consumerKey),
                                  *width);
        }();
        if (!sink)
          return sink.takeError();
        transfers.logicalNetSinkBindings_[absoluteSink] = *sink;
      }
    }

    if (llvm::Error error =
            preflightPnrIndexCapacity(demandCountContext, demands.size()))
      return std::move(error);
    std::vector<std::vector<PnrIndex>> computeDemands(
        realizations.computeRealizations_.size());
    std::vector<std::vector<PnrIndex>> memoryDemands(
        realizations.memoryRealizations_.size());
    for (auto [ordinal, demand] : llvm::enumerate(demands)) {
      auto index = checked(demandIndexContext, ordinal);
      if (!index)
        return index.takeError();
      auto &ownerDemands =
          demand.frozen.kind == FrozenSpatialPortDemandKind::Compute
              ? computeDemands[demand.frozen.realization]
              : memoryDemands[demand.frozen.realization];
      ownerDemands.push_back(*index);
    }

    const auto endpointIndex = [&](const FabricTransportEndpointRef &endpoint)
        -> std::optional<PnrIndex> {
      auto found = endpointByRef.find(canonicalFabricBytes(endpoint));
      return found == endpointByRef.end()
                 ? std::nullopt
                 : std::optional<PnrIndex>(found->second);
    };
    std::map<ComputeAttachmentClassKey, std::vector<AttachmentDraft>>
        computeAttachmentClasses;
    const auto computeOptions =
        [&](const FrozenSpatialPortDemand &demand,
            const FrozenSpatialComputePlacement &placement)
        -> llvm::Expected<const std::vector<AttachmentDraft> *> {
      const auto &templatePort =
          std::get<FabricFuTemplatePortRef>(demand.templateTerminal);
      const ComputeAttachmentClassKey key{
          placement.fu.id(),
          placement.parentPe.id(),
          placement.schedule,
          templatePort.fu.id(),
          templatePort.direction,
          templatePort.ordinal,
          directionOf(demand.terminal),
          demand.payloadWidthBits,
      };
      auto [classIt, inserted] = computeAttachmentClasses.try_emplace(key);
      ++result.computeAttachmentClassLookupCount_;
      if (!inserted) {
        ++result.computeAttachmentClassHitCount_;
        return &classIt->second;
      }
      ++result.computeAttachmentClassMissCount_;

      std::vector<AttachmentDraft> options;
      const FabricFuOccurrencePortRef concretePort{
          placement.fu, templatePort.direction, templatePort.ordinal};
      auto sharedEnqueueUnit =
          deriveSharedOperandEnqueueUnit(fabric, placement, concretePort);
      if (!sharedEnqueueUnit)
        return sharedEnqueueUnit.takeError();
      const auto fixed = fabric.fuOccurrenceTransportEndpoint(concretePort);
      if (!fixed)
        return &classIt->second;
      const auto fixedIndex = endpointIndex(*fixed);
      if (!fixedIndex)
        return &classIt->second;
      const auto &fixedEndpoint = routing.routingEndpoints()[*fixedIndex];
      if (fixedEndpoint.direction != directionOf(demand.terminal) ||
          fixedEndpoint.dataPath.payloadWidthBits < demand.payloadWidthBits)
        return &classIt->second;

      for (const FabricFuPortAttachmentView &attachment :
           fabric.fuOccurrencePortAttachments(concretePort)) {
        const auto attachmentIndex = endpointIndex(attachment.endpoint);
        if (!attachmentIndex)
          continue;
        const auto &endpoint = routing.routingEndpoints()[*attachmentIndex];
        if (endpoint.direction != templatePort.direction ||
            endpoint.dataPath.payloadWidthBits < demand.payloadWidthBits)
          continue;
        const auto traversalIndex = traversalByRef.find(
            canonicalFabricBytes(attachment.localTraversal));
        if (traversalIndex == traversalByRef.end())
          continue;
        auto progress = classifySpatialAttachmentDurableProgressBoundary(
            fabric, attachment.localTraversal, concretePort);
        if (!progress)
          return progress.takeError();
        options.push_back({*attachmentIndex, traversalIndex->second, *progress,
                           *sharedEnqueueUnit});
      }
      canonicalizeOptions(options);
      classIt->second = std::move(options);
      return &classIt->second;
    };
    const auto memoryOptions =
        [&](const FrozenSpatialPortDemand &demand,
            const FrozenSpatialMemoryPlacement &placement) {
          std::vector<AttachmentDraft> options;
          const auto &templateEndpoint =
              std::get<FabricMemoryEngineTemplateEndpointRef>(
                  demand.templateTerminal);
          const FabricTransportEndpointRef endpoint{
              FabricTransportEndpointOwnerRef::of(placement.memory),
              templateEndpoint.ordinal};
          const auto index = endpointIndex(endpoint);
          if (!index)
            return options;
          const auto &resolved = routing.routingEndpoints()[*index];
          if (resolved.direction == directionOf(demand.terminal) &&
              resolved.dataPath.payloadWidthBits >= demand.payloadWidthBits)
            options.push_back({*index, std::nullopt,
                               SpatialDurableProgressBoundaryKind::None,
                               std::nullopt});
          return options;
        };

    std::vector<FrozenSpatialComputePlacement> oldComputePlacements =
        std::move(realizations.computePlacements_);
    std::vector<InstructionContextRef> oldContexts =
        std::move(realizations.computeInstructionContexts_);
    for (auto [realizationOrdinal, realization] :
         llvm::enumerate(realizations.computeRealizations_)) {
      auto realizationIndex = checked(demandIndexContext, realizationOrdinal);
      if (!realizationIndex)
        return realizationIndex.takeError();
      auto newOffset = checked(placementOffsetContext,
                               realizations.computePlacements_.size());
      if (!newOffset)
        return newOffset.takeError();
      const auto ownerDemands = computeDemands[realizationOrdinal];
      for (const FrozenSpatialComputePlacement &placement :
           llvm::ArrayRef<FrozenSpatialComputePlacement>(oldComputePlacements)
               .slice(realization.placementOffset,
                      realization.placementCount)) {
        std::vector<const std::vector<AttachmentDraft> *> options;
        options.reserve(ownerDemands.size());
        bool admissible = true;
        for (PnrIndex demand : ownerDemands) {
          auto projected = computeOptions(demands[demand].frozen, placement);
          if (!projected)
            return projected.takeError();
          options.push_back(*projected);
          if (options.back()->empty()) {
            admissible = false;
            break;
          }
        }
        if (!admissible)
          continue;
        auto placementIndex = checked(placementIndexContext,
                                      realizations.computePlacements_.size());
        if (!placementIndex)
          return placementIndex.takeError();
        auto newContextOffset =
            checked(contextOffsetContext,
                    realizations.computeInstructionContexts_.size());
        if (!newContextOffset)
          return newContextOffset.takeError();
        const auto contexts =
            llvm::ArrayRef<InstructionContextRef>(oldContexts)
                .slice(placement.contextOffset, placement.contextCount);
        if (llvm::Error error =
                preflightAppend(contextCountContext,
                                realizations.computeInstructionContexts_.size(),
                                contexts.size()))
          return std::move(error);
        realizations.computeInstructionContexts_.insert(
            realizations.computeInstructionContexts_.end(), contexts.begin(),
            contexts.end());
        auto contextCount = checked(contextCountContext, contexts.size());
        if (!contextCount)
          return contextCount.takeError();
        realizations.computePlacements_.push_back(
            {*realizationIndex, placement.fu, placement.parentPe,
             placement.schedule, *newContextOffset, *contextCount});
        for (auto [demandOrdinal, demand] : llvm::enumerate(ownerDemands))
          demands[demand].domains.emplace_back(*placementIndex,
                                               options[demandOrdinal]);
      }
      const std::size_t countValue =
          realizations.computePlacements_.size() - *newOffset;
      if (countValue == 0)
        return infeasible(
            "a compute realization has no occurrence with complete port "
            "attachment domains");
      auto count = checked(placementCountContext, countValue);
      if (!count)
        return count.takeError();
      realization.placementOffset = *newOffset;
      realization.placementCount = *count;
    }

    std::vector<FrozenSpatialMemoryPlacement> oldMemoryPlacements =
        std::move(realizations.memoryPlacements_);
    for (auto [realizationOrdinal, realization] :
         llvm::enumerate(realizations.memoryRealizations_)) {
      auto realizationIndex = checked(demandIndexContext, realizationOrdinal);
      if (!realizationIndex)
        return realizationIndex.takeError();
      auto newOffset = checked(placementOffsetContext,
                               realizations.memoryPlacements_.size());
      if (!newOffset)
        return newOffset.takeError();
      const auto ownerDemands = memoryDemands[realizationOrdinal];
      for (const FrozenSpatialMemoryPlacement &placement :
           llvm::ArrayRef<FrozenSpatialMemoryPlacement>(oldMemoryPlacements)
               .slice(realization.placementOffset,
                      realization.placementCount)) {
        std::vector<std::vector<AttachmentDraft>> options;
        options.reserve(ownerDemands.size());
        bool admissible = true;
        for (PnrIndex demand : ownerDemands) {
          options.push_back(memoryOptions(demands[demand].frozen, placement));
          if (options.back().empty()) {
            admissible = false;
            break;
          }
        }
        if (!admissible)
          continue;
        auto placementIndex = checked(placementIndexContext,
                                      realizations.memoryPlacements_.size());
        if (!placementIndex)
          return placementIndex.takeError();
        realizations.memoryPlacements_.push_back(
            {*realizationIndex, placement.memory, placement.schedule,
             placement.residentContextCount});
        for (auto [demandOrdinal, demand] : llvm::enumerate(ownerDemands))
          demands[demand].domains.emplace_back(
              *placementIndex, std::move(options[demandOrdinal]));
      }
      const std::size_t countValue =
          realizations.memoryPlacements_.size() - *newOffset;
      if (countValue == 0)
        return infeasible(
            "a memory realization has no occurrence with complete port "
            "attachment domains");
      auto count = checked(placementCountContext, countValue);
      if (!count)
        return count.takeError();
      realization.placementOffset = *newOffset;
      realization.placementCount = *count;
    }

    const auto appendDemandCsr =
        [&](const std::vector<std::vector<PnrIndex>> &lists,
            std::vector<PnrIndex> &offsets,
            std::vector<PnrIndex> &values) -> llvm::Error {
      offsets.reserve(lists.size() + 1);
      for (const auto &list : lists) {
        auto offset = checked(demandOffsetContext, values.size());
        if (!offset)
          return offset.takeError();
        offsets.push_back(*offset);
        if (llvm::Error error =
                preflightAppend(demandCountContext, values.size(), list.size()))
          return error;
        values.insert(values.end(), list.begin(), list.end());
      }
      auto end = checked(demandOffsetContext, values.size());
      if (!end)
        return end.takeError();
      offsets.push_back(*end);
      return llvm::Error::success();
    };
    if (llvm::Error error = appendDemandCsr(
            computeDemands, result.computeRealizationDemandOffsets_,
            result.computeRealizationDemands_))
      return std::move(error);
    if (llvm::Error error = appendDemandCsr(
            memoryDemands, result.memoryRealizationDemandOffsets_,
            result.memoryRealizationDemands_))
      return std::move(error);

    result.portDemands_.reserve(demands.size());
    for (PortDemandDraft &demand : demands) {
      auto domainOffset =
          checked(domainOffsetContext, result.placementDomains_.size());
      if (!domainOffset)
        return domainOffset.takeError();
      for (PlacementDomainDraft &domain : demand.domains) {
        const llvm::ArrayRef<AttachmentDraft> options = domain.options();
        auto domainIndex =
            checked(domainIndexContext, result.placementDomains_.size());
        if (!domainIndex)
          return domainIndex.takeError();
        auto optionOffset =
            checked(optionOffsetContext, result.attachmentOptions_.size());
        if (!optionOffset)
          return optionOffset.takeError();
        if (llvm::Error error = preflightAppend(
                optionCountContext, result.attachmentOptions_.size(),
                options.size()))
          return std::move(error);
        for (const AttachmentDraft &option : options)
          result.attachmentOptions_.push_back(
              {option.endpoint, option.localTraversal, option.progressBoundary,
               FrozenSpatialAttachmentOwnerKind::PlacementDomain, *domainIndex,
               option.sharedOperandEnqueueUnit});
        auto optionCount = checked(optionCountContext, options.size());
        if (!optionCount)
          return optionCount.takeError();
        result.placementDomains_.push_back(
            {domain.placement, *optionOffset, *optionCount});
      }
      auto domainCount = checked(domainCountContext, demand.domains.size());
      if (!domainCount)
        return domainCount.takeError();
      demand.frozen.placementDomainOffset = *domainOffset;
      demand.frozen.placementDomainCount = *domainCount;
      result.portDemands_.push_back(std::move(demand.frozen));
    }

    for (GraphBoundaryDraft &boundary : graphBoundaries) {
      const FabricPortDirection direction =
          directionOf(boundary.frozen.terminal);
      for (const FabricModuleBoundaryTransportAttachmentView &attachment :
           fabric.moduleBoundaryTransportAttachments()) {
        if (attachment.boundary.direction != direction)
          continue;
        const auto index = endpointIndex(attachment.endpoint);
        if (!index)
          return invalid(
              "Module boundary attachment endpoint is absent from routing");
        const auto &endpoint = routing.routingEndpoints()[*index];
        if (endpoint.direction != direction ||
            endpoint.dataPath.payloadWidthBits <
                boundary.frozen.payloadWidthBits)
          continue;
        boundary.options.push_back({*index, std::nullopt,
                                    SpatialDurableProgressBoundaryKind::None,
                                    std::nullopt});
      }
      canonicalizeOptions(boundary.options);
      if (boundary.options.empty())
        return infeasible(
            "a graph boundary terminal has no compatible Module attachment");
      auto boundaryIndex =
          checked(boundaryIndexContext, result.graphBoundaries_.size());
      if (!boundaryIndex)
        return boundaryIndex.takeError();
      auto optionOffset =
          checked(optionOffsetContext, result.attachmentOptions_.size());
      if (!optionOffset)
        return optionOffset.takeError();
      if (llvm::Error error = preflightAppend(optionCountContext,
                                              result.attachmentOptions_.size(),
                                              boundary.options.size()))
        return std::move(error);
      for (const AttachmentDraft &option : boundary.options)
        result.attachmentOptions_.push_back(
            {option.endpoint, std::nullopt,
             SpatialDurableProgressBoundaryKind::None,
             FrozenSpatialAttachmentOwnerKind::GraphBoundary, *boundaryIndex,
             std::nullopt});
      auto optionCount = checked(optionCountContext, boundary.options.size());
      if (!optionCount)
        return optionCount.takeError();
      boundary.frozen.attachmentOptionOffset = *optionOffset;
      boundary.frozen.attachmentOptionCount = *optionCount;
      result.graphBoundaries_.push_back(std::move(boundary.frozen));
    }

    std::vector<std::vector<PnrIndex>> endpointIncidence(
        routing.routingEndpoints().size());
    for (auto [ordinal, option] : llvm::enumerate(result.attachmentOptions_)) {
      auto optionIndex = checked(optionIndexContext, ordinal);
      if (!optionIndex)
        return optionIndex.takeError();
      endpointIncidence[option.endpoint].push_back(*optionIndex);
    }
    result.endpointAttachmentOffsets_.reserve(endpointIncidence.size() + 1);
    for (const auto &incidence : endpointIncidence) {
      auto offset = checked(optionOffsetContext,
                            result.endpointAttachmentOptions_.size());
      if (!offset)
        return offset.takeError();
      result.endpointAttachmentOffsets_.push_back(*offset);
      if (llvm::Error error = preflightAppend(
              optionCountContext, result.endpointAttachmentOptions_.size(),
              incidence.size()))
        return std::move(error);
      result.endpointAttachmentOptions_.insert(
          result.endpointAttachmentOptions_.end(), incidence.begin(),
          incidence.end());
    }
    auto incidenceEnd =
        checked(optionOffsetContext, result.endpointAttachmentOptions_.size());
    if (!incidenceEnd)
      return incidenceEnd.takeError();
    result.endpointAttachmentOffsets_.push_back(*incidenceEnd);
    return result;
  }
};

llvm::Expected<FrozenSpatialPortIndex>
loom::pnr::detail::buildFrozenSpatialPortIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    FrozenSpatialRealizationIndex &realizations,
    FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialRoutingGraph &routing) {
  return FrozenSpatialPortIndexBuilder::build(dataflow, techMapping, fabric,
                                              realizations, transfers, routing);
}

llvm::Error loom::pnr::detail::verifyFrozenSpatialPortIndex(
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialPortIndex &ports,
    const FrozenSpatialRoutingGraph &routing) {
  const auto rangeFits = [](PnrIndex offset, PnrIndex count, std::size_t size) {
    const std::size_t begin = static_cast<std::size_t>(offset);
    const std::size_t length = static_cast<std::size_t>(count);
    return begin <= size && length <= size - begin;
  };
  using FuPortKey =
      std::tuple<FabricEntityId, FabricPortDirection, FabricOrdinal>;
  using FuDirectionKey = std::pair<FabricEntityId, FabricPortDirection>;
  std::map<FuDirectionKey, FabricOrdinal> nextFuPortOrdinal;
  std::map<FuPortKey, PnrIndex> fuPortEndpoints;
  for (auto [index, endpoint] : llvm::enumerate(routing.routingEndpoints())) {
    if (endpoint.reference.owner.kind() !=
        FabricTransportEndpointOwnerKind::FabricFuOccurrence)
      continue;
    if (index > getPnrIndexMax())
      return invalid("FU endpoint index exceeds PnrIndex");
    const FabricFuOccurrenceRef fu =
        std::get<FabricFuOccurrenceRef>(endpoint.reference.owner.payload);
    FabricOrdinal &ordinal = nextFuPortOrdinal[{fu.id(), endpoint.direction}];
    if (!fuPortEndpoints
             .try_emplace({fu.id(), endpoint.direction, ordinal++},
                          static_cast<PnrIndex>(index))
             .second)
      return invalid("FU endpoint lookup contains a duplicate");
  }
  const auto fuPortEndpoint =
      [&](FabricFuOccurrenceRef fu, FabricPortDirection direction,
          FabricOrdinal directionOrdinal) -> std::optional<PnrIndex> {
    const auto found =
        fuPortEndpoints.find({fu.id(), direction, directionOrdinal});
    return found == fuPortEndpoints.end()
               ? std::nullopt
               : std::optional<PnrIndex>(found->second);
  };

  if (transfers.logicalNetSourceBindings().size() !=
          transfers.logicalNets().size() ||
      transfers.logicalNetSinkBindings().size() !=
          transfers.logicalNetSinks().size())
    return invalid(
        "logical-net attachment binding dimensions are inconsistent");

  const auto bindingInRange = [&](FrozenSpatialTerminalBinding binding) {
    return binding.kind == FrozenSpatialTerminalBindingKind::PortDemand
               ? binding.index < ports.portDemands().size()
               : binding.index < ports.graphBoundaries().size();
  };
  for (auto [netOrdinal, net] : llvm::enumerate(transfers.logicalNets())) {
    const FrozenSpatialTerminalBinding source =
        transfers.logicalNetSourceBindings()[netOrdinal];
    if (!bindingInRange(source))
      return invalid("logical-net source attachment is out of range");
    if (source.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
      const FrozenSpatialPortDemand &demand = ports.portDemands()[source.index];
      const auto *terminal =
          std::get_if<::dataflow::ActorTokenResultRef>(&demand.terminal);
      const auto *producer =
          std::get_if<::dataflow::ActorTokenResultRef>(&net.producer);
      if (!terminal || !producer || *terminal != *producer ||
          demand.logicalNet != netOrdinal)
        return invalid("logical-net source PortDemand does not match its net");
    } else {
      const FrozenSpatialGraphBoundary &boundary =
          ports.graphBoundaries()[source.index];
      const auto *terminal =
          std::get_if<::dataflow::GraphIngressTokenRef>(&boundary.terminal);
      const auto *producer =
          std::get_if<::dataflow::GraphIngressTokenRef>(&net.producer);
      if (!terminal || !producer || *terminal != *producer ||
          boundary.logicalNet != netOrdinal)
        return invalid(
            "logical-net source graph boundary does not match its net");
    }

    for (std::size_t ordinal = 0; ordinal < net.sinkCount; ++ordinal) {
      const std::size_t sinkIndex =
          static_cast<std::size_t>(net.sinkOffset) + ordinal;
      const auto &consumer = transfers.logicalNetSinks()[sinkIndex];
      const FrozenSpatialTerminalBinding sink =
          transfers.logicalNetSinkBindings()[sinkIndex];
      if (!bindingInRange(sink))
        return invalid("logical-net sink attachment is out of range");
      if (sink.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
        const FrozenSpatialPortDemand &demand = ports.portDemands()[sink.index];
        const auto *terminal =
            std::get_if<::dataflow::ActorTokenOperandRef>(&demand.terminal);
        const auto *operand =
            std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
        if (!terminal || !operand || *terminal != *operand ||
            demand.logicalNet != netOrdinal)
          return invalid("logical-net sink PortDemand does not match its net");
      } else {
        const FrozenSpatialGraphBoundary &boundary =
            ports.graphBoundaries()[sink.index];
        const auto *terminal =
            std::get_if<::dataflow::GraphEgressTokenRef>(&boundary.terminal);
        const auto *egress =
            std::get_if<::dataflow::GraphEgressTokenRef>(&consumer);
        if (!terminal || !egress || *terminal != *egress ||
            boundary.logicalNet != netOrdinal)
          return invalid(
              "logical-net sink graph boundary does not match its net");
      }
    }
  }

  const auto verifyRealizationDemandCsr =
      [&](llvm::ArrayRef<PnrIndex> offsets, llvm::ArrayRef<PnrIndex> values,
          std::size_t realizationCount,
          FrozenSpatialPortDemandKind kind) -> llvm::Error {
    if (offsets.size() != realizationCount + 1 || offsets.empty() ||
        offsets.front() != 0 || offsets.back() != values.size())
      return invalid(
          "realization-to-PortDemand CSR dimensions are inconsistent");
    std::vector<bool> seen(ports.portDemands().size(), false);
    for (std::size_t realization = 0; realization < realizationCount;
         ++realization) {
      const PnrIndex begin = offsets[realization];
      const PnrIndex end = offsets[realization + 1];
      if (begin > end || end > values.size())
        return invalid(
            "realization-to-PortDemand CSR offsets are inconsistent");
      for (PnrIndex cursor = begin; cursor < end; ++cursor) {
        const PnrIndex demand = values[cursor];
        if (demand >= ports.portDemands().size() || seen[demand])
          return invalid("realization-to-PortDemand incidence is invalid");
        const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
        if (record.kind != kind || record.realization != realization)
          return invalid("PortDemand is attached to the wrong realization");
        seen[demand] = true;
      }
    }
    for (auto [ordinal, demand] : llvm::enumerate(ports.portDemands()))
      if (demand.kind == kind && !seen[ordinal])
        return invalid("realization-to-PortDemand incidence is incomplete");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          verifyRealizationDemandCsr(ports.computeRealizationDemandOffsets(),
                                     ports.computeRealizationDemands(),
                                     realizations.computeRealizations().size(),
                                     FrozenSpatialPortDemandKind::Compute))
    return error;
  if (llvm::Error error =
          verifyRealizationDemandCsr(ports.memoryRealizationDemandOffsets(),
                                     ports.memoryRealizationDemands(),
                                     realizations.memoryRealizations().size(),
                                     FrozenSpatialPortDemandKind::Memory))
    return error;

  for (auto [demandOrdinal, demand] : llvm::enumerate(ports.portDemands())) {
    const FabricPortDirection direction = directionOf(demand.terminal);
    PnrIndex placementOffset = 0;
    PnrIndex placementCount = 0;
    if (demand.kind == FrozenSpatialPortDemandKind::Compute) {
      if (demand.realization >= realizations.computeRealizations().size() ||
          !std::holds_alternative<FabricFuTemplatePortRef>(
              demand.templateTerminal))
        return invalid("compute PortDemand has an invalid owner or terminal");
      const auto &realization =
          realizations.computeRealizations()[demand.realization];
      const auto &terminal =
          std::get<FabricFuTemplatePortRef>(demand.templateTerminal);
      if (terminal.fu != realization.capabilityTemplate.fu ||
          terminal.direction != direction)
        return invalid("compute PortDemand changed its template terminal");
      placementOffset = realization.placementOffset;
      placementCount = realization.placementCount;
    } else {
      if (demand.realization >= realizations.memoryRealizations().size() ||
          !std::holds_alternative<FabricMemoryEngineTemplateEndpointRef>(
              demand.templateTerminal))
        return invalid("memory PortDemand has an invalid owner or terminal");
      const auto &realization =
          realizations.memoryRealizations()[demand.realization];
      const auto &terminal = std::get<FabricMemoryEngineTemplateEndpointRef>(
          demand.templateTerminal);
      if (terminal.engine != realization.engine)
        return invalid("memory PortDemand changed its template terminal");
      placementOffset = realization.placementOffset;
      placementCount = realization.placementCount;
    }
    if (demand.placementDomainCount != placementCount ||
        !rangeFits(demand.placementDomainOffset, demand.placementDomainCount,
                   ports.placementDomains().size()))
      return invalid("PortDemand placement-domain slice is inconsistent");

    for (auto [localOrdinal, domain] :
         llvm::enumerate(ports.placementDomains().slice(
             demand.placementDomainOffset, demand.placementDomainCount))) {
      const PnrIndex domainIndex =
          demand.placementDomainOffset + static_cast<PnrIndex>(localOrdinal);
      if (domain.placement != placementOffset + localOrdinal ||
          domain.attachmentOptionCount == 0 ||
          !rangeFits(domain.attachmentOptionOffset,
                     domain.attachmentOptionCount,
                     ports.attachmentOptions().size()))
        return invalid("PortDemand attachment-domain slice is inconsistent");
      for (const FrozenSpatialAttachmentOption &option :
           ports.attachmentOptions().slice(domain.attachmentOptionOffset,
                                           domain.attachmentOptionCount)) {
        if (option.ownerKind !=
                FrozenSpatialAttachmentOwnerKind::PlacementDomain ||
            option.owner != domainIndex ||
            option.endpoint >= routing.routingEndpoints().size())
          return invalid("PortDemand attachment option has an invalid owner");
        const auto &endpoint = routing.routingEndpoints()[option.endpoint];
        if (endpoint.direction != direction ||
            endpoint.dataPath.payloadWidthBits < demand.payloadWidthBits)
          return invalid("PortDemand attachment endpoint is incompatible");

        if (demand.kind == FrozenSpatialPortDemandKind::Compute) {
          if (!option.localTraversal ||
              *option.localTraversal >= routing.traversals().size())
            return invalid(
                "compute PortDemand has no local selector traversal");
          const auto &placement =
              realizations.computePlacements()[domain.placement];
          const auto expectedBoundary =
              placement.schedule == ::fabric::Schedule::Temporal &&
                      direction == FabricPortDirection::Input
                  ? SpatialDurableProgressBoundaryKind::TemporalPeOperandQueue
                  : SpatialDurableProgressBoundaryKind::None;
          if (option.progressBoundary != expectedBoundary)
            return invalid(
                "compute attachment changed its durable progress boundary");
          if (option.sharedOperandEnqueueUnit &&
              expectedBoundary !=
                  SpatialDurableProgressBoundaryKind::TemporalPeOperandQueue)
            return invalid(
                "non-Temporal attachment gained a shared enqueue unit");
          if (endpoint.reference.owner.kind() !=
                  FabricTransportEndpointOwnerKind::FabricPeOccurrence ||
              std::get<FabricPeOccurrenceRef>(
                  endpoint.reference.owner.payload) != placement.parentPe)
            return invalid("compute PortDemand uses a foreign PE endpoint");
          const auto *selector = std::get_if<FabricPeSelectorPayload>(
              &routing.traversals()[*option.localTraversal].reference.payload);
          const auto &templatePort =
              std::get<FabricFuTemplatePortRef>(demand.templateTerminal);
          const auto fixed = fuPortEndpoint(
              placement.fu, templatePort.direction, templatePort.ordinal);
          if (!selector || selector->owner != placement.parentPe || !fixed)
            return invalid("compute PortDemand selector does not resolve");
          const auto &fixedEndpoint =
              routing.routingEndpoints()[*fixed].reference;
          if (direction == FabricPortDirection::Input) {
            if (selector->source != endpoint.reference ||
                selector->destination != fixedEndpoint)
              return invalid("compute input selector changed its endpoints");
          } else if (selector->source != fixedEndpoint ||
                     selector->destination != endpoint.reference) {
            return invalid("compute output selector changed its endpoints");
          }
        } else {
          if (option.localTraversal || option.sharedOperandEnqueueUnit ||
              option.progressBoundary !=
                  SpatialDurableProgressBoundaryKind::None)
            return invalid("memory PortDemand invented a local traversal");
          const auto &placement =
              realizations.memoryPlacements()[domain.placement];
          const auto &templateEndpoint =
              std::get<FabricMemoryEngineTemplateEndpointRef>(
                  demand.templateTerminal);
          if (endpoint.reference.owner.kind() !=
                  FabricTransportEndpointOwnerKind::FabricMemoryOccurrence ||
              std::get<FabricMemoryOccurrenceRef>(
                  endpoint.reference.owner.payload) != placement.memory ||
              endpoint.reference.ordinal != templateEndpoint.ordinal)
            return invalid("memory PortDemand changed its occurrence endpoint");
        }
      }
    }
    (void)demandOrdinal;
  }

  for (auto [boundaryOrdinal, boundary] :
       llvm::enumerate(ports.graphBoundaries())) {
    const FabricPortDirection direction = directionOf(boundary.terminal);
    if (boundary.attachmentOptionCount == 0 ||
        !rangeFits(boundary.attachmentOptionOffset,
                   boundary.attachmentOptionCount,
                   ports.attachmentOptions().size()))
      return invalid("graph-boundary attachment slice is inconsistent");
    for (const FrozenSpatialAttachmentOption &option :
         ports.attachmentOptions().slice(boundary.attachmentOptionOffset,
                                         boundary.attachmentOptionCount)) {
      if (option.ownerKind != FrozenSpatialAttachmentOwnerKind::GraphBoundary ||
          option.owner != boundaryOrdinal || option.localTraversal ||
          option.sharedOperandEnqueueUnit ||
          option.progressBoundary != SpatialDurableProgressBoundaryKind::None ||
          option.endpoint >= routing.routingEndpoints().size())
        return invalid("graph-boundary attachment option is inconsistent");
      const auto &endpoint = routing.routingEndpoints()[option.endpoint];
      if (endpoint.direction != direction ||
          endpoint.dataPath.payloadWidthBits < boundary.payloadWidthBits)
        return invalid("graph-boundary attachment endpoint is incompatible");
    }
  }

  if (ports.endpointAttachmentOffsets().size() !=
          routing.routingEndpoints().size() + 1 ||
      ports.endpointAttachmentOffsets().empty() ||
      ports.endpointAttachmentOffsets().front() != 0 ||
      ports.endpointAttachmentOffsets().back() !=
          ports.endpointAttachmentOptions().size())
    return invalid(
        "endpoint attachment reverse CSR dimensions are inconsistent");
  std::vector<std::uint32_t> optionIncidence(ports.attachmentOptions().size(),
                                             0);
  for (std::size_t endpoint = 0; endpoint < routing.routingEndpoints().size();
       ++endpoint) {
    const PnrIndex begin = ports.endpointAttachmentOffsets()[endpoint];
    const PnrIndex end = ports.endpointAttachmentOffsets()[endpoint + 1];
    if (begin > end || end > ports.endpointAttachmentOptions().size())
      return invalid(
          "endpoint attachment reverse CSR offsets are inconsistent");
    for (PnrIndex cursor = begin; cursor < end; ++cursor) {
      const PnrIndex option = ports.endpointAttachmentOptions()[cursor];
      if (option >= ports.attachmentOptions().size() ||
          ports.attachmentOptions()[option].endpoint != endpoint ||
          ++optionIncidence[option] != 1)
        return invalid("endpoint attachment reverse incidence is invalid");
    }
  }
  if (llvm::any_of(optionIncidence,
                   [](std::uint32_t count) { return count != 1; }))
    return invalid("endpoint attachment reverse incidence is incomplete");
  return llvm::Error::success();
}
