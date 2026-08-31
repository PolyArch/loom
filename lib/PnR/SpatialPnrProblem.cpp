#include "PnR/SpatialPnrProblem.h"
#include "PnR/RoutingNegotiation.h"

#include "PnrDerivedContextInternal.h"
#include "PnrDerivedContextSessionInternal.h"
#include "SpatialActiveProblemStatistics.h"
#include "SpatialActiveRoutingDomain.h"
#include "SpatialBindingRelationModel.h"
#include "SpatialLocalTransferIndex.h"
#include "SpatialMemoryConstraintModel.h"
#include "SpatialPnrCapacityIndex.h"
#include "SpatialPnrHandshakeIndex.h"
#include "SpatialPnrMemoryIndex.h"
#include "SpatialPnrPortIndex.h"
#include "SpatialPnrProblemIdentity.h"
#include "SpatialPnrResourceIndex.h"
#include "SpatialPnrTransferIndex.h"
#include "SpatialProgressIndex.h"
#include "SpatialRecurrenceTimingInternal.h"
#include "SpatialRouteConstraintModel.h"
#include "SpatialTagConstraintModel.h"
#include "StaticSchedulePressure.h"

#include "Fabric/Artifact/FabricTopologyQuality.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;

using SpatialConstraintProjection = ::mapping::SpatialConstraintProjection;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialPnrProblem";
constexpr PnrCapacityContext realizationCountContext{
    frozenArtifact, "realizations", "realizations", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext realizationIndexContext{
    frozenArtifact, "realizations", "realizations", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext actorOffsetContext{
    frozenArtifact, "realizations", "actors", PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext actorCountContext{
    frozenArtifact, "actors", "actors", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext externalIngressOffsetContext{
    frozenArtifact, "memory_realizations", "external_ingresses",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext externalIngressCountContext{
    frozenArtifact, "external_ingresses", "external_ingresses",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext internalConnectionOffsetContext{
    frozenArtifact, "memory_realizations", "internal_connections",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext internalConnectionCountContext{
    frozenArtifact, "internal_connections", "internal_connections",
    PnrCapacityMeasure::Count};
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
constexpr PnrCapacityContext traversalIndexContext{
    frozenArtifact, "traversals", "traversals", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext replicationGroupIndexContext{
    frozenArtifact, "traversal_replication_groups", "replication_groups",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext traversalResourceStateOffsetContext{
    frozenArtifact, "traversals", "traversal_resource_states",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext traversalResourceStateCountContext{
    frozenArtifact, "traversal_resource_states", "traversal_resource_states",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext routeClaimOffsetContext{
    frozenArtifact, "traversals", "traversal_claim_keys",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext routeClaimCountContext{
    frozenArtifact, "traversal_claim_keys", "traversal_claim_keys",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext routeClaimIndexContext{
    frozenArtifact, "route_claims", "route_claims", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext routeClaimCapacityContext{
    frozenArtifact, "route_claims", "capacity_dimensions",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext capacityRouteClaimOffsetContext{
    frozenArtifact, "capacity_route_claims", "capacity_route_claims",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext capacityRouteClaimCountContext{
    frozenArtifact, "capacity_route_claims", "route_claims",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext routeClaimTraversalOffsetContext{
    frozenArtifact, "route_claim_traversals", "route_claim_traversals",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext routeClaimTraversalCountContext{
    frozenArtifact, "route_claim_traversals", "traversals",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext traversalArcOffsetContext{
    frozenArtifact, "traversal_arcs", "traversal_arcs",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext traversalArcCountContext{
    frozenArtifact, "traversal_arcs", "routing_arcs",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext arcIndexContext{
    frozenArtifact, "routing_arcs", "routing_arcs", PnrCapacityMeasure::Index};

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref> std::string refKey(const Ref &reference) {
  return byteKey(canonicalFabricBytes(reference));
}

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
bool domainContains(
    const std::optional<llvm::ArrayRef<SpatialConstraintDomainValue>> &domain,
    const Ref &reference) {
  if (!domain)
    return true;
  return llvm::any_of(*domain, [&](const SpatialConstraintDomainValue &value) {
    const Ref *typed = std::get_if<Ref>(&value);
    return typed && *typed == reference;
  });
}

template <typename Subject>
std::optional<llvm::ArrayRef<SpatialConstraintDomainValue>>
restriction(const FrozenConstraintIndex &constraints,
            ::mapping::SpatialConstraintProjection projection,
            Subject subject) {
  return constraints.shard(projection)
      .restrictedDomain(SpatialConstraintSubject{std::move(subject)});
}

std::string
routeClaimKey(const FabricTraversalRequesterGroupView &requesterGroup,
              PnrIndex capacityDimension) {
  std::vector<std::uint8_t> bytes;
  const auto owner = canonicalFabricBytes(requesterGroup.owner);
  bytes.reserve(20 + owner.size());
  appendU32Be(bytes, static_cast<std::uint32_t>(requesterGroup.kind));
  appendU64Be(bytes, requesterGroup.ordinal);
  appendU64Be(bytes, capacityDimension);
  bytes.insert(bytes.end(), owner.begin(), owner.end());
  return byteKey(bytes);
}

std::string switchReplicationKey(const FabricSwitchTraversalPayload &payload) {
  std::vector<std::uint8_t> bytes;
  const auto owner = canonicalFabricBytes(payload.owner);
  bytes.reserve(8 + owner.size());
  appendU64Be(bytes, payload.input);
  bytes.insert(bytes.end(), owner.begin(), owner.end());
  return byteKey(bytes);
}

const FabricSwitchTraversalPayload *
switchTraversalPayload(const FabricPhysicalTraversalRef &traversal) {
  return std::get_if<FabricSwitchTraversalPayload>(&traversal.payload);
}

llvm::Expected<PnrIndex>
internReplicationGroup(llvm::StringMap<PnrIndex> &groups,
                       const FabricSwitchTraversalPayload &payload) {
  const std::string key = switchReplicationKey(payload);
  auto found = groups.find(key);
  if (found == groups.end()) {
    auto group = checked(replicationGroupIndexContext, groups.size());
    if (!group)
      return group.takeError();
    found = groups.try_emplace(key, *group).first;
  }
  return found->second;
}

bool matchesSwitchOwner(const FabricTraversalRequesterGroupView &requester,
                        const FabricSwitchTraversalPayload &payload) {
  return requester.kind == FabricTraversalRequesterGroupKind::SwitchRequester &&
         requester.owner == FabricInventoryOwnerRef::of(payload.owner);
}

} // namespace

class loom::pnr::FrozenSpatialPnrProblemBuilder final {
public:
  static llvm::Expected<FrozenSpatialPnrProblemHandle>
  build(const dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping, const FabricArtifactView &fabric,
        const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
        const ResolvedPnrConfigView &config,
        const SpatialMappingConstraintSetView &constraintSet,
        const FabricDerivedContextBundle *derivedContexts) {
    if (llvm::Error error = detail::SpatialPnrProblemIdentity::validateInputs(
            dataflow, techMapping, fabric, config, constraintSet))
      return std::move(error);

    std::optional<FabricDerivedContextBundle> ownedContexts;
    if (!derivedContexts) {
      auto built = buildFabricDerivedContextBundle(fabric, physicalTiming);
      if (!built)
        return built.takeError();
      ownedContexts.emplace(std::move(*built));
      derivedContexts = &*ownedContexts;
    } else if (llvm::Error error = revalidateFabricDerivedContextBundle(
                   *derivedContexts, fabric, physicalTiming)) {
      return std::move(error);
    }
    const auto &contextStorage = *derivedContexts->storage_;
    const auto &staticContext = *contextStorage.staticContext;
    const auto &timingContext = *contextStorage.timingContext;
    const auto activeProblemBegin = std::chrono::steady_clock::now();

    auto objectiveProgram = MappingObjectiveProgram::get(
        config.selectedObjectiveCatalogs(), config.policy().objectiveSelection);
    if (!objectiveProgram)
      return objectiveProgram.takeError();

    auto constraints = detail::buildFrozenConstraintIndex(constraintSet);
    if (!constraints)
      return constraints.takeError();
    auto realizations =
        buildRealizations(dataflow, techMapping, fabric, *constraints);
    if (!realizations)
      return realizations.takeError();
    auto schedulePressure = detail::SpatialSchedulePressureIndex::build(
        dataflow, techMapping, *realizations);
    if (!schedulePressure)
      return schedulePressure.takeError();
    auto memory = FrozenSpatialMemoryIndexBuilder::build(dataflow, techMapping,
                                                         fabric, *realizations);
    if (!memory)
      return memory.takeError();
    auto transfers =
        detail::buildFrozenSpatialTransferIndex(dataflow, techMapping);
    if (!transfers)
      return transfers.takeError();
    // No-good literals name logical nets, sinks, traversals, and endpoints, so
    // they can only be resolved once both the transfer and routing domains
    // exist. An unresolvable literal is a freeze failure, not a dropped clause.
    if (llvm::Error error = detail::resolveFrozenConstraintNoGoods(
            *constraints, *transfers, *staticContext.routingTopology))
      return std::move(error);
    const auto &resources = staticContext.resources;
    const auto &routing = timingContext.routing;
    auto localTransfers = detail::buildFrozenSpatialLocalTransferIndex(
        dataflow, techMapping, fabric, *realizations, *transfers, *routing);
    if (!localTransfers)
      return localTransfers.takeError();
    auto ports = detail::buildFrozenSpatialPortIndex(
        dataflow, techMapping, fabric, *realizations, *transfers, *routing);
    if (!ports)
      return ports.takeError();
    auto activeRouting = buildFrozenSpatialActiveRoutingDomain(
        *transfers, *localTransfers, *ports, *routing);
    if (!activeRouting)
      return activeRouting.takeError();
    const auto &progressIndex = timingContext.progressIndex;
    auto bindingRelations = detail::SpatialBindingRelationModel::create(
        dataflow.identity(), *realizations, *constraints, *transfers, *ports,
        *routing);
    if (!bindingRelations)
      return bindingRelations.takeError();
    auto memoryConstraints =
        detail::SpatialMemoryConstraintModel::create(*memory, *constraints);
    if (!memoryConstraints)
      return memoryConstraints.takeError();
    auto tagConstraints = detail::SpatialTagConstraintModel::create(
        dataflow.identity(), *transfers, *constraints);
    if (!tagConstraints)
      return tagConstraints.takeError();
    auto routeConstraints = detail::SpatialRouteConstraintModel::create(
        dataflow.identity(), *constraints, *transfers, *resources, *routing);
    if (!routeConstraints)
      return routeConstraints.takeError();
    auto handshake = detail::buildFrozenSpatialHandshakeIndex(
        dataflow, techMapping, fabric, *staticContext.handshake,
        *realizations, *resources, *routing, *activeRouting);
    if (!handshake)
      return handshake.takeError();
    auto capacity = detail::buildFrozenSpatialCapacityIndex(
        dataflow, techMapping, fabric, *realizations, *memory, *resources,
        *routing, *handshake);
    if (!capacity)
      return capacity.takeError();
    auto recurrenceTiming = detail::SpatialRecurrenceTimingIndex::build(
        dataflow, techMapping, fabric, *realizations, *memory, *transfers,
        *handshake, (*schedulePressure)->analysis());
    if (!recurrenceTiming)
      return recurrenceTiming.takeError();
    auto progressBasis = ::loom::mapping::deriveMappingDataflowProgressBasis(
        dataflow, techMapping.covers());
    if (!progressBasis)
      return progressBasis.takeError();
    if (llvm::Error error = verifyAggregate(
            *realizations, *memory, *transfers, *localTransfers, *ports,
            *resources, *capacity, *routing, *activeRouting, *handshake))
      return std::move(error);

    FrozenSpatialPnrCacheKey cacheKey =
        detail::SpatialPnrProblemIdentity::deriveCacheKey(
            dataflow, techMapping, fabric, config, constraintSet,
            physicalTiming.digest());
    std::vector<DeterministicWorkBudgetEntry> workBudget =
        deriveDeterministicWorkBudgetView(config);
    SpatialActiveProblemStatistics statistics =
        buildSpatialActiveProblemStatistics(
            *realizations, *memory, *transfers, *localTransfers, *ports,
            *capacity, *activeRouting, *handshake,
            detail::elapsedNanoseconds(activeProblemBegin));

    return FrozenSpatialPnrProblemHandle(new FrozenSpatialPnrProblem(
        dataflow.identity(), techMapping.identity(), fabric.identity(),
        constraintSet.identity(), config, std::move(*objectiveProgram),
        std::move(workBudget), std::move(*constraints),
        std::move(*realizations), std::move(*memory), std::move(*transfers),
        std::move(*localTransfers), std::move(*ports), resources,
        std::move(*capacity), routing, std::move(*activeRouting),
        std::move(*handshake), progressIndex,
        std::move(*schedulePressure),
        std::move(*recurrenceTiming), *progressBasis,
        std::move(*bindingRelations), std::move(*memoryConstraints),
        std::move(*tagConstraints), std::move(*routeConstraints), cacheKey,
        std::move(statistics)));
  }

  static llvm::Error
  verifyAggregate(const FrozenSpatialRealizationIndex &realizations,
                  const FrozenSpatialMemoryIndex &memory,
                  const FrozenSpatialTransferIndex &transfers,
                  const FrozenSpatialLocalTransferIndex &localTransfers,
                  const FrozenSpatialPortIndex &ports,
                  const FrozenSpatialResourceIndex &resources,
                  const FrozenSpatialCapacityIndex &capacity,
                  const FrozenSpatialRoutingGraph &routing,
                  const FrozenSpatialActiveRoutingDomain &activeRouting,
                  const FrozenSpatialHandshakeIndex &handshake) {
    const auto rangeFits = [](PnrIndex offset, PnrIndex count,
                              std::size_t size) {
      const std::size_t begin = static_cast<std::size_t>(offset);
      const std::size_t length = static_cast<std::size_t>(count);
      return begin <= size && length <= size - begin;
    };
    const auto rangeContains = [](PnrIndex offset, PnrIndex count,
                                  PnrIndex index) {
      return index >= offset && index - offset < count;
    };

    if (activeRouting.activeEndpoints().size() !=
            routing.routingEndpoints().size() ||
        activeRouting.activeTraversals().size() !=
            routing.traversals().size() ||
        activeRouting.activeArcs().size() != routing.routingArcs().size() ||
        activeRouting.activeTraversalBits().size() !=
            (routing.traversals().size() + 63) / 64)
      return invalid("active routing domain has the wrong shape");
    std::uint64_t activeEndpointCount = 0;
    std::uint64_t activeTraversalCount = 0;
    std::uint64_t activeArcCount = 0;
    for (auto [endpoint, active] :
         llvm::enumerate(activeRouting.activeEndpoints()))
      if (active) {
        ++activeEndpointCount;
        if (endpoint >= routing.routingEndpoints().size())
          return invalid("active routing endpoint is out of range");
      }
    for (auto [traversal, active] :
         llvm::enumerate(activeRouting.activeTraversals())) {
      const bool bit = (activeRouting.activeTraversalBits()[traversal / 64] &
                        (std::uint64_t{1} << (traversal % 64))) != 0;
      if (static_cast<bool>(active) != bit)
        return invalid("active traversal bitset diverges from its domain");
      activeTraversalCount += active != 0;
    }
    for (auto [arc, active] : llvm::enumerate(activeRouting.activeArcs())) {
      if (!active)
        continue;
      ++activeArcCount;
      const EndpointRoutingArc &record = routing.routingArcs()[arc];
      if (record.traversal >= routing.traversals().size() ||
          !activeRouting.traversalIsActive(record.traversal) ||
          !activeRouting.endpointIsActive(routing.arcSources()[arc]) ||
          !activeRouting.endpointIsActive(record.target))
        return invalid("active routing arc has an inactive dependency");
    }
    if (activeEndpointCount != activeRouting.activeEndpointCount() ||
        activeTraversalCount != activeRouting.activeTraversalCount() ||
        activeArcCount != activeRouting.activeArcCount())
      return invalid("active routing domain count is inconsistent");
    for (const FrozenSpatialAttachmentOption &option :
         ports.attachmentOptions())
      if (!activeRouting.endpointIsActive(option.endpoint) ||
          (option.localTraversal &&
           !activeRouting.traversalIsActive(*option.localTraversal)))
        return invalid("attachment option is absent from active routing");

    if (realizations.computeActorRealizations().size() !=
            realizations.computeActors().size() ||
        realizations.memoryActorRealizations().size() !=
            realizations.memoryActors().size())
      return invalid("actor-owner reverse projections are incomplete");
    if (localTransfers.domains().size() != transfers.logicalNets().size())
      return invalid("local-transfer domains are incomplete");
    for (auto [logicalNet, domain] :
         llvm::enumerate(localTransfers.domains())) {
      if (!rangeFits(domain.optionOffset, domain.optionCount,
                     localTransfers.options().size()))
        return invalid("local-transfer option slice is inconsistent");
      for (const FrozenSpatialRegisterFifoTransferOption &option :
           localTransfers.options(static_cast<PnrIndex>(logicalNet))) {
        if (option.logicalNet != logicalNet ||
            option.producerRealization >=
                realizations.computeRealizations().size() ||
            option.consumerRealization >=
                realizations.computeRealizations().size() ||
            option.producerPlacement >=
                realizations.computePlacements().size() ||
            option.consumerPlacement >=
                realizations.computePlacements().size() ||
            option.writeTraversal >= routing.traversals().size() ||
            option.readTraversal >= routing.traversals().size() ||
            option.writeTraversal == option.readTraversal ||
            !activeRouting.traversalIsActive(option.writeTraversal) ||
            !activeRouting.traversalIsActive(option.readTraversal))
          return invalid("local-transfer option is inconsistent");
      }
    }
    if (llvm::Error error =
            FrozenSpatialMemoryIndexBuilder::verify(memory, realizations))
      return error;
    if (capacity.computeInstructionContextOveruse().size() !=
            realizations.computeInstructionContexts().size() ||
        capacity.computeInstructionContextEnvelopeOffsets().size() !=
            realizations.computeInstructionContexts().size() + 1 ||
        capacity.memoryOperationPlanOveruse().size() !=
            handshake.memoryOperationPlans().size() ||
        capacity.memoryOperationPlanEnvelopes().size() !=
            handshake.memoryOperationPlans().size() ||
        capacity.memoryServiceGroupEnvelopeOffsets().size() !=
            memory.serviceUseGroups().size() + 1 ||
        capacity.memoryDispatchOptionOveruse().size() !=
            memory.dispatchOptions().size() ||
        capacity.memoryDispatchOptionPatterns().size() !=
            memory.dispatchOptions().size())
      return invalid("capacity envelope projection is incomplete");

    for (auto [optionOrdinal, option] :
         llvm::enumerate(memory.dispatchOptions())) {
      const PnrIndex pattern =
          capacity.memoryDispatchOptionPatterns()[optionOrdinal];
      if (!option.serviceUsePattern) {
        if (pattern != getInvalidPnrIndex())
          return invalid("manager memory dispatch gained a UsePattern");
        continue;
      }
      if (pattern >= resources.usePatterns().size() ||
          resources.usePatterns()[pattern].reference !=
              *option.serviceUsePattern)
        return invalid("memory dispatch UsePattern projection diverges");
    }

    const auto envelopeOffsets =
        capacity.computeInstructionContextEnvelopeOffsets();
    if (envelopeOffsets.empty() || envelopeOffsets.front() != 0 ||
        envelopeOffsets.back() > capacity.resourceTimeEnvelopes().size())
      return invalid("resource-time envelope offsets are incomplete");
    const auto verifyEnvelope =
        [&](const FrozenSpatialResourceTimeEnvelope &envelope)
        -> llvm::Expected<std::uint64_t> {
      if (envelope.event >= capacity.resourceEvents().size() ||
          !rangeFits(envelope.useOffset, envelope.useCount,
                     capacity.resourceUses().size()) ||
          !rangeFits(envelope.segmentOffset, envelope.segmentCount,
                     capacity.resourceTimeSegments().size()) ||
          envelope.useCount == 0)
        return invalid("resource-time envelope slices are inconsistent");
      for (const FrozenSpatialResourceUse &use :
           capacity.resourceUses().slice(envelope.useOffset, envelope.useCount))
        if (use.event != envelope.event ||
            use.pattern >= resources.usePatterns().size())
          return invalid("resource-time use projection is inconsistent");

      llvm::SmallDenseMap<PnrIndex, std::uint64_t, 8> maximumOveruse;
      for (const FrozenSpatialResourceTimeSegment &segment :
           capacity.resourceTimeSegments().slice(envelope.segmentOffset,
                                                 envelope.segmentCount)) {
        if (segment.capacityDimension >=
                resources.capacityDimensions().size() ||
            segment.beginRank >= segment.endRank)
          return invalid("resource-time segment is invalid");
        const auto &dimension =
            resources.capacityDimensions()[segment.capacityDimension];
        const std::uint64_t expected =
            segment.usageRaw > dimension.capacity
                ? segment.usageRaw - dimension.capacity
                : 0;
        if (segment.overuseRaw != expected)
          return invalid("resource-time segment overuse is inconsistent");
        std::uint64_t &maximum = maximumOveruse[segment.capacityDimension];
        maximum = std::max(maximum, segment.overuseRaw);
      }
      std::uint64_t envelopeOveruse = 0;
      for (const auto &entry : maximumOveruse) {
        if (entry.second >
            std::numeric_limits<std::uint64_t>::max() - envelopeOveruse)
          return invalid("resource-time envelope overuse overflows u64");
        envelopeOveruse += entry.second;
      }
      if (envelopeOveruse != envelope.capacityOveruse)
        return invalid("resource-time envelope overuse is inconsistent");
      return envelopeOveruse;
    };
    for (PnrIndex context = 0;
         context < realizations.computeInstructionContexts().size();
         ++context) {
      if (envelopeOffsets[context] > envelopeOffsets[context + 1])
        return invalid("resource-time envelope offsets are not monotonic");
      std::uint64_t contextOveruse = 0;
      for (const FrozenSpatialResourceTimeEnvelope &envelope :
           capacity.resourceTimeEnvelopes().slice(
               envelopeOffsets[context],
               envelopeOffsets[context + 1] - envelopeOffsets[context])) {
        auto envelopeOveruse = verifyEnvelope(envelope);
        if (!envelopeOveruse)
          return envelopeOveruse.takeError();
        if (*envelopeOveruse >
            std::numeric_limits<std::uint64_t>::max() - contextOveruse)
          return invalid("resource-time context overuse overflows u64");
        contextOveruse += *envelopeOveruse;
      }
      if (contextOveruse !=
          capacity.computeInstructionContextOveruse()[context])
        return invalid("resource-time context overuse is inconsistent");
    }
    PnrIndex expectedMemoryEnvelope = envelopeOffsets.back();
    for (auto [planOrdinal, plan] :
         llvm::enumerate(handshake.memoryOperationPlans())) {
      const PnrIndex envelopeOrdinal =
          capacity.memoryOperationPlanEnvelopes()[planOrdinal];
      if (envelopeOrdinal != expectedMemoryEnvelope++ ||
          envelopeOrdinal >= capacity.resourceTimeEnvelopes().size())
        return invalid("memory plan envelope ordering is inconsistent");
      const FrozenSpatialResourceTimeEnvelope &envelope =
          capacity.resourceTimeEnvelopes()[envelopeOrdinal];
      auto envelopeOveruse = verifyEnvelope(envelope);
      if (!envelopeOveruse)
        return envelopeOveruse.takeError();
      if (envelope.useCount != 1 ||
          capacity.resourceUses()[envelope.useOffset].pattern !=
              plan.usePattern ||
          *envelopeOveruse !=
              capacity.memoryOperationPlanOveruse()[planOrdinal])
        return invalid("memory plan resource-time envelope diverges");
    }
    if (expectedMemoryEnvelope > capacity.resourceTimeEnvelopes().size())
      return invalid("memory operation envelope inventory is incomplete");

    std::vector<std::vector<PnrIndex>> actorPatterns(
        realizations.memoryActors().size());
    for (const FrozenSpatialMemoryDispatchDomain &domain :
         memory.dispatchDomains()) {
      if (domain.actor >= actorPatterns.size() ||
          !rangeFits(domain.optionOffset, domain.optionCount,
                     memory.dispatchOptions().size()))
        return invalid("memory dispatch domain is outside its frozen index");
      auto &patterns = actorPatterns[domain.actor];
      for (PnrIndex option = domain.optionOffset;
           option != domain.optionOffset + domain.optionCount; ++option) {
        const PnrIndex pattern =
            capacity.memoryDispatchOptionPatterns()[option];
        if (pattern != getInvalidPnrIndex())
          patterns.push_back(pattern);
      }
    }
    for (auto &patterns : actorPatterns) {
      llvm::sort(patterns);
      patterns.erase(std::unique(patterns.begin(), patterns.end()),
                     patterns.end());
    }

    std::vector<std::optional<SpatialActorTransitionEventRef>> actorIssueEvents(
        realizations.memoryActors().size());
    for (const FrozenSpatialMemoryOperationHandshakeDomain &domain :
         handshake.memoryOperationDomains()) {
      if (domain.actor >= actorIssueEvents.size() || domain.planCount == 0 ||
          !rangeFits(domain.planOffset, domain.planCount,
                     handshake.memoryOperationPlans().size()))
        return invalid("memory operation domain has no exact actor event");
      const PnrIndex envelopeOrdinal =
          capacity.memoryOperationPlanEnvelopes()[domain.planOffset];
      if (envelopeOrdinal >= capacity.resourceTimeEnvelopes().size())
        return invalid("memory operation domain has no resource-time event");
      const PnrIndex eventOrdinal =
          capacity.resourceTimeEnvelopes()[envelopeOrdinal].event;
      if (eventOrdinal >= capacity.resourceEvents().size())
        return invalid("memory operation domain has a foreign event");
      const auto *event = std::get_if<SpatialActorTransitionEventRef>(
          &capacity.resourceEvents()[eventOrdinal].reference);
      if (!event)
        return invalid("memory operation domain event is not an actor issue");
      if (actorIssueEvents[domain.actor] &&
          !(*actorIssueEvents[domain.actor] == *event))
        return invalid("memory actor has divergent issue events");
      actorIssueEvents[domain.actor] = *event;
    }

    const auto groupOffsets = capacity.memoryServiceGroupEnvelopeOffsets();
    if (groupOffsets.empty() || groupOffsets.front() != 0 ||
        groupOffsets.back() != capacity.memoryServicePatternEnvelopes().size())
      return invalid("memory service envelope offsets are incomplete");
    PnrIndex expectedServiceEnvelope = expectedMemoryEnvelope;
    for (auto [groupOrdinal, group] :
         llvm::enumerate(memory.serviceUseGroups())) {
      if (group.actor >= actorPatterns.size() ||
          group.logicalBinding >= memory.logicalBindings().size() ||
          !actorIssueEvents[group.actor] ||
          groupOffsets[groupOrdinal] > groupOffsets[groupOrdinal + 1])
        return invalid("memory service-use group has no exact event domain");
      const auto choices = capacity.memoryServicePatternEnvelopes().slice(
          groupOffsets[groupOrdinal],
          groupOffsets[groupOrdinal + 1] - groupOffsets[groupOrdinal]);
      if (choices.size() != actorPatterns[group.actor].size())
        return invalid("memory service envelope domain is incomplete");
      for (auto [choiceOrdinal, choice] : llvm::enumerate(choices)) {
        if (choice.pattern != actorPatterns[group.actor][choiceOrdinal] ||
            choice.envelope != expectedServiceEnvelope++ ||
            choice.envelope >= capacity.resourceTimeEnvelopes().size())
          return invalid("memory service envelope ordering is inconsistent");
        const auto &envelope =
            capacity.resourceTimeEnvelopes()[choice.envelope];
        auto envelopeOveruse = verifyEnvelope(envelope);
        if (!envelopeOveruse)
          return envelopeOveruse.takeError();
        const auto &event = capacity.resourceEvents()[envelope.event];
        const auto *issue =
            std::get_if<SpatialActorTransitionEventRef>(&event.reference);
        if (envelope.useCount != 1 ||
            capacity.resourceUses()[envelope.useOffset].pattern !=
                choice.pattern ||
            event.ownerKind !=
                FrozenSpatialResourceEventOwnerKind::LogicalMemoryBinding ||
            event.owner != group.logicalBinding || !issue ||
            !(*issue == *actorIssueEvents[group.actor]))
          return invalid("memory service resource-time envelope diverges");
      }
    }
    if (expectedServiceEnvelope != capacity.resourceTimeEnvelopes().size())
      return invalid("resource-time envelope inventory has unowned records");

    for (const FrozenSpatialResourceEvent &event : capacity.resourceEvents()) {
      switch (event.ownerKind) {
      case FrozenSpatialResourceEventOwnerKind::ComputeRealization:
        if (event.owner >= realizations.computeRealizations().size())
          return invalid("resource-time event has a foreign compute owner");
        break;
      case FrozenSpatialResourceEventOwnerKind::MemoryRealization:
        if (event.owner >= realizations.memoryRealizations().size())
          return invalid("resource-time event has a foreign memory owner");
        break;
      case FrozenSpatialResourceEventOwnerKind::LogicalMemoryBinding:
        if (event.owner >= memory.logicalBindings().size())
          return invalid(
              "resource-time event has a foreign logical-memory owner");
        break;
      }
    }

    for (auto [ordinal, realization] :
         llvm::enumerate(realizations.computeRealizations())) {
      if (!rangeFits(realization.actorOffset, realization.actorCount,
                     realizations.computeActors().size()) ||
          !rangeFits(realization.placementOffset, realization.placementCount,
                     realizations.computePlacements().size()) ||
          realization.placementCount == 0)
        return invalid("compute realization slices are inconsistent");
      for (PnrIndex localActor = 0; localActor < realization.actorCount;
           ++localActor)
        if (realizations.computeActorRealizations()[realization.actorOffset +
                                                    localActor] != ordinal)
          return invalid("compute actor-owner projection is inconsistent");
      for (const FrozenSpatialComputePlacement &placement :
           realizations.computePlacements().slice(realization.placementOffset,
                                                  realization.placementCount)) {
        if (placement.realization != ordinal || placement.contextCount == 0 ||
            !rangeFits(placement.contextOffset, placement.contextCount,
                       realizations.computeInstructionContexts().size()))
          return invalid("compute placement slices are inconsistent");
      }
    }

    for (auto [ordinal, realization] :
         llvm::enumerate(realizations.memoryRealizations())) {
      if (!rangeFits(realization.actorOffset, realization.actorCount,
                     realizations.memoryActors().size()) ||
          !rangeFits(realization.externalIngressOffset,
                     realization.externalIngressCount,
                     realizations.memoryExternalIngresses().size()) ||
          !rangeFits(realization.internalConnectionOffset,
                     realization.internalConnectionCount,
                     realizations.memoryInternalConnections().size()) ||
          !rangeFits(realization.placementOffset, realization.placementCount,
                     realizations.memoryPlacements().size()) ||
          realization.placementCount == 0)
        return invalid("memory realization slices are inconsistent");
      for (PnrIndex localActor = 0; localActor < realization.actorCount;
           ++localActor)
        if (realizations.memoryActorRealizations()[realization.actorOffset +
                                                   localActor] != ordinal)
          return invalid("memory actor-owner projection is inconsistent");
      for (const FrozenSpatialMemoryPlacement &placement :
           realizations.memoryPlacements().slice(realization.placementOffset,
                                                 realization.placementCount))
        if (placement.realization != ordinal ||
            (placement.schedule == ::fabric::Schedule::Temporal) !=
                placement.residentContextCount.has_value() ||
            (placement.residentContextCount &&
             *placement.residentContextCount == 0))
          return invalid("memory placement slices are inconsistent");
    }

    for (const FrozenSpatialLogicalNet &net : transfers.logicalNets()) {
      if (net.sinkCount == 0 || !rangeFits(net.sinkOffset, net.sinkCount,
                                           transfers.logicalNetSinks().size()))
        return invalid("residual logical-net slices are inconsistent");
    }
    if (llvm::Error error = detail::verifyFrozenSpatialPortIndex(
            realizations, transfers, ports, routing))
      return error;

    for (const FrozenSpatialResourceOwner &owner : resources.resourceOwners()) {
      if (!rangeFits(owner.stateOffset, owner.stateCount,
                     resources.resourceStates().size()) ||
          !rangeFits(owner.patternOffset, owner.patternCount,
                     resources.usePatterns().size()) ||
          !rangeFits(owner.timingOffset, owner.timingCount,
                     resources.timingContracts().size()) ||
          !rangeFits(owner.grantOrderOffset, owner.grantOrderCount,
                     resources.grantRequesterOrder().size()))
        return invalid("resource-owner slices are inconsistent");
      if ((owner.grantPolicy == FrozenSpatialGrantPolicyKind::None) !=
              (owner.grantOrderCount == 0) ||
          (owner.grantPolicy == FrozenSpatialGrantPolicyKind::RoundRobin) !=
              owner.roundRobinResetRequester.has_value())
        return invalid("resource-owner grant policy is inconsistent");
      const auto grantOrder = resources.grantRequesterOrder().slice(
          owner.grantOrderOffset, owner.grantOrderCount);
      for (std::uint32_t requester : grantOrder)
        if (requester >= owner.requesterCount)
          return invalid("grant policy requester is out of range");
      if (owner.roundRobinResetRequester &&
          llvm::find(grantOrder, *owner.roundRobinResetRequester) ==
              grantOrder.end())
        return invalid("round-robin reset requester is outside its cycle");

      for (auto [ordinal, state] :
           llvm::enumerate(resources.resourceStates().slice(
               owner.stateOffset, owner.stateCount))) {
        if (state.reference.owner.catalog() != owner.reference ||
            state.reference.ordinal != ordinal ||
            !rangeFits(state.capacityOffset, state.capacityCount,
                       resources.capacityDimensions().size()))
          return invalid("resource-state projection is inconsistent");
      }
      for (const FrozenSpatialTimingContract &timing :
           resources.timingContracts().slice(owner.timingOffset,
                                             owner.timingCount))
        if (timing.eventRankCount != owner.eventCount ||
            !rangeFits(timing.eventRankOffset, timing.eventRankCount,
                       resources.eventRanks().size()))
          return invalid("resource timing projection is inconsistent");
      for (auto [ordinal, pattern] :
           llvm::enumerate(resources.usePatterns().slice(owner.patternOffset,
                                                         owner.patternCount))) {
        if (pattern.reference.owner.catalog() != owner.reference ||
            pattern.reference.ordinal != ordinal ||
            pattern.requester >= owner.requesterCount ||
            pattern.eligibility >= owner.eligibilityCount ||
            pattern.acquireEvent >= owner.eventCount ||
            pattern.releaseEvent >= owner.eventCount ||
            !rangeContains(owner.timingOffset, owner.timingCount,
                           pattern.timingContract) ||
            !rangeFits(pattern.claimOffset, pattern.claimCount,
                       resources.claims().size()) ||
            !rangeFits(pattern.transactionOffset, pattern.transactionCount,
                       resources.internalTransactions().size()))
          return invalid("resource use-pattern projection is inconsistent");
        if (pattern.commit &&
            (pattern.commit->event >= owner.eventCount ||
             pattern.commit->transition >= owner.resourceTransitionCount))
          return invalid("resource commit projection is inconsistent");
        for (const FrozenSpatialResourceClaim &claim : resources.claims().slice(
                 pattern.claimOffset, pattern.claimCount)) {
          if (!rangeContains(owner.stateOffset, owner.stateCount, claim.state))
            return invalid("resource claim names a foreign state");
          const FrozenSpatialResourceState &state =
              resources.resourceStates()[claim.state];
          if (claim.dimension >= state.capacityCount)
            return invalid("resource claim dimension is out of range");
        }
        for (const FrozenSpatialInternalTransaction &transaction :
             resources.internalTransactions().slice(pattern.transactionOffset,
                                                    pattern.transactionCount)) {
          if (!rangeFits(transaction.claimOffset, transaction.claimCount,
                         resources.transactionClaims().size()))
            return invalid("internal transaction slice is inconsistent");
          for (PnrIndex claim : resources.transactionClaims().slice(
                   transaction.claimOffset, transaction.claimCount))
            if (!rangeContains(pattern.claimOffset, pattern.claimCount, claim))
              return invalid("internal transaction names a foreign claim");
        }
      }
    }

    if (routing.adjacencyOffsets().size() !=
            routing.routingEndpoints().size() + 1 ||
        routing.reverseAdjacencyOffsets().size() !=
            routing.routingEndpoints().size() + 1 ||
        routing.reverseArcOrdinals().size() != routing.routingArcs().size() ||
        routing.traversalReplicationGroups().size() !=
            routing.traversals().size() ||
        routing.arcSources().size() != routing.routingArcs().size() ||
        routing.adjacencyOffsets().empty() ||
        routing.adjacencyOffsets().front() != 0 ||
        routing.adjacencyOffsets().back() != routing.routingArcs().size() ||
        routing.reverseAdjacencyOffsets().front() != 0 ||
        routing.reverseAdjacencyOffsets().back() !=
            routing.routingArcs().size())
      return invalid("routing CSR dimensions are inconsistent");

    for (std::size_t source = 0; source < routing.routingEndpoints().size();
         ++source) {
      const PnrIndex begin = routing.adjacencyOffsets()[source];
      const PnrIndex end = routing.adjacencyOffsets()[source + 1];
      if (begin > end || end > routing.routingArcs().size())
        return invalid("routing CSR offsets are inconsistent");
      for (PnrIndex arc = begin; arc < end; ++arc) {
        const EndpointRoutingArc &record = routing.routingArcs()[arc];
        if (routing.arcSources()[arc] != source ||
            record.target >= routing.routingEndpoints().size() ||
            record.traversal >= routing.traversals().size())
          return invalid("routing arc projection is inconsistent");
      }
    }
    std::vector<std::uint8_t> reverseArcSeen(routing.routingArcs().size(), 0);
    for (std::size_t target = 0; target < routing.routingEndpoints().size();
         ++target) {
      const PnrIndex begin = routing.reverseAdjacencyOffsets()[target];
      const PnrIndex end = routing.reverseAdjacencyOffsets()[target + 1];
      if (begin > end || end > routing.reverseArcOrdinals().size())
        return invalid("routing reverse CSR offsets are inconsistent");
      PnrIndex previousSource = 0;
      PnrIndex previousTraversal = 0;
      bool first = true;
      for (PnrIndex cursor = begin; cursor < end; ++cursor) {
        const PnrIndex arc = routing.reverseArcOrdinals()[cursor];
        if (arc >= routing.routingArcs().size() || reverseArcSeen[arc] ||
            routing.routingArcs()[arc].target != target)
          return invalid("routing reverse CSR is not an exact arc partition");
        const PnrIndex source = routing.arcSources()[arc];
        const PnrIndex traversal = routing.routingArcs()[arc].traversal;
        if (!first && std::tie(previousSource, previousTraversal) >
                          std::tie(source, traversal))
          return invalid("routing reverse CSR is not canonical");
        first = false;
        previousSource = source;
        previousTraversal = traversal;
        reverseArcSeen[arc] = 1;
      }
    }
    if (llvm::find(reverseArcSeen, 0) != reverseArcSeen.end())
      return invalid("routing reverse CSR omitted an arc");
    llvm::StringMap<PnrIndex> seenRouteClaims;
    for (auto [ordinal, claim] : llvm::enumerate(routing.routeClaims())) {
      if (claim.capacityDimension >= resources.capacityDimensions().size())
        return invalid("route claim capacity dimension is out of range");
      const FrozenSpatialCapacityDimension &capacity =
          resources.capacityDimensions()[claim.capacityDimension];
      auto qCost = normalizedRouteClaimCost(claim.amount, capacity.capacity);
      if (!qCost)
        return invalid("invalid normalized route claim: " +
                       llvm::toString(qCost.takeError()));
      if (claim.qCost != *qCost)
        return invalid("route claim Q-scaled projection is inconsistent");
      auto index = checked(routeClaimIndexContext, ordinal);
      if (!index)
        return index.takeError();
      if (!seenRouteClaims
               .try_emplace(
                   routeClaimKey(claim.requesterGroup, claim.capacityDimension),
                   *index)
               .second)
        return invalid("route claim key is duplicated");
    }
    if (routing.capacityRouteClaimOffsets().size() !=
            resources.capacityDimensions().size() + 1 ||
        routing.routeClaimTraversalOffsets().size() !=
            routing.routeClaims().size() + 1 ||
        routing.traversalArcOffsets().size() !=
            routing.traversals().size() + 1 ||
        routing.capacityRouteClaimOffsets().back() !=
            routing.capacityRouteClaims().size() ||
        routing.routeClaimTraversalOffsets().back() !=
            routing.routeClaimTraversals().size() ||
        routing.traversalArcOffsets().back() != routing.traversalArcs().size())
      return invalid("routing reverse-incidence CSR shape is inconsistent");

    std::vector<PnrIndex> expectedCapacityOffsets(
        resources.capacityDimensions().size() + 1, 0);
    for (const FrozenSpatialRouteClaim &claim : routing.routeClaims())
      ++expectedCapacityOffsets[claim.capacityDimension + 1];
    for (std::size_t capacity = 1; capacity < expectedCapacityOffsets.size();
         ++capacity)
      expectedCapacityOffsets[capacity] +=
          expectedCapacityOffsets[capacity - 1];
    std::vector<PnrIndex> expectedCapacityClaims(routing.routeClaims().size());
    std::vector<PnrIndex> capacityCursors = expectedCapacityOffsets;
    for (auto [claimOrdinal, claim] : llvm::enumerate(routing.routeClaims()))
      expectedCapacityClaims[capacityCursors[claim.capacityDimension]++] =
          static_cast<PnrIndex>(claimOrdinal);
    if (routing.capacityRouteClaimOffsets() !=
            llvm::ArrayRef<PnrIndex>(expectedCapacityOffsets) ||
        routing.capacityRouteClaims() !=
            llvm::ArrayRef<PnrIndex>(expectedCapacityClaims))
      return invalid(
          "capacity-to-route-claim reverse incidence is inconsistent");

    std::vector<PnrIndex> expectedClaimOffsets(routing.routeClaims().size() + 1,
                                               0);
    for (const FrozenSpatialTraversal &traversal : routing.traversals())
      for (PnrIndex claim : routing.traversalClaimKeys().slice(
               traversal.routeClaimOffset, traversal.routeClaimCount))
        ++expectedClaimOffsets[claim + 1];
    for (std::size_t claim = 1; claim < expectedClaimOffsets.size(); ++claim)
      expectedClaimOffsets[claim] += expectedClaimOffsets[claim - 1];
    std::vector<PnrIndex> expectedClaimTraversals(
        routing.traversalClaimKeys().size());
    std::vector<PnrIndex> claimCursors = expectedClaimOffsets;
    for (auto [traversalOrdinal, traversal] :
         llvm::enumerate(routing.traversals()))
      for (PnrIndex claim : routing.traversalClaimKeys().slice(
               traversal.routeClaimOffset, traversal.routeClaimCount))
        expectedClaimTraversals[claimCursors[claim]++] =
            static_cast<PnrIndex>(traversalOrdinal);
    if (routing.routeClaimTraversalOffsets() !=
            llvm::ArrayRef<PnrIndex>(expectedClaimOffsets) ||
        routing.routeClaimTraversals() !=
            llvm::ArrayRef<PnrIndex>(expectedClaimTraversals))
      return invalid(
          "route-claim-to-traversal reverse incidence is inconsistent");

    std::vector<PnrIndex> expectedTraversalOffsets(
        routing.traversals().size() + 1, 0);
    for (const EndpointRoutingArc &arc : routing.routingArcs())
      ++expectedTraversalOffsets[arc.traversal + 1];
    for (std::size_t traversal = 1; traversal < expectedTraversalOffsets.size();
         ++traversal)
      expectedTraversalOffsets[traversal] +=
          expectedTraversalOffsets[traversal - 1];
    std::vector<PnrIndex> expectedTraversalArcs(routing.routingArcs().size());
    std::vector<PnrIndex> traversalCursors = expectedTraversalOffsets;
    for (auto [arcOrdinal, arc] : llvm::enumerate(routing.routingArcs()))
      expectedTraversalArcs[traversalCursors[arc.traversal]++] =
          static_cast<PnrIndex>(arcOrdinal);
    if (routing.traversalArcOffsets() !=
            llvm::ArrayRef<PnrIndex>(expectedTraversalOffsets) ||
        routing.traversalArcs() !=
            llvm::ArrayRef<PnrIndex>(expectedTraversalArcs))
      return invalid("traversal-to-arc reverse incidence is inconsistent");
    llvm::StringMap<PnrIndex> expectedReplicationGroups;
    for (auto [traversalOrdinal, traversal] :
         llvm::enumerate(routing.traversals())) {
      const bool endpointlessRegisterFifo =
          traversal.reference.kind() ==
          FabricPhysicalTraversalKind::PeRegisterFifoTraversal;
      if ((!endpointlessRegisterFifo &&
           (traversal.sourceCount == 0 || traversal.destinationCount == 0)) ||
          (endpointlessRegisterFifo &&
           (traversal.sourceCount != 0 || traversal.destinationCount != 0 ||
            traversal.resourceStateCount == 0)) ||
          !rangeFits(traversal.sourceOffset, traversal.sourceCount,
                     routing.traversalEndpoints().size()) ||
          !rangeFits(traversal.destinationOffset, traversal.destinationCount,
                     routing.traversalEndpoints().size()) ||
          !rangeFits(traversal.resourceStateOffset,
                     traversal.resourceStateCount,
                     routing.traversalResourceStates().size()) ||
          !rangeFits(traversal.routeClaimOffset, traversal.routeClaimCount,
                     routing.traversalClaimKeys().size()))
        return invalid("traversal endpoint slices are inconsistent");
      for (PnrIndex endpoint : routing.traversalEndpoints().slice(
               traversal.sourceOffset, traversal.sourceCount))
        if (endpoint >= routing.routingEndpoints().size())
          return invalid("traversal source endpoint is out of range");
      for (PnrIndex endpoint : routing.traversalEndpoints().slice(
               traversal.destinationOffset, traversal.destinationCount))
        if (endpoint >= routing.routingEndpoints().size())
          return invalid("traversal destination endpoint is out of range");
      for (PnrIndex state : routing.traversalResourceStates().slice(
               traversal.resourceStateOffset, traversal.resourceStateCount))
        if (state >= resources.resourceStates().size())
          return invalid("traversal resource state is out of range");
      PnrIndex previousClaim = 0;
      bool firstClaim = true;
      for (PnrIndex claim : routing.traversalClaimKeys().slice(
               traversal.routeClaimOffset, traversal.routeClaimCount)) {
        if (claim >= routing.routeClaims().size())
          return invalid("traversal route claim is out of range");
        if (!firstClaim && claim <= previousClaim)
          return invalid("traversal route claims are not canonical");
        firstClaim = false;
        previousClaim = claim;
      }
      const auto *switchPayload = switchTraversalPayload(traversal.reference);
      PnrIndex expectedReplicationGroup = getInvalidPnrIndex();
      if (switchPayload) {
        auto group =
            internReplicationGroup(expectedReplicationGroups, *switchPayload);
        if (!group)
          return group.takeError();
        expectedReplicationGroup = *group;
      }
      for (PnrIndex claimOrdinal : routing.traversalClaimKeys().slice(
               traversal.routeClaimOffset, traversal.routeClaimCount)) {
        const FabricTraversalRequesterGroupView &requester =
            routing.routeClaims()[claimOrdinal].requesterGroup;
        if (requester.kind !=
            FabricTraversalRequesterGroupKind::SwitchRequester)
          continue;
        if (!switchPayload || !matchesSwitchOwner(requester, *switchPayload))
          return invalid("switch requester group disagrees with its traversal");
      }
      if (routing.traversalReplicationGroups()[traversalOrdinal] !=
          expectedReplicationGroup)
        return invalid(
            "traversal replication group is not the Fabric projection");
    }
    if (llvm::Error error = detail::verifyFrozenSpatialHandshakeIndex(
            handshake, realizations, resources, routing))
      return error;
    return llvm::Error::success();
  }

  static llvm::Expected<FabricDerivedContextBundle> buildDerivedContexts(
      const FabricArtifactView &fabric,
      const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
      DerivedContextCacheAccess *staticAccess,
      DerivedContextCacheAccess *timingAccess) {
    if (fabric.rootKind() != FabricRootKind::Module)
      return invalid("FabricStaticContext requires one Module root");

    if (staticAccess)
      *staticAccess = {};
    if (timingAccess)
      *timingAccess = {};
    auto session = detail::currentPnrDerivedContextSession();
    const auto staticKey = detail::deriveFabricStaticContextKey(fabric);
    const auto staticBegin = std::chrono::steady_clock::now();
    bool staticReserved = false;
    llvm::scope_exit abandonStatic([&] {
      if (session && staticReserved)
        session->abandon(detail::PnrDerivedContextDomain::FabricStatic,
                         staticKey, detail::elapsedNanoseconds(staticBegin));
    });
    std::shared_ptr<const detail::FabricStaticContext> staticContext;
    DerivedContextConstructionStatistics staticStatistics;
    if (session) {
      auto lookup = session->lookupOrReserve(
          detail::PnrDerivedContextDomain::FabricStatic, staticKey);
      if (!lookup)
        return lookup.takeError();
      if (lookup->entry) {
        staticContext =
            detail::contextFromEntry<detail::FabricStaticContext>(lookup->entry);
        staticStatistics = {1, lookup->entry->constructionNanoseconds,
                            lookup->entry->retainedBytes,
                            lookup->entry->deterministicWork};
        if (staticAccess)
          staticAccess->hits = 1;
      } else {
        staticReserved = lookup->reservedConstruction;
        if (staticAccess)
          staticAccess->misses = 1;
      }
    } else if (staticAccess) {
      staticAccess->misses = 1;
    }

    if (!staticContext) {
      auto resources = detail::buildFrozenSpatialResourceIndex(fabric);
      if (!resources)
        return resources.takeError();
      auto topology = freezeEndpointRoutingTopology(fabric);
      if (!topology)
        return topology.takeError();
      auto tagContinuity = freezeSpatialTagContinuityIndex(fabric);
      if (!tagContinuity)
        return tagContinuity.takeError();
      auto handshake = acquireFabricHandshakeContext(fabric);
      if (!handshake)
        return handshake.takeError();
      std::optional<FabricTopologyQualityReport> topologyQuality;
      if (mapping_debug::enabled(mapping_debug::Level::Summary)) {
        auto analyzed = analyzeFabricTopologyQuality(fabric);
        if (!analyzed)
          return analyzed.takeError();
        topologyQuality.emplace(std::move(*analyzed));
      }

      auto resourcesOwner = std::make_shared<const FrozenSpatialResourceIndex>(
          std::move(*resources));
      auto topologyOwner =
          std::make_shared<const FrozenEndpointRoutingTopology>(
              std::move(*topology));
      auto tagOwner = std::make_shared<const FrozenSpatialTagContinuityIndex>(
          std::move(*tagContinuity));
      staticContext = std::make_shared<const detail::FabricStaticContext>(
          detail::FabricStaticContext{staticKey, fabric.identity(),
                                      resourcesOwner, topologyOwner, tagOwner,
                                      std::move(*handshake),
                                      std::move(topologyQuality)});
      staticStatistics.constructionCount = 1;
      staticStatistics.constructionNanoseconds =
          detail::elapsedNanoseconds(staticBegin);
      staticStatistics.retainedBytes = detail::staticContextRetainedBytes(
          *resourcesOwner, *topologyOwner, *tagOwner,
          *staticContext->handshake,
          staticContext->topologyQuality);
      const FabricHandshakeContextStatistics &handshakeStatistics =
          staticContext->handshake->statistics();
      staticStatistics.deterministicWork =
          resourcesOwner->resourceOwners().size() +
          topologyOwner->endpoints().size() +
          topologyOwner->traversals().size() + topologyOwner->arcs().size() +
          handshakeStatistics.deterministicWork;
      if (staticContext->topologyQuality)
        staticStatistics.deterministicWork +=
            detail::topologyQualityDeterministicWork(
                *staticContext->topologyQuality);
      if (session) {
        auto entry = session->complete(
            detail::PnrDerivedContextDomain::FabricStatic, staticKey,
            staticContext, staticStatistics.constructionNanoseconds,
            staticStatistics.retainedBytes,
            staticStatistics.deterministicWork);
        staticContext =
            detail::contextFromEntry<detail::FabricStaticContext>(entry);
        staticReserved = false;
      }
    }

    FabricDerivedContextStatistics statistics;
    statistics.staticContext = staticStatistics;
    const auto &resourcesOwner = staticContext->resources;
    const auto &topologyOwner = staticContext->routingTopology;
    statistics.resourceOwnerCount = resourcesOwner->resourceOwners().size();
    statistics.endpointCount = topologyOwner->endpoints().size();
    statistics.traversalCount = topologyOwner->traversals().size();
    statistics.routingArcCount = topologyOwner->arcs().size();
    const FabricHandshakeContextStatistics &handshakeStatistics =
        staticContext->handshake->statistics();
    statistics.handshakeOwnerCount = handshakeStatistics.ownerCount;
    statistics.handshakeStructuralTemplateCount =
        handshakeStatistics.structuralTemplateCount;
    statistics.handshakeBindingInstanceCount =
        handshakeStatistics.bindingInstanceCount;
    statistics.handshakeStructuralNodeCount =
        handshakeStatistics.structuralNodeCount;
    statistics.handshakeStructuralArcCount =
        handshakeStatistics.structuralArcCount;
    statistics.handshakeStructuralFragmentCount =
        handshakeStatistics.structuralFragmentCount;
    statistics.handshakeUnconditionalArcCount =
        handshakeStatistics.unconditionalArcCount;
    statistics.handshakeNodeCount = handshakeStatistics.nodeCount;
    statistics.handshakeArcCount = handshakeStatistics.arcCount;
    statistics.handshakeFragmentCount = handshakeStatistics.fragmentCount;
    const auto timingKey =
        detail::deriveFabricTimingContextKey(staticContext->key, physicalTiming);
    const auto timingBegin = std::chrono::steady_clock::now();
    bool timingReserved = false;
    llvm::scope_exit abandonTiming([&] {
      if (session && timingReserved)
        session->abandon(detail::PnrDerivedContextDomain::FabricTiming,
                         timingKey, detail::elapsedNanoseconds(timingBegin));
    });
    std::shared_ptr<const detail::FabricTimingContext> timingContext;
    DerivedContextConstructionStatistics timingStatistics;
    if (session) {
      auto lookup = session->lookupOrReserve(
          detail::PnrDerivedContextDomain::FabricTiming, timingKey);
      if (!lookup)
        return lookup.takeError();
      if (lookup->entry) {
        timingContext =
            detail::contextFromEntry<detail::FabricTimingContext>(lookup->entry);
        timingStatistics = {1, lookup->entry->constructionNanoseconds,
                            lookup->entry->retainedBytes,
                            lookup->entry->deterministicWork};
        if (timingAccess)
          timingAccess->hits = 1;
      } else {
        timingReserved = lookup->reservedConstruction;
        if (timingAccess)
          timingAccess->misses = 1;
      }
    } else if (timingAccess) {
      timingAccess->misses = 1;
    }
    if (!timingContext) {
      if (llvm::Error error =
              ::loom::fabric::validateFabricPhysicalTimingProfile(
                  fabric, physicalTiming))
        return std::move(error);
      auto routing = buildRouting(fabric, physicalTiming, *resourcesOwner,
                                  topologyOwner,
                                  staticContext->tagContinuity);
      if (!routing)
        return routing.takeError();
      auto routingOwner =
          std::make_shared<const FrozenSpatialRoutingGraph>(std::move(*routing));
      auto progressIndex =
          detail::buildFrozenSpatialProgressIndex(*routingOwner, fabric);
      if (!progressIndex)
        return progressIndex.takeError();
      timingContext = std::make_shared<const detail::FabricTimingContext>(
          detail::FabricTimingContext{timingKey, fabric.identity(),
                                      physicalTiming.digest().bytes(),
                                      staticContext, routingOwner,
                                      std::move(*progressIndex)});
      timingStatistics.constructionCount = 1;
      timingStatistics.constructionNanoseconds =
          detail::elapsedNanoseconds(timingBegin);
      timingStatistics.retainedBytes =
          detail::timingContextRetainedBytes(
              *routingOwner, *timingContext->progressIndex);
      timingStatistics.deterministicWork =
          routingOwner->traversals().size() +
          routingOwner->routeClaims().size() +
          routingOwner->traversalClaimKeys().size() +
          routingOwner->traversalArcs().size() +
          timingContext->progressIndex->traversalOwnerOrdinals().size() +
          timingContext->progressIndex->ownerTraversals().size();
      if (session) {
        auto entry = session->complete(
            detail::PnrDerivedContextDomain::FabricTiming, timingKey,
            timingContext, timingStatistics.constructionNanoseconds,
            timingStatistics.retainedBytes,
            timingStatistics.deterministicWork);
        timingContext =
            detail::contextFromEntry<detail::FabricTimingContext>(entry);
        timingReserved = false;
      }
    }
    statistics.timingContext = timingStatistics;

    auto storage = std::make_shared<const detail::FabricDerivedContextStorage>(
        detail::FabricDerivedContextStorage{staticContext, timingContext,
                                            statistics});
    FabricDerivedContextBundle result(std::move(storage));
    if ((staticAccess && staticAccess->hits != 0) ||
        (timingAccess && timingAccess->hits != 0)) {
      if (llvm::Error error =
              revalidateDerivedContexts(result, fabric, physicalTiming))
        return std::move(error);
      if (session)
        session->recordRevalidation();
    }
    return result;
  }

  static llvm::Error revalidateDerivedContexts(
      const FabricDerivedContextBundle &bundle,
      const FabricArtifactView &fabric,
      const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming) {
    if (!bundle.storage_ || !bundle.storage_->staticContext ||
        !bundle.storage_->timingContext)
      return invalid("Fabric derived context bundle is incomplete");
    const auto &staticContext = *bundle.storage_->staticContext;
    const auto &timingContext = *bundle.storage_->timingContext;
    if (physicalTiming.fabricIdentity() != fabric.identity() ||
        staticContext.fabricIdentity != fabric.identity() ||
        timingContext.fabricIdentity != fabric.identity() ||
        timingContext.staticContext.get() != &staticContext)
      return invalid("Fabric derived context binds another Fabric identity");
    if (staticContext.key != detail::deriveFabricStaticContextKey(fabric) ||
        timingContext.key != detail::deriveFabricTimingContextKey(
                                 staticContext.key, physicalTiming) ||
        timingContext.physicalTimingDigestBytes !=
            physicalTiming.digest().bytes())
      return invalid("Fabric derived context key does not match its inputs");
    if (!staticContext.resources || !staticContext.routingTopology ||
        !staticContext.tagContinuity || !timingContext.routing ||
        !timingContext.progressIndex ||
        &timingContext.routing->topology() !=
            staticContext.routingTopology.get() ||
        &timingContext.routing->tagContinuity() !=
            staticContext.tagContinuity.get())
      return invalid("Fabric derived context lost a static projection");
    if (llvm::Error error =
            revalidateFabricHandshakeContext(*staticContext.handshake, fabric))
      return error;
    return llvm::Error::success();
  }

private:
  static llvm::Expected<FrozenSpatialRealizationIndex>
  buildRealizations(const dataflow::CanonicalDataflowProgramView &dataflow,
                    const TechMappingView &techMapping,
                    const FabricArtifactView &fabric,
                    const FrozenConstraintIndex &constraints) {
    (void)dataflow;
    if (llvm::Error error = preflightPnrIndexCapacity(
            realizationCountContext, techMapping.computeRealizations().size()))
      return std::move(error);
    if (llvm::Error error = preflightPnrIndexCapacity(
            realizationCountContext, techMapping.memoryRealizations().size()))
      return std::move(error);

    FrozenSpatialRealizationIndex result;
    auto computeContextDemands =
        ::loom::mapping::deriveSpatialComputeContextDemands(techMapping,
                                                            fabric);
    if (!computeContextDemands)
      return computeContextDemands.takeError();
    if (computeContextDemands->size() !=
        techMapping.computeRealizations().size())
      return invalid("compute context demand projection is incomplete");
    result.computeRealizations_.reserve(
        techMapping.computeRealizations().size());
    for (auto [realizationOrdinal, realization] :
         llvm::enumerate(techMapping.computeRealizations())) {
      auto realizationIndex =
          checked(realizationIndexContext, realizationOrdinal);
      if (!realizationIndex)
        return realizationIndex.takeError();
      auto actorOffset =
          checked(actorOffsetContext, result.computeActors_.size());
      if (!actorOffset)
        return actorOffset.takeError();
      if (llvm::Error error =
              preflightAppend(actorCountContext, result.computeActors_.size(),
                              realization.actors.size()))
        return error;
      for (const TechComputeActorView &actor : realization.actors) {
        result.computeActors_.push_back(actor.actor);
        result.computeActorRealizations_.push_back(*realizationIndex);
      }
      auto actorCount = checked(actorCountContext, realization.actors.size());
      if (!actorCount)
        return actorCount.takeError();
      auto placementOffset =
          checked(placementOffsetContext, result.computePlacements_.size());
      if (!placementOffset)
        return placementOffset.takeError();

      const TechComputeRealizationRef subject{realization.entityId};
      const auto fuDomain = restriction(
          constraints, SpatialConstraintProjection::ComputePlacement, subject);
      const auto peDomain = restriction(
          constraints, SpatialConstraintProjection::ComputeParentPe, subject);
      const auto contextDomain = restriction(
          constraints, SpatialConstraintProjection::ComputeInstructionContext,
          subject);
      const auto correlatedDomain = restriction(
          constraints, SpatialConstraintProjection::ComputeFuContext, subject);

      const auto &contextDemand = (*computeContextDemands)[realizationOrdinal];
      if (contextDemand.realization != realizationOrdinal ||
          contextDemand.capabilityTemplate != realization.capabilityTemplate)
        return invalid("compute context demand projection changed owner");
      for (const auto &placementSupply : contextDemand.placements) {
        const FabricFuOccurrenceRef fu = placementSupply.fu;
        if (!domainContains(fuDomain, fu))
          continue;
        const FabricPeOccurrenceRef parent = placementSupply.parentPe;
        if (!domainContains(peDomain, parent))
          continue;
        auto contextOffset = checked(contextOffsetContext,
                                     result.computeInstructionContexts_.size());
        if (!contextOffset)
          return contextOffset.takeError();
        if (llvm::Error error = preflightAppend(
                contextCountContext, result.computeInstructionContexts_.size(),
                placementSupply.contexts.size()))
          return error;
        for (const InstructionContextRef &context : placementSupply.contexts) {
          if (!domainContains(contextDomain, context) ||
              !domainContains(correlatedDomain,
                              SpatialConstraintFuContext{fu, context}))
            continue;
          result.computeInstructionContexts_.push_back(context);
        }
        const std::size_t acceptedContextCount =
            result.computeInstructionContexts_.size() - *contextOffset;
        if (acceptedContextCount == 0)
          continue;
        auto frozenContextCount =
            checked(contextCountContext, acceptedContextCount);
        if (!frozenContextCount)
          return frozenContextCount.takeError();
        if (llvm::Error error = preflightAppend(
                placementCountContext, result.computePlacements_.size(), 1))
          return error;
        result.computePlacements_.push_back(
            {*realizationIndex, fu, parent, placementSupply.schedule,
             *contextOffset, *frozenContextCount});
      }
      const std::size_t placementCountValue =
          result.computePlacements_.size() - *placementOffset;
      if (placementCountValue == 0)
        return infeasible("a compute realization has no legal FU/context pair");
      auto placementCount = checked(placementCountContext, placementCountValue);
      if (!placementCount)
        return placementCount.takeError();
      result.computeRealizations_.push_back(
          {subject, realization.capabilityTemplate, *actorOffset, *actorCount,
           *placementOffset, *placementCount});
    }

    result.memoryRealizations_.reserve(techMapping.memoryRealizations().size());
    for (auto [realizationOrdinal, realization] :
         llvm::enumerate(techMapping.memoryRealizations())) {
      auto realizationIndex =
          checked(realizationIndexContext, realizationOrdinal);
      if (!realizationIndex)
        return realizationIndex.takeError();
      auto actorOffset =
          checked(actorOffsetContext, result.memoryActors_.size());
      if (!actorOffset)
        return actorOffset.takeError();
      if (llvm::Error error =
              preflightAppend(actorCountContext, result.memoryActors_.size(),
                              realization.actors.size()))
        return error;
      for (const TechMemoryActorView &actor : realization.actors) {
        result.memoryActors_.push_back(
            {actor.actor, actor.operationPort, actor.capability});
        result.memoryActorRealizations_.push_back(*realizationIndex);
      }
      auto actorCount = checked(actorCountContext, realization.actors.size());
      if (!actorCount)
        return actorCount.takeError();
      const auto *engine = fabric.memoryEngineTemplate(realization.engine);
      if (!engine)
        return invalid("a memory realization names a foreign engine template");
      auto externalIngressOffset = checked(
          externalIngressOffsetContext, result.memoryExternalIngresses_.size());
      if (!externalIngressOffset)
        return externalIngressOffset.takeError();
      if (engine->schedule == ::fabric::Schedule::Temporal) {
        auto ingresses =
            deriveTechMemoryExternalIngresses(realization, dataflow);
        if (!ingresses)
          return ingresses.takeError();
        if (llvm::Error error = preflightAppend(
                externalIngressCountContext,
                result.memoryExternalIngresses_.size(), ingresses->size()))
          return error;
        result.memoryExternalIngresses_.insert(
            result.memoryExternalIngresses_.end(),
            std::make_move_iterator(ingresses->begin()),
            std::make_move_iterator(ingresses->end()));
      }
      auto externalIngressCount = checked(
          externalIngressCountContext,
          result.memoryExternalIngresses_.size() - *externalIngressOffset);
      if (!externalIngressCount)
        return externalIngressCount.takeError();
      auto internalConnectionOffset =
          checked(internalConnectionOffsetContext,
                  result.memoryInternalConnections_.size());
      if (!internalConnectionOffset)
        return internalConnectionOffset.takeError();
      std::map<std::vector<std::uint8_t>,
               FabricMemoryEngineTemplateInternalConnectionRef>
          selectedInternalConnections;
      for (const TechMemoryInternalEdgeView &edge : realization.internalEdges)
        selectedInternalConnections.try_emplace(
            canonicalFabricBytes(edge.connection), edge.connection);
      if (llvm::Error error =
              preflightAppend(internalConnectionCountContext,
                              result.memoryInternalConnections_.size(),
                              selectedInternalConnections.size()))
        return error;
      for (const auto &[key, connection] : selectedInternalConnections) {
        (void)key;
        result.memoryInternalConnections_.push_back(connection);
      }
      auto internalConnectionCount = checked(
          internalConnectionCountContext,
          result.memoryInternalConnections_.size() - *internalConnectionOffset);
      if (!internalConnectionCount)
        return internalConnectionCount.takeError();
      auto placementOffset =
          checked(placementOffsetContext, result.memoryPlacements_.size());
      if (!placementOffset)
        return placementOffset.takeError();

      const TechMemoryRealizationRef subject{realization.entityId};
      const auto placementDomain = restriction(
          constraints, SpatialConstraintProjection::MemoryPlacement, subject);
      for (FabricMemoryOccurrenceRef memory : fabric.memoryOccurrences()) {
        const std::optional<FabricMemoryEngineTemplateRef> definition =
            fabric.memoryEngineTemplateOf(memory);
        if (!definition || *definition != realization.engine ||
            !domainContains(placementDomain, memory))
          continue;
        const std::optional<::fabric::Schedule> schedule =
            fabric.memorySchedule(memory);
        if (!schedule)
          return invalid(
              "a Fabric memory occurrence has no scheduling contract");
        if (*schedule != engine->schedule)
          return invalid(
              "a memory occurrence disagrees with its engine schedule");

        bool actorsAdmitted = true;
        for (const TechMemoryActorView &actor : realization.actors) {
          const FabricMemoryOperationPortRef operationPort{
              memory, actor.operationPort.ordinal};
          if (!fabric.memoryOperationPort(operationPort)) {
            actorsAdmitted = false;
            break;
          }
          const auto operationDomain = restriction(
              constraints, SpatialConstraintProjection::MemoryOperationPort,
              actor.actor);
          if (!domainContains(operationDomain, operationPort)) {
            actorsAdmitted = false;
            break;
          }
        }
        if (!actorsAdmitted)
          continue;
        if (llvm::Error error = preflightAppend(
                placementCountContext, result.memoryPlacements_.size(), 1))
          return error;
        result.memoryPlacements_.push_back({*realizationIndex, memory,
                                            *schedule,
                                            engine->residentContextCount});
      }
      const std::size_t placementCountValue =
          result.memoryPlacements_.size() - *placementOffset;
      if (placementCountValue == 0)
        return infeasible(
            "a memory realization has no legal memory occurrence");
      auto placementCount = checked(placementCountContext, placementCountValue);
      if (!placementCount)
        return placementCount.takeError();
      result.memoryRealizations_.push_back(
          {subject, realization.engine, *actorOffset, *actorCount,
           *externalIngressOffset, *externalIngressCount,
           *internalConnectionOffset, *internalConnectionCount,
           *placementOffset, *placementCount});
    }
    return result;
  }

  static llvm::Expected<FrozenSpatialRoutingGraph> buildRouting(
      const FabricArtifactView &fabric,
      const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
      const FrozenSpatialResourceIndex &resources,
      std::shared_ptr<const FrozenEndpointRoutingTopology> topology,
      std::shared_ptr<const FrozenSpatialTagContinuityIndex> tagContinuity) {
    if (llvm::Error error = ::loom::fabric::validateFabricPhysicalTimingProfile(
            fabric, physicalTiming))
      return std::move(error);
    const auto traversalViews = fabric.physicalTraversals();
    FrozenSpatialRoutingGraph result;
    result.requiredCombinationalDelayQuanta_ =
        physicalTiming.requiredCombinationalDelayQuanta();
    result.physicalTimingProfileDigestBytes_ = physicalTiming.digest().bytes();
    result.physicalTimingProfileKind_ = physicalTiming.kind();
    result.physicalTimingProviderIdentity_ =
        physicalTiming.providerIdentity().str();
    result.physicalTimingTechnologyIdentity_ =
        physicalTiming.technologyIdentity().str();
    result.physicalTimingCharacterizationIdentity_ =
        physicalTiming.characterizationIdentity().str();
    if (!topology || !tagContinuity)
      return invalid("Fabric timing context has a null static projection");
    result.topology_ = std::move(topology);
    if (result.topology_->traversals().size() != traversalViews.size())
      return invalid("routing topology lost a physical traversal");
    result.tagContinuity_ = std::move(tagContinuity);
    llvm::StringMap<PnrIndex> stateByCanonicalRef;
    for (auto [ordinal, state] : llvm::enumerate(resources.resourceStates())) {
      auto index = checked(traversalResourceStateCountContext, ordinal);
      if (!index)
        return index.takeError();
      if (!stateByCanonicalRef.try_emplace(refKey(state.reference), *index)
               .second)
        return invalid("the frozen resource-state inventory has a duplicate");
    }
    llvm::StringMap<PnrIndex> patternByCanonicalRef;
    for (auto [ordinal, pattern] : llvm::enumerate(resources.usePatterns())) {
      auto index = checked(routeClaimIndexContext, ordinal);
      if (!index)
        return index.takeError();
      if (!patternByCanonicalRef.try_emplace(refKey(pattern.reference), *index)
               .second)
        return invalid("the frozen use-pattern inventory has a duplicate");
    }
    llvm::StringMap<PnrIndex> routeClaimByKey;
    llvm::StringMap<const ::loom::fabric::FabricTraversalPhysicalTiming *>
        physicalTimingByTraversal;
    for (const auto &timing : physicalTiming.traversals())
      if (!physicalTimingByTraversal
               .try_emplace(refKey(timing.traversal), &timing)
               .second)
        return invalid("the physical timing profile repeats a traversal");
    result.traversals_.reserve(traversalViews.size());
    for (auto [traversalOrdinal, traversal] : llvm::enumerate(traversalViews)) {
      const EndpointRoutingTraversal &routingTraversal =
          result.topology_->traversals()[traversalOrdinal];
      if (routingTraversal.reference != traversal.reference)
        return invalid("routing topology traversal order is not canonical");
      const auto physicalTimingFound =
          physicalTimingByTraversal.find(refKey(traversal.reference));
      if (physicalTimingFound == physicalTimingByTraversal.end())
        return invalid("routing traversal has no physical timing record");
      auto resourceStateOffset =
          checked(traversalResourceStateOffsetContext,
                  result.traversalResourceStates_.size());
      if (!resourceStateOffset)
        return resourceStateOffset.takeError();
      if (llvm::Error error =
              preflightAppend(traversalResourceStateCountContext,
                              result.traversalResourceStates_.size(),
                              traversal.resourceStates.size()))
        return error;
      for (const FabricResourceStateRef &state : traversal.resourceStates) {
        const auto found = stateByCanonicalRef.find(refKey(state));
        if (found == stateByCanonicalRef.end())
          return invalid(
              "a traversal state is absent from the resource inventory");
        result.traversalResourceStates_.push_back(found->second);
      }
      auto resourceStateCount = checked(traversalResourceStateCountContext,
                                        traversal.resourceStates.size());
      if (!resourceStateCount)
        return resourceStateCount.takeError();
      auto routeClaimOffset =
          checked(routeClaimOffsetContext, result.traversalClaimKeys_.size());
      if (!routeClaimOffset)
        return routeClaimOffset.takeError();
      std::vector<PnrIndex> traversalClaimKeys;
      for (const FabricTraversalUseView &use : traversal.impliedUses) {
        if (use.occupancyKind ==
            FabricTraversalUseOccupancyKind::RuntimeService)
          continue;
        const auto patternFound =
            patternByCanonicalRef.find(refKey(use.pattern));
        if (patternFound == patternByCanonicalRef.end())
          return invalid(
              "a traversal use is absent from the resource inventory");
        const FrozenSpatialUsePattern &pattern =
            resources.usePatterns()[patternFound->second];
        if (use.requesterGroup.owner != pattern.reference.owner.catalog())
          return invalid("a traversal requester names a foreign owner");
        switch (use.requesterGroup.kind) {
        case FabricTraversalRequesterGroupKind::UsePattern:
          if (use.requesterGroup.ordinal != pattern.reference.ordinal)
            return invalid(
                "a traversal requester disagrees with its use pattern");
          break;
        case FabricTraversalRequesterGroupKind::SwitchRequester:
          if (use.requesterGroup.ordinal != pattern.requester)
            return invalid(
                "a switch traversal requester disagrees with its pattern");
          break;
        }

        for (const FrozenSpatialResourceClaim &claim : resources.claims().slice(
                 pattern.claimOffset, pattern.claimCount)) {
          if (claim.state >= resources.resourceStates().size())
            return invalid("a traversal claim names an out-of-range state");
          const FrozenSpatialResourceState &state =
              resources.resourceStates()[claim.state];
          if (claim.dimension >= state.capacityCount)
            return invalid(
                "a traversal claim names an out-of-range capacity dimension");
          auto capacityDimension = checkedPnrIndexAdd(
              routeClaimCapacityContext, state.capacityOffset, claim.dimension);
          if (!capacityDimension)
            return capacityDimension.takeError();
          const FrozenSpatialCapacityDimension &capacity =
              resources.capacityDimensions()[*capacityDimension];
          auto qCost =
              normalizedRouteClaimCost(claim.amount, capacity.capacity);
          if (!qCost)
            return invalid("invalid normalized route claim: " +
                           llvm::toString(qCost.takeError()));

          const std::string key =
              routeClaimKey(use.requesterGroup, *capacityDimension);
          auto found = routeClaimByKey.find(key);
          if (found == routeClaimByKey.end()) {
            auto routeClaim =
                checked(routeClaimIndexContext, result.routeClaims_.size());
            if (!routeClaim)
              return routeClaim.takeError();
            found = routeClaimByKey.try_emplace(key, *routeClaim).first;
            result.routeClaims_.push_back(
                {use.requesterGroup, *capacityDimension, claim.amount, *qCost});
          } else {
            const FrozenSpatialRouteClaim &existing =
                result.routeClaims_[found->second];
            if (existing.amount != claim.amount || existing.qCost != *qCost)
              return invalid(
                  "one route claim key has inconsistent capacity demand");
          }
          traversalClaimKeys.push_back(found->second);
        }
      }
      llvm::sort(traversalClaimKeys);
      traversalClaimKeys.erase(
          std::unique(traversalClaimKeys.begin(), traversalClaimKeys.end()),
          traversalClaimKeys.end());
      if (llvm::Error error = preflightAppend(routeClaimCountContext,
                                              result.traversalClaimKeys_.size(),
                                              traversalClaimKeys.size()))
        return error;
      result.traversalClaimKeys_.insert(result.traversalClaimKeys_.end(),
                                        traversalClaimKeys.begin(),
                                        traversalClaimKeys.end());
      auto routeClaimCount =
          checked(routeClaimCountContext, traversalClaimKeys.size());
      if (!routeClaimCount)
        return routeClaimCount.takeError();
      result.traversals_.push_back(
          {routingTraversal.reference, routingTraversal.sourceOffset,
           routingTraversal.sourceCount, routingTraversal.destinationOffset,
           routingTraversal.destinationCount, *resourceStateOffset,
           *resourceStateCount, *routeClaimOffset, *routeClaimCount,
           traversal.timing.architecturalLatencyCycles,
           traversal.timing.releaseLatencyCycles,
           traversal.timing.minimumInitiationIntervalCycles,
           physicalTimingFound->second->delayQuanta,
           physicalTimingFound->second->boundary});
    }

    result.capacityRouteClaimOffsets_.assign(
        resources.capacityDimensions().size() + 1, 0);
    for (const FrozenSpatialRouteClaim &claim : result.routeClaims_) {
      auto count = checkedPnrIndexAdd(
          capacityRouteClaimCountContext,
          result.capacityRouteClaimOffsets_[claim.capacityDimension + 1], 1);
      if (!count)
        return count.takeError();
      result.capacityRouteClaimOffsets_[claim.capacityDimension + 1] = *count;
    }
    for (std::size_t capacity = 1;
         capacity < result.capacityRouteClaimOffsets_.size(); ++capacity) {
      auto prefix =
          checkedPnrIndexAdd(capacityRouteClaimOffsetContext,
                             result.capacityRouteClaimOffsets_[capacity - 1],
                             result.capacityRouteClaimOffsets_[capacity]);
      if (!prefix)
        return prefix.takeError();
      result.capacityRouteClaimOffsets_[capacity] = *prefix;
    }
    result.capacityRouteClaims_.resize(result.routeClaims_.size());
    std::vector<PnrIndex> capacityClaimCursors =
        result.capacityRouteClaimOffsets_;
    for (auto [claimOrdinal, claim] : llvm::enumerate(result.routeClaims_)) {
      auto claimIndex = checked(routeClaimIndexContext, claimOrdinal);
      if (!claimIndex)
        return claimIndex.takeError();
      result.capacityRouteClaims_
          [capacityClaimCursors[claim.capacityDimension]++] = *claimIndex;
    }

    result.routeClaimTraversalOffsets_.assign(result.routeClaims_.size() + 1,
                                              0);
    for (const FrozenSpatialTraversal &traversal : result.traversals_)
      for (PnrIndex claim :
           llvm::ArrayRef(result.traversalClaimKeys_)
               .slice(traversal.routeClaimOffset, traversal.routeClaimCount)) {
        auto count = checkedPnrIndexAdd(
            routeClaimTraversalCountContext,
            result.routeClaimTraversalOffsets_[claim + 1], 1);
        if (!count)
          return count.takeError();
        result.routeClaimTraversalOffsets_[claim + 1] = *count;
      }
    for (std::size_t claim = 1;
         claim < result.routeClaimTraversalOffsets_.size(); ++claim) {
      auto prefix =
          checkedPnrIndexAdd(routeClaimTraversalOffsetContext,
                             result.routeClaimTraversalOffsets_[claim - 1],
                             result.routeClaimTraversalOffsets_[claim]);
      if (!prefix)
        return prefix.takeError();
      result.routeClaimTraversalOffsets_[claim] = *prefix;
    }
    result.routeClaimTraversals_.resize(result.traversalClaimKeys_.size());
    std::vector<PnrIndex> claimTraversalCursors =
        result.routeClaimTraversalOffsets_;
    for (auto [traversalOrdinal, traversal] :
         llvm::enumerate(result.traversals_)) {
      auto traversalIndex = checked(traversalIndexContext, traversalOrdinal);
      if (!traversalIndex)
        return traversalIndex.takeError();
      for (PnrIndex claim :
           llvm::ArrayRef(result.traversalClaimKeys_)
               .slice(traversal.routeClaimOffset, traversal.routeClaimCount))
        result.routeClaimTraversals_[claimTraversalCursors[claim]++] =
            *traversalIndex;
    }

    result.traversalArcOffsets_.assign(result.traversals_.size() + 1, 0);
    for (const EndpointRoutingArc &arc : result.topology_->arcs()) {
      auto count =
          checkedPnrIndexAdd(traversalArcCountContext,
                             result.traversalArcOffsets_[arc.traversal + 1], 1);
      if (!count)
        return count.takeError();
      result.traversalArcOffsets_[arc.traversal + 1] = *count;
    }
    for (std::size_t traversal = 1;
         traversal < result.traversalArcOffsets_.size(); ++traversal) {
      auto prefix = checkedPnrIndexAdd(
          traversalArcOffsetContext, result.traversalArcOffsets_[traversal - 1],
          result.traversalArcOffsets_[traversal]);
      if (!prefix)
        return prefix.takeError();
      result.traversalArcOffsets_[traversal] = *prefix;
    }
    result.traversalArcs_.resize(result.topology_->arcs().size());
    std::vector<PnrIndex> traversalArcCursors = result.traversalArcOffsets_;
    for (auto [arcOrdinal, arc] : llvm::enumerate(result.topology_->arcs())) {
      auto arcIndex = checked(arcIndexContext, arcOrdinal);
      if (!arcIndex)
        return arcIndex.takeError();
      result.traversalArcs_[traversalArcCursors[arc.traversal]++] = *arcIndex;
    }
    return result;
  }
};

llvm::Expected<FrozenSpatialPnrProblemHandle>
loom::pnr::freezeSpatialPnrProblem(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const ResolvedPnrConfigView &config,
    const SpatialMappingConstraintSetView &constraints,
    const FabricDerivedContextBundle *derivedContexts) {
  return FrozenSpatialPnrProblemBuilder::build(dataflow, techMapping, fabric,
                                               physicalTiming, config,
                                               constraints, derivedContexts);
}

llvm::Expected<FrozenSpatialPnrProblemHandle>
loom::pnr::freezeSpatialPnrProblem(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const ResolvedPnrConfigView &config,
    const SpatialMappingConstraintSetView &constraints,
    const FabricDerivedContextBundle *derivedContexts) {
  auto physicalTiming =
      ::loom::fabric::projectNormalizedFabricPhysicalTimingProfile(fabric);
  if (!physicalTiming)
    return physicalTiming.takeError();
  return freezeSpatialPnrProblem(dataflow, techMapping, fabric, *physicalTiming,
                                 config, constraints, derivedContexts);
}

llvm::Expected<FabricDerivedContextBundle>
loom::pnr::buildFabricDerivedContextBundle(
    const FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
    DerivedContextCacheAccess *staticAccess,
    DerivedContextCacheAccess *timingAccess) {
  return FrozenSpatialPnrProblemBuilder::buildDerivedContexts(fabric,
                                                              physicalTiming,
                                                              staticAccess,
                                                              timingAccess);
}

llvm::Error loom::pnr::revalidateFabricDerivedContextBundle(
    const FabricDerivedContextBundle &bundle, const FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming) {
  return FrozenSpatialPnrProblemBuilder::revalidateDerivedContexts(
      bundle, fabric, physicalTiming);
}

llvm::Error loom::pnr::revalidateFrozenSpatialPnrCacheHit(
    const FrozenSpatialPnrProblem &problem,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const ResolvedPnrConfigView &config,
    const SpatialMappingConstraintSetView &constraints) {
  return detail::SpatialPnrProblemIdentity::revalidateCacheHit(
      problem, dataflow, techMapping, fabric, physicalTiming, config,
      constraints);
}

llvm::Error loom::pnr::revalidateFrozenSpatialPnrCacheHit(
    const FrozenSpatialPnrProblem &problem,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const ResolvedPnrConfigView &config,
    const SpatialMappingConstraintSetView &constraints) {
  auto physicalTiming =
      ::loom::fabric::projectNormalizedFabricPhysicalTimingProfile(fabric);
  if (!physicalTiming)
    return physicalTiming.takeError();
  return revalidateFrozenSpatialPnrCacheHit(problem, dataflow, techMapping,
                                            fabric, *physicalTiming, config,
                                            constraints);
}
