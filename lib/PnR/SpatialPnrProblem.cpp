#include "PnR/SpatialPnrProblem.h"
#include "PnR/RoutingNegotiation.h"

#include "SpatialBindingRelationModel.h"
#include "SpatialPnrCapacityIndex.h"
#include "SpatialPnrHandshakeIndex.h"
#include "SpatialPnrMemoryIndex.h"
#include "SpatialPnrPortIndex.h"
#include "SpatialPnrResourceIndex.h"
#include "SpatialPnrTransferIndex.h"

#include "Common/ComponentViewDigest.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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
constexpr PnrCapacityContext endpointCountContext{
    frozenArtifact, "routing_endpoints", "routing_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext traversalCountContext{
    frozenArtifact, "traversals", "traversals", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext traversalIndexContext{
    frozenArtifact, "traversals", "traversals", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext replicationGroupIndexContext{
    frozenArtifact, "traversal_replication_groups", "replication_groups",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext traversalEndpointOffsetContext{
    frozenArtifact, "traversals", "traversal_endpoints",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext traversalEndpointCountContext{
    frozenArtifact, "traversal_endpoints", "traversal_endpoints",
    PnrCapacityMeasure::Count};
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
constexpr PnrCapacityContext arcOffsetContext{
    frozenArtifact, "adjacency", "routing_arcs", PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext arcCountContext{
    frozenArtifact, "routing_arcs", "routing_arcs", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext arcIndexContext{
    frozenArtifact, "routing_arcs", "routing_arcs", PnrCapacityMeasure::Index};

constexpr char cacheKeyDomain[] = "loom.spatial_pnr.frozen_model.key.v2.7\0";
constexpr std::size_t cacheKeyDomainSize = sizeof(cacheKeyDomain) - 1;
constexpr std::uint32_t cacheSchemaMajor = 2;
constexpr std::uint32_t cacheSchemaMinor = 7;
constexpr llvm::StringLiteral freezeSemanticIdentity =
    "loom.spatial_pnr.freeze.2.7";
constexpr llvm::StringLiteral importerSemanticIdentity =
    "loom.spatial_pnr.importers.2.1";
constexpr llvm::StringLiteral nativeLayoutAbi =
    "loom.spatial_pnr.native_layout.2.6";

enum class CacheField : std::uint32_t {
  DataflowIdentity = 1,
  TechMappingIdentity = 2,
  FabricIdentity = 3,
  ConstraintSetIdentity = 4,
  ConfigViewDescriptor = 5,
  ConfigViewDigest = 6,
  FreezeSemanticIdentity = 7,
  ImporterSemanticIdentity = 8,
  NativeLayoutAbi = 9,
  PnrIndexWidth = 10,
};

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

void appendField(std::vector<std::uint8_t> &bytes, CacheField field,
                 llvm::ArrayRef<std::uint8_t> value) {
  appendU32Be(bytes, static_cast<std::uint32_t>(field));
  appendU64Be(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendField(std::vector<std::uint8_t> &bytes, CacheField field,
                 llvm::StringRef value) {
  appendField(
      bytes, field,
      llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(value.data()), value.size()));
}

void appendU32Field(std::vector<std::uint8_t> &bytes, CacheField field,
                    std::uint32_t value) {
  std::vector<std::uint8_t> encoded;
  encoded.reserve(4);
  appendU32Be(encoded, value);
  appendField(bytes, field, encoded);
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

std::uint32_t tagCapacity(const ::fabric::DataPathType &path) {
  return path.kind == ::fabric::DataPathKind::BitsTag ? path.tagWidthBits : 0;
}

std::string
routeClaimKey(const FabricTraversalActivationGroupView &activationGroup,
              PnrIndex capacityDimension) {
  std::vector<std::uint8_t> bytes;
  const auto owner = canonicalFabricBytes(activationGroup.owner);
  bytes.reserve(20 + owner.size());
  appendU32Be(bytes, static_cast<std::uint32_t>(activationGroup.kind));
  appendU64Be(bytes, activationGroup.ordinal);
  appendU64Be(bytes, capacityDimension);
  bytes.insert(bytes.end(), owner.begin(), owner.end());
  return byteKey(bytes);
}

std::string
activationGroupKey(const FabricTraversalActivationGroupView &activationGroup) {
  std::vector<std::uint8_t> bytes;
  const auto owner = canonicalFabricBytes(activationGroup.owner);
  bytes.reserve(12 + owner.size());
  appendU32Be(bytes, static_cast<std::uint32_t>(activationGroup.kind));
  appendU64Be(bytes, activationGroup.ordinal);
  bytes.insert(bytes.end(), owner.begin(), owner.end());
  return byteKey(bytes);
}

llvm::Error
validateInputs(const dataflow::CanonicalDataflowProgramView &dataflow,
               const TechMappingView &techMapping,
               const FabricArtifactView &fabric,
               const ResolvedPnrConfigView &config,
               const SpatialMappingConstraintSetView &constraintSet) {
  if (config.domain() != PnrConfigDomain::Spatial)
    return invalid("Spatial PnR requires the Spatial config projection");
  if (techMapping.dataflowIdentity() != dataflow.identity())
    return invalid("TechMapping is bound to a different Dataflow artifact");
  if (techMapping.fabricIdentity() != fabric.identity())
    return invalid("TechMapping is bound to a different Fabric artifact");
  if (constraintSet.dataflowIdentity() != dataflow.identity() ||
      constraintSet.techMappingIdentity() != techMapping.identity() ||
      constraintSet.fabricIdentity() != fabric.identity())
    return invalid("MappingConstraintSet is bound to a different D/T/F tuple");
  if (fabric.rootKind() != FabricRootKind::Module)
    return invalid("Spatial PnR requires one fully elaborated Module root");
  if (llvm::Error error = validateComponentViewDigest(
          config.schemaDescriptorBytes(), config.canonicalViewBytes(),
          config.digest()))
    return llvm::joinErrors(
        invalid("PnR config component-view digest is invalid"),
        std::move(error));
  return llvm::Error::success();
}

} // namespace

class loom::pnr::FrozenSpatialPnrProblemBuilder final {
public:
  static llvm::Expected<FrozenSpatialPnrProblemHandle>
  build(const dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping, const FabricArtifactView &fabric,
        const ResolvedPnrConfigView &config,
        const SpatialMappingConstraintSetView &constraintSet) {
    if (llvm::Error error = validateInputs(dataflow, techMapping, fabric,
                                           config, constraintSet))
      return std::move(error);

    auto objectiveProgram = SpatialObjectiveProgram::get(
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
    auto memory = FrozenSpatialMemoryIndexBuilder::build(dataflow, techMapping,
                                                         fabric, *realizations);
    if (!memory)
      return memory.takeError();
    auto transfers = detail::buildFrozenSpatialTransferIndex(techMapping);
    if (!transfers)
      return transfers.takeError();
    auto resources = detail::buildFrozenSpatialResourceIndex(fabric);
    if (!resources)
      return resources.takeError();
    auto routing = buildRouting(fabric, *resources);
    if (!routing)
      return routing.takeError();
    auto ports = detail::buildFrozenSpatialPortIndex(
        dataflow, techMapping, fabric, *realizations, *transfers, *routing);
    if (!ports)
      return ports.takeError();
    auto bindingRelations = detail::SpatialBindingRelationModel::create(
        *realizations, *constraints, *ports);
    if (!bindingRelations)
      return bindingRelations.takeError();
    auto handshake = detail::buildFrozenSpatialHandshakeIndex(
        dataflow, techMapping, fabric, *realizations, *resources, *routing);
    if (!handshake)
      return handshake.takeError();
    auto capacity = detail::buildFrozenSpatialCapacityIndex(
        dataflow, techMapping, fabric, *realizations, *memory, *resources,
        *routing, *handshake);
    if (!capacity)
      return capacity.takeError();
    if (llvm::Error error =
            verifyAggregate(*realizations, *memory, *transfers, *ports,
                            *resources, *capacity, *routing, *handshake))
      return std::move(error);

    FrozenSpatialPnrCacheKey cacheKey =
        deriveCacheKey(dataflow, techMapping, fabric, config, constraintSet);
    std::vector<DeterministicWorkBudgetEntry> workBudget =
        deriveDeterministicWorkBudgetView(config);

    return FrozenSpatialPnrProblemHandle(new FrozenSpatialPnrProblem(
        dataflow.identity(), techMapping.identity(), fabric.identity(),
        constraintSet.identity(), config, std::move(*objectiveProgram),
        std::move(workBudget), std::move(*constraints),
        std::move(*realizations), std::move(*memory), std::move(*transfers),
        std::move(*ports), std::move(*resources), std::move(*capacity),
        std::move(*routing), std::move(*handshake),
        std::move(*bindingRelations), cacheKey));
  }

  static FrozenSpatialPnrCacheKey
  deriveCacheKey(const dataflow::CanonicalDataflowProgramView &dataflow,
                 const TechMappingView &techMapping,
                 const FabricArtifactView &fabric,
                 const ResolvedPnrConfigView &config,
                 const SpatialMappingConstraintSetView &constraintSet) {
    std::vector<std::uint8_t> preimage;
    preimage.reserve(cacheKeyDomainSize + 2 * sizeof(std::uint32_t) + 512);
    preimage.insert(preimage.end(), cacheKeyDomain,
                    cacheKeyDomain + cacheKeyDomainSize);
    appendU32Be(preimage, cacheSchemaMajor);
    appendU32Be(preimage, cacheSchemaMinor);
    appendField(preimage, CacheField::DataflowIdentity,
                dataflow.identity().bytes());
    appendField(preimage, CacheField::TechMappingIdentity,
                techMapping.identity().bytes());
    appendField(preimage, CacheField::FabricIdentity,
                fabric.identity().bytes());
    appendField(preimage, CacheField::ConstraintSetIdentity,
                constraintSet.identity().bytes());
    appendField(preimage, CacheField::ConfigViewDescriptor,
                config.schemaDescriptorBytes());
    appendField(preimage, CacheField::ConfigViewDigest,
                config.digest().bytes());
    appendField(preimage, CacheField::FreezeSemanticIdentity,
                freezeSemanticIdentity);
    appendField(preimage, CacheField::ImporterSemanticIdentity,
                importerSemanticIdentity);
    appendField(preimage, CacheField::NativeLayoutAbi, nativeLayoutAbi);
    appendU32Field(preimage, CacheField::PnrIndexWidth, getPnrIndexBits());
    return FrozenSpatialPnrCacheKey(llvm::SHA256::hash(preimage));
  }

  static llvm::Error
  verifyAggregate(const FrozenSpatialRealizationIndex &realizations,
                  const FrozenSpatialMemoryIndex &memory,
                  const FrozenSpatialTransferIndex &transfers,
                  const FrozenSpatialPortIndex &ports,
                  const FrozenSpatialResourceIndex &resources,
                  const FrozenSpatialCapacityIndex &capacity,
                  const FrozenSpatialRoutingGraph &routing,
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

    if (realizations.computeActorRealizations().size() !=
            realizations.computeActors().size() ||
        realizations.memoryActorRealizations().size() !=
            realizations.memoryActors().size())
      return invalid("actor-owner reverse projections are incomplete");
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
        if (placement.realization != ordinal)
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
        const FrozenSpatialRoutingArc &record = routing.routingArcs()[arc];
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
               .try_emplace(routeClaimKey(claim.activationGroup,
                                          claim.capacityDimension),
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
    for (const FrozenSpatialRoutingArc &arc : routing.routingArcs())
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
      PnrIndex expectedReplicationGroup = getInvalidPnrIndex();
      for (PnrIndex claimOrdinal : routing.traversalClaimKeys().slice(
               traversal.routeClaimOffset, traversal.routeClaimCount)) {
        const FabricTraversalActivationGroupView &activation =
            routing.routeClaims()[claimOrdinal].activationGroup;
        if (activation.kind !=
            FabricTraversalActivationGroupKind::SwitchRequester)
          continue;
        const std::string key = activationGroupKey(activation);
        auto found = expectedReplicationGroups.find(key);
        if (found == expectedReplicationGroups.end()) {
          auto group = checked(replicationGroupIndexContext,
                               expectedReplicationGroups.size());
          if (!group)
            return group.takeError();
          found = expectedReplicationGroups.try_emplace(key, *group).first;
        }
        if (expectedReplicationGroup != getInvalidPnrIndex() &&
            expectedReplicationGroup != found->second)
          return invalid("one traversal names multiple replication groups");
        expectedReplicationGroup = found->second;
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

  static llvm::Error
  revalidateCacheHit(const FrozenSpatialPnrProblem &problem,
                     const dataflow::CanonicalDataflowProgramView &dataflow,
                     const TechMappingView &techMapping,
                     const FabricArtifactView &fabric,
                     const ResolvedPnrConfigView &config,
                     const SpatialMappingConstraintSetView &constraintSet) {
    if (llvm::Error error = validateInputs(dataflow, techMapping, fabric,
                                           config, constraintSet))
      return error;
    if (problem.dataflowIdentity() != dataflow.identity() ||
        problem.techMappingIdentity() != techMapping.identity() ||
        problem.fabricIdentity() != fabric.identity() ||
        problem.constraintSetIdentity() != constraintSet.identity())
      return invalid("cache hit does not bind the exact artifact inputs");
    if (problem.config().schemaDescriptorBytes() !=
            config.schemaDescriptorBytes() ||
        problem.config().canonicalViewBytes() != config.canonicalViewBytes() ||
        problem.config().digest() != config.digest())
      return invalid("cache hit does not bind the exact PnR config view");
    if (problem.cacheKey() !=
        deriveCacheKey(dataflow, techMapping, fabric, config, constraintSet))
      return invalid("cache key does not match its dependency closure");
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

      for (FabricFuOccurrenceRef fu : fabric.fuOccurrences()) {
        const std::optional<FabricFuTemplateRef> definition =
            fabric.fuTemplateOf(fu);
        if (!definition || *definition != realization.capabilityTemplate.fu ||
            !domainContains(fuDomain, fu))
          continue;
        const std::optional<FabricPeOccurrenceRef> parent =
            fabric.parentPeOf(fu);
        if (!parent)
          return invalid("a Fabric FU occurrence has no parent PE relation");
        if (!domainContains(peDomain, *parent))
          continue;
        auto contextOffset = checked(contextOffsetContext,
                                     result.computeInstructionContexts_.size());
        if (!contextOffset)
          return contextOffset.takeError();
        const std::uint64_t contextCount =
            fabric.peResidentContextCount(*parent);
        if (llvm::Error error = preflightAppend(
                contextCountContext, result.computeInstructionContexts_.size(),
                contextCount))
          return error;
        for (std::uint64_t ordinal = 0; ordinal < contextCount; ++ordinal) {
          const InstructionContextRef context{*parent, ordinal};
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
        result.computePlacements_.push_back({*realizationIndex, fu, *parent,
                                             *contextOffset,
                                             *frozenContextCount});
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
        result.memoryPlacements_.push_back({*realizationIndex, memory});
      }
      const std::size_t placementCountValue =
          result.memoryPlacements_.size() - *placementOffset;
      if (placementCountValue == 0)
        return infeasible(
            "a memory realization has no legal memory occurrence");
      auto placementCount = checked(placementCountContext, placementCountValue);
      if (!placementCount)
        return placementCount.takeError();
      result.memoryRealizations_.push_back({subject, realization.engine,
                                            *actorOffset, *actorCount,
                                            *placementOffset, *placementCount});
    }
    return result;
  }

  static llvm::Expected<FrozenSpatialRoutingGraph>
  buildRouting(const FabricArtifactView &fabric,
               const FrozenSpatialResourceIndex &resources) {
    const auto endpointRefs = fabric.transportEndpoints();
    const auto traversalViews = fabric.physicalTraversals();
    if (llvm::Error error = preflightPnrIndexCapacity(endpointCountContext,
                                                      endpointRefs.size()))
      return std::move(error);
    if (llvm::Error error = preflightPnrIndexCapacity(traversalCountContext,
                                                      traversalViews.size()))
      return std::move(error);

    FrozenSpatialRoutingGraph result;
    auto tagContinuity = freezeSpatialTagContinuityIndex(fabric);
    if (!tagContinuity)
      return tagContinuity.takeError();
    result.tagContinuity_ = std::move(*tagContinuity);
    result.endpoints_.reserve(endpointRefs.size());
    llvm::StringMap<PnrIndex> endpointByCanonicalRef;
    for (auto [ordinal, reference] : llvm::enumerate(endpointRefs)) {
      const auto direction = fabric.transportEndpointDirection(reference);
      const auto dataPath = fabric.transportEndpointDataPath(reference);
      if (!direction || !dataPath)
        return invalid("a canonical Fabric endpoint has no typed projection");
      result.endpoints_.push_back({reference, *direction, *dataPath});
      auto index = checked(endpointCountContext, ordinal);
      if (!index)
        return index.takeError();
      if (!endpointByCanonicalRef.try_emplace(refKey(reference), *index).second)
        return invalid(
            "the canonical Fabric endpoint inventory has a duplicate");
    }
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
    llvm::StringMap<PnrIndex> replicationGroupByKey;

    struct ArcDraft final {
      PnrIndex source;
      FrozenSpatialRoutingArc arc;
    };
    std::vector<ArcDraft> arcDrafts;
    result.traversals_.reserve(traversalViews.size());
    result.traversalReplicationGroups_.reserve(traversalViews.size());
    for (auto [traversalOrdinal, traversal] : llvm::enumerate(traversalViews)) {
      auto traversalIndex = checked(traversalIndexContext, traversalOrdinal);
      if (!traversalIndex)
        return traversalIndex.takeError();
      auto endpointEnd = checkedPnrIndexAdd(traversalEndpointCountContext,
                                            result.traversalEndpoints_.size(),
                                            traversal.sources.size());
      if (!endpointEnd)
        return endpointEnd.takeError();
      endpointEnd =
          checkedPnrIndexAdd(traversalEndpointCountContext, *endpointEnd,
                             traversal.destinations.size());
      if (!endpointEnd)
        return endpointEnd.takeError();
      auto arcProduct =
          checkedPnrIndexMultiply(arcCountContext, traversal.sources.size(),
                                  traversal.destinations.size());
      if (!arcProduct)
        return arcProduct.takeError();
      if (llvm::Error error =
              preflightAppend(arcCountContext, arcDrafts.size(), *arcProduct))
        return error;
      auto sourceOffset = checked(traversalEndpointOffsetContext,
                                  result.traversalEndpoints_.size());
      if (!sourceOffset)
        return sourceOffset.takeError();
      std::vector<PnrIndex> sources;
      sources.reserve(traversal.sources.size());
      for (const FabricTransportEndpointRef &source : traversal.sources) {
        const auto index = endpointByCanonicalRef.find(refKey(source));
        if (index == endpointByCanonicalRef.end())
          return invalid(
              "a traversal source is absent from the endpoint inventory");
        sources.push_back(index->second);
        result.traversalEndpoints_.push_back(index->second);
      }
      auto sourceCount = checked(traversalEndpointCountContext, sources.size());
      if (!sourceCount)
        return sourceCount.takeError();
      auto destinationOffset = checked(traversalEndpointOffsetContext,
                                       result.traversalEndpoints_.size());
      if (!destinationOffset)
        return destinationOffset.takeError();
      std::vector<PnrIndex> destinations;
      destinations.reserve(traversal.destinations.size());
      for (const FabricTransportEndpointRef &destination :
           traversal.destinations) {
        const auto index = endpointByCanonicalRef.find(refKey(destination));
        if (index == endpointByCanonicalRef.end())
          return invalid(
              "a traversal destination is absent from the endpoint inventory");
        destinations.push_back(index->second);
        result.traversalEndpoints_.push_back(index->second);
      }
      auto destinationCount =
          checked(traversalEndpointCountContext, destinations.size());
      if (!destinationCount)
        return destinationCount.takeError();
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
      PnrIndex replicationGroup = getInvalidPnrIndex();
      for (const FabricTraversalUseView &use : traversal.impliedUses) {
        const auto patternFound =
            patternByCanonicalRef.find(refKey(use.pattern));
        if (patternFound == patternByCanonicalRef.end())
          return invalid(
              "a traversal use is absent from the resource inventory");
        const FrozenSpatialUsePattern &pattern =
            resources.usePatterns()[patternFound->second];
        if (use.activationGroup.owner != pattern.reference.owner.catalog())
          return invalid("a traversal activation names a foreign owner");
        switch (use.activationGroup.kind) {
        case FabricTraversalActivationGroupKind::UsePattern:
          if (use.activationGroup.ordinal != pattern.reference.ordinal)
            return invalid(
                "a traversal activation disagrees with its use pattern");
          break;
        case FabricTraversalActivationGroupKind::SwitchRequester:
          if (use.activationGroup.ordinal != pattern.requester)
            return invalid(
                "a switch traversal activation disagrees with its requester");
          {
            const std::string key = activationGroupKey(use.activationGroup);
            auto found = replicationGroupByKey.find(key);
            if (found == replicationGroupByKey.end()) {
              auto group = checked(replicationGroupIndexContext,
                                   replicationGroupByKey.size());
              if (!group)
                return group.takeError();
              found = replicationGroupByKey.try_emplace(key, *group).first;
            }
            if (replicationGroup != getInvalidPnrIndex() &&
                replicationGroup != found->second)
              return invalid("one traversal names multiple replication groups");
            replicationGroup = found->second;
          }
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
              routeClaimKey(use.activationGroup, *capacityDimension);
          auto found = routeClaimByKey.find(key);
          if (found == routeClaimByKey.end()) {
            auto routeClaim =
                checked(routeClaimIndexContext, result.routeClaims_.size());
            if (!routeClaim)
              return routeClaim.takeError();
            found = routeClaimByKey.try_emplace(key, *routeClaim).first;
            result.routeClaims_.push_back({use.activationGroup,
                                           *capacityDimension, claim.amount,
                                           *qCost});
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
          {traversal.reference, *sourceOffset, *sourceCount, *destinationOffset,
           *destinationCount, *resourceStateOffset, *resourceStateCount,
           *routeClaimOffset, *routeClaimCount});
      result.traversalReplicationGroups_.push_back(replicationGroup);

      for (PnrIndex source : sources) {
        const auto &sourcePath = result.endpoints_[source].dataPath;
        for (PnrIndex destination : destinations) {
          const auto &destinationPath = result.endpoints_[destination].dataPath;
          arcDrafts.push_back({source,
                               {destination, *traversalIndex,
                                std::min(sourcePath.payloadWidthBits,
                                         destinationPath.payloadWidthBits),
                                std::min(tagCapacity(sourcePath),
                                         tagCapacity(destinationPath))}});
        }
      }
    }
    llvm::sort(arcDrafts, [](const ArcDraft &lhs, const ArcDraft &rhs) {
      return std::tie(lhs.source, lhs.arc.target, lhs.arc.traversal) <
             std::tie(rhs.source, rhs.arc.target, rhs.arc.traversal);
    });
    result.adjacencyOffsets_.reserve(result.endpoints_.size() + 1);
    std::size_t cursor = 0;
    for (std::size_t source = 0; source < result.endpoints_.size(); ++source) {
      auto offset = checked(arcOffsetContext, result.arcs_.size());
      if (!offset)
        return offset.takeError();
      result.adjacencyOffsets_.push_back(*offset);
      while (cursor < arcDrafts.size() && arcDrafts[cursor].source == source) {
        result.arcSources_.push_back(arcDrafts[cursor].source);
        result.arcs_.push_back(arcDrafts[cursor++].arc);
      }
    }
    auto end = checked(arcOffsetContext, result.arcs_.size());
    if (!end)
      return end.takeError();
    result.adjacencyOffsets_.push_back(*end);
    if (cursor != arcDrafts.size())
      return invalid("routing CSR construction left an out-of-range source");

    std::vector<PnrIndex> reverseOffsets(result.endpoints_.size() + 1, 0);
    for (const FrozenSpatialRoutingArc &arc : result.arcs_) {
      auto count = checkedPnrIndexAdd(arcCountContext,
                                      reverseOffsets[arc.target + 1], 1);
      if (!count)
        return count.takeError();
      reverseOffsets[arc.target + 1] = *count;
    }
    for (std::size_t endpoint = 1; endpoint < reverseOffsets.size();
         ++endpoint) {
      auto prefix =
          checkedPnrIndexAdd(arcOffsetContext, reverseOffsets[endpoint - 1],
                             reverseOffsets[endpoint]);
      if (!prefix)
        return prefix.takeError();
      reverseOffsets[endpoint] = *prefix;
    }
    result.reverseAdjacencyOffsets_ = reverseOffsets;
    result.reverseArcOrdinals_.resize(result.arcs_.size());
    std::vector<PnrIndex> reverseCursors = std::move(reverseOffsets);
    for (auto [arcOrdinal, arc] : llvm::enumerate(result.arcs_)) {
      auto index = checked(arcIndexContext, arcOrdinal);
      if (!index)
        return index.takeError();
      const PnrIndex slot = reverseCursors[arc.target]++;
      result.reverseArcOrdinals_[slot] = *index;
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
    for (const FrozenSpatialRoutingArc &arc : result.arcs_) {
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
    result.traversalArcs_.resize(result.arcs_.size());
    std::vector<PnrIndex> traversalArcCursors = result.traversalArcOffsets_;
    for (auto [arcOrdinal, arc] : llvm::enumerate(result.arcs_)) {
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
    const ResolvedPnrConfigView &config,
    const SpatialMappingConstraintSetView &constraints) {
  return FrozenSpatialPnrProblemBuilder::build(dataflow, techMapping, fabric,
                                               config, constraints);
}

llvm::Error loom::pnr::revalidateFrozenSpatialPnrCacheHit(
    const FrozenSpatialPnrProblem &problem,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const ResolvedPnrConfigView &config,
    const SpatialMappingConstraintSetView &constraints) {
  return FrozenSpatialPnrProblemBuilder::revalidateCacheHit(
      problem, dataflow, techMapping, fabric, config, constraints);
}
