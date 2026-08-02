#include "PnR/SpatialPnrProblem.h"

#include "Common/ComponentViewDigest.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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
constexpr PnrCapacityContext traversalEndpointOffsetContext{
    frozenArtifact, "traversals", "traversal_endpoints",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext traversalEndpointCountContext{
    frozenArtifact, "traversal_endpoints", "traversal_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext arcOffsetContext{
    frozenArtifact, "adjacency", "routing_arcs", PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext arcCountContext{
    frozenArtifact, "routing_arcs", "routing_arcs", PnrCapacityMeasure::Count};

constexpr char cacheKeyDomain[] = "loom.spatial_pnr.frozen_model.key.v2\0";
constexpr std::size_t cacheKeyDomainSize = sizeof(cacheKeyDomain) - 1;
constexpr std::uint32_t cacheSchemaMajor = 2;
constexpr std::uint32_t cacheSchemaMinor = 0;
constexpr llvm::StringLiteral freezeSemanticIdentity =
    "loom.spatial_pnr.freeze.2.0";
constexpr llvm::StringLiteral importerSemanticIdentity =
    "loom.spatial_pnr.importers.2.0";
constexpr llvm::StringLiteral nativeLayoutAbi =
    "loom.spatial_pnr.native_layout.2.0";

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

    auto constraints = detail::buildFrozenConstraintIndex(constraintSet);
    if (!constraints)
      return constraints.takeError();
    auto realizations =
        buildRealizations(dataflow, techMapping, fabric, *constraints);
    if (!realizations)
      return realizations.takeError();
    auto routing = buildRouting(fabric);
    if (!routing)
      return routing.takeError();
    if (llvm::Error error = verifyAggregate(*realizations, *routing))
      return std::move(error);

    FrozenSpatialPnrCacheKey cacheKey =
        deriveCacheKey(dataflow, techMapping, fabric, config, constraintSet);
    std::vector<DeterministicWorkBudgetEntry> workBudget =
        deriveDeterministicWorkBudgetView(config);

    return FrozenSpatialPnrProblemHandle(new FrozenSpatialPnrProblem(
        dataflow.identity(), techMapping.identity(), fabric.identity(),
        constraintSet.identity(), config, std::move(workBudget),
        std::move(*constraints), std::move(*realizations), std::move(*routing),
        cacheKey));
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
                  const FrozenSpatialRoutingGraph &routing) {
    const auto rangeFits = [](PnrIndex offset, PnrIndex count,
                              std::size_t size) {
      const std::size_t begin = static_cast<std::size_t>(offset);
      const std::size_t length = static_cast<std::size_t>(count);
      return begin <= size && length <= size - begin;
    };

    for (auto [ordinal, realization] :
         llvm::enumerate(realizations.computeRealizations())) {
      if (!rangeFits(realization.actorOffset, realization.actorCount,
                     realizations.computeActors().size()) ||
          !rangeFits(realization.placementOffset, realization.placementCount,
                     realizations.computePlacements().size()) ||
          realization.placementCount == 0)
        return invalid("compute realization slices are inconsistent");
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
      for (const FrozenSpatialMemoryPlacement &placement :
           realizations.memoryPlacements().slice(realization.placementOffset,
                                                 realization.placementCount))
        if (placement.realization != ordinal)
          return invalid("memory placement slices are inconsistent");
    }

    if (routing.adjacencyOffsets().size() !=
            routing.routingEndpoints().size() + 1 ||
        routing.arcSources().size() != routing.routingArcs().size() ||
        routing.adjacencyOffsets().empty() ||
        routing.adjacencyOffsets().front() != 0 ||
        routing.adjacencyOffsets().back() != routing.routingArcs().size())
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
    for (const FrozenSpatialTraversal &traversal : routing.traversals()) {
      if (traversal.sourceCount == 0 || traversal.destinationCount == 0 ||
          !rangeFits(traversal.sourceOffset, traversal.sourceCount,
                     routing.traversalEndpoints().size()) ||
          !rangeFits(traversal.destinationOffset, traversal.destinationCount,
                     routing.traversalEndpoints().size()))
        return invalid("traversal endpoint slices are inconsistent");
      for (PnrIndex endpoint : routing.traversalEndpoints().slice(
               traversal.sourceOffset, traversal.sourceCount))
        if (endpoint >= routing.routingEndpoints().size())
          return invalid("traversal source endpoint is out of range");
      for (PnrIndex endpoint : routing.traversalEndpoints().slice(
               traversal.destinationOffset, traversal.destinationCount))
        if (endpoint >= routing.routingEndpoints().size())
          return invalid("traversal destination endpoint is out of range");
    }
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
      for (const TechComputeActorView &actor : realization.actors)
        result.computeActors_.push_back(actor.actor);
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
      for (const TechMemoryActorView &actor : realization.actors)
        result.memoryActors_.push_back({actor.actor, actor.operationPort});
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
  buildRouting(const FabricArtifactView &fabric) {
    const auto endpointRefs = fabric.transportEndpoints();
    const auto traversalViews = fabric.physicalTraversals();
    if (llvm::Error error = preflightPnrIndexCapacity(endpointCountContext,
                                                      endpointRefs.size()))
      return std::move(error);
    if (llvm::Error error = preflightPnrIndexCapacity(traversalCountContext,
                                                      traversalViews.size()))
      return std::move(error);

    FrozenSpatialRoutingGraph result;
    result.endpoints_.reserve(endpointRefs.size());
    std::map<std::vector<std::uint8_t>, PnrIndex> endpointByCanonicalRef;
    for (auto [ordinal, reference] : llvm::enumerate(endpointRefs)) {
      const auto direction = fabric.transportEndpointDirection(reference);
      const auto dataPath = fabric.transportEndpointDataPath(reference);
      if (!direction || !dataPath)
        return invalid("a canonical Fabric endpoint has no typed projection");
      result.endpoints_.push_back({reference, *direction, *dataPath});
      auto index = checked(endpointCountContext, ordinal);
      if (!index)
        return index.takeError();
      if (!endpointByCanonicalRef
               .emplace(canonicalFabricBytes(reference), *index)
               .second)
        return invalid(
            "the canonical Fabric endpoint inventory has a duplicate");
    }

    struct ArcDraft final {
      PnrIndex source;
      FrozenSpatialRoutingArc arc;
    };
    std::vector<ArcDraft> arcDrafts;
    result.traversals_.reserve(traversalViews.size());
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
        const auto index =
            endpointByCanonicalRef.find(canonicalFabricBytes(source));
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
        const auto index =
            endpointByCanonicalRef.find(canonicalFabricBytes(destination));
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
      result.traversals_.push_back({traversal.reference, *sourceOffset,
                                    *sourceCount, *destinationOffset,
                                    *destinationCount});

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
