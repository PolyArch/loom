#include "SpatialPnrHandshakeIndex.h"

#include "Common/IndexWidth.h"
#include "Common/PointerLayout.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialPnrProblem";
constexpr PnrCapacityContext nodeCountContext{frozenArtifact, "handshake_nodes",
                                              "handshake_nodes",
                                              PnrCapacityMeasure::Count};
constexpr PnrCapacityContext nodeIndexContext{frozenArtifact, "handshake_nodes",
                                              "handshake_nodes",
                                              PnrCapacityMeasure::Index};
constexpr PnrCapacityContext arcCountContext{frozenArtifact, "handshake_arcs",
                                             "handshake_arcs",
                                             PnrCapacityMeasure::Count};
constexpr PnrCapacityContext arcIndexContext{frozenArtifact, "handshake_arcs",
                                             "handshake_arcs",
                                             PnrCapacityMeasure::Index};
constexpr PnrCapacityContext fragmentCountContext{
    frozenArtifact, "handshake_fragments", "handshake_fragments",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext fragmentIndexContext{
    frozenArtifact, "handshake_fragments", "handshake_fragments",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext incidenceCountContext{
    frozenArtifact, "handshake_incidence", "handshake_incidence",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext planCountContext{
    frozenArtifact, "memory_handshake_plans", "memory_handshake_plans",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext memoryDomainOffsetContext{
    frozenArtifact, "memory_handshake_domains", "memory_handshake_domains",
    PnrCapacityMeasure::Offset};

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

bool rangeFits(PnrIndex offset, PnrIndex count, std::size_t size) {
  return offset <= size && count <= size - static_cast<std::size_t>(offset);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename T> std::string refKey(const T &reference) {
  return byteKey(canonicalFabricBytes(reference));
}

std::string ownerKey(const FabricHandshakeOwner &owner) {
  std::vector<std::uint8_t> bytes{static_cast<std::uint8_t>(owner.kind())};
  std::visit(
      [&](const auto &payload) {
        if constexpr (std::is_same_v<std::decay_t<decltype(payload)>,
                                     FabricPointConnectionPayload>) {
          const auto source = canonicalFabricBytes(payload.source);
          const auto destination = canonicalFabricBytes(payload.destination);
          bytes.insert(bytes.end(), source.begin(), source.end());
          bytes.push_back(0xff);
          bytes.insert(bytes.end(), destination.begin(), destination.end());
        } else {
          const auto payloadBytes = canonicalFabricBytes(payload);
          bytes.insert(bytes.end(), payloadBytes.begin(), payloadBytes.end());
        }
      },
      owner.payload());
  return byteKey(bytes);
}

struct ArcPair final {
  PnrIndex source = 0;
  PnrIndex destination = 0;

  friend bool operator==(const ArcPair &lhs, const ArcPair &rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
};

struct ArcPairHash final {
  std::size_t operator()(const ArcPair &arc) const {
    std::uint64_t source = static_cast<std::uint64_t>(arc.source);
    std::uint64_t destination = static_cast<std::uint64_t>(arc.destination);
    source ^= source >> 30;
    source *= UINT64_C(0xbf58476d1ce4e5b9);
    destination ^= destination >> 27;
    destination *= UINT64_C(0x94d049bb133111eb);
    return static_cast<std::size_t>(source ^ destination);
  }
};

template <typename Key>
void stableCountingSort(llvm::ArrayRef<ArcPair> input,
                        llvm::MutableArrayRef<ArcPair> output,
                        std::size_t keyCount, Key key) {
  std::vector<std::size_t> offsets(keyCount + 1, 0);
  for (const ArcPair &arc : input)
    ++offsets[static_cast<std::size_t>(key(arc)) + 1];
  for (std::size_t index = 1; index < offsets.size(); ++index)
    offsets[index] += offsets[index - 1];
  for (const ArcPair &arc : input)
    output[offsets[static_cast<std::size_t>(key(arc))]++] = arc;
}

llvm::Expected<std::optional<PointerLayout>>
pointerLayoutFor(const dataflow::CanonicalDataflowProgramView &dataflow,
                 const dataflow::CanonicalActorSchemaProjection &actor) {
  auto addressSpace = dataflow::projectActorPointerAddressSpace(actor);
  if (!addressSpace)
    return addressSpace.takeError();
  if (!*addressSpace)
    return std::optional<PointerLayout>{};
  auto layout = dataflow.pointerLayout(**addressSpace);
  if (!layout)
    return layout.takeError();
  return std::optional<PointerLayout>(*layout);
}

llvm::Expected<dataflow::semantics::MemoryMaskForm>
memoryMaskForm(mlir::Operation *actor) {
  auto projection = dataflow::projectRegisteredActorSchemaProjection(actor);
  if (!projection)
    return projection.takeError();
  auto kind = dataflow::semantics::getMemoryServiceKind(projection->schema);
  if (!kind)
    return kind.takeError();
  if (*kind == dataflow::semantics::ServiceKind::MemoryFence)
    return dataflow::semantics::MemoryMaskForm::Absent;
  auto access = dataflow::semantics::getCanonicalMemoryAccessView(actor);
  if (!access)
    return access.takeError();
  return access->maskForm();
}

template <typename Values>
llvm::Error flattenSlices(const Values &values, std::vector<PnrIndex> &offsets,
                          std::vector<PnrIndex> &flattened) {
  offsets.clear();
  flattened.clear();
  offsets.reserve(values.size() + 1);
  auto zero = checked(incidenceCountContext, 0);
  if (!zero)
    return zero.takeError();
  offsets.push_back(*zero);
  for (const auto &slice : values) {
    if (llvm::Error error = preflightPnrIndexCapacity(
            incidenceCountContext,
            static_cast<std::uint64_t>(flattened.size()) + slice.size()))
      return error;
    flattened.insert(flattened.end(), slice.begin(), slice.end());
    auto end = checked(incidenceCountContext, flattened.size());
    if (!end)
      return end.takeError();
    offsets.push_back(*end);
  }
  return llvm::Error::success();
}

} // namespace

class loom::pnr::FrozenSpatialHandshakeIndexBuilder final {
public:
  static llvm::Expected<FrozenSpatialHandshakeIndex>
  build(const dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping, const FabricArtifactView &fabric,
        const FrozenSpatialRealizationIndex &realizations,
        const FrozenSpatialResourceIndex &resources,
        const FrozenSpatialRoutingGraph &routing) {
    auto compiled = compileHandshakeOwnerModels(fabric);
    if (!compiled)
      return compiled.takeError();

    FrozenSpatialHandshakeIndex result;
    BuildState state{result, *compiled, routing};
    if (llvm::Error error = state.buildNodesAndArcs())
      return std::move(error);
    if (llvm::Error error = state.buildFragments())
      return std::move(error);
    if (llvm::Error error = state.buildComputeSelections(dataflow, techMapping,
                                                         fabric, realizations))
      return std::move(error);
    if (llvm::Error error = state.buildMemorySelections(
            dataflow, techMapping, fabric, realizations, resources))
      return std::move(error);
    if (llvm::Error error = detail::verifyFrozenSpatialHandshakeIndex(
            result, realizations, resources, routing))
      return std::move(error);
    return result;
  }

private:
  class BuildState final {
  public:
    BuildState(FrozenSpatialHandshakeIndex &result,
               llvm::ArrayRef<HandshakeOwnerModel> models,
               const FrozenSpatialRoutingGraph &routing)
        : result_(result), models_(models), routing_(routing) {}

    llvm::Error buildNodesAndArcs() {
      auto endpointSignalCount = checkedPnrIndexMultiply(
          nodeCountContext, routing_.routingEndpoints().size(), 2);
      if (!endpointSignalCount)
        return endpointSignalCount.takeError();
      result_.nodeSignals_.reserve(*endpointSignalCount);
      for (auto [endpointOrdinal, endpoint] :
           llvm::enumerate(routing_.routingEndpoints())) {
        endpointOrdinals_.try_emplace(refKey(endpoint.reference),
                                      static_cast<PnrIndex>(endpointOrdinal));
        result_.nodeSignals_.push_back(
            HandshakeSignalRef{endpoint.reference, HandshakeSignalKind::Valid});
        result_.nodeSignals_.push_back(
            HandshakeSignalRef{endpoint.reference, HandshakeSignalKind::Ready});
      }

      modelNodes_.resize(models_.size());
      modelArcPairs_.resize(models_.size());
      std::vector<ArcPair> allArcs;
      for (auto [modelOrdinal, model] : llvm::enumerate(models_)) {
        modelOrdinals_.try_emplace(ownerKey(model.owner()),
                                   static_cast<PnrIndex>(modelOrdinal));
        auto &nodeMap = modelNodes_[modelOrdinal];
        nodeMap.reserve(model.nodes().size());
        for (const HandshakeOwnerNode &node : model.nodes()) {
          if (node.boundarySignal) {
            auto endpoint =
                endpointOrdinals_.find(refKey(node.boundarySignal->endpoint));
            if (endpoint == endpointOrdinals_.end())
              return invalid(
                  "handshake owner names an unknown routing endpoint");
            const PnrIndex signal =
                node.boundarySignal->signal == HandshakeSignalKind::Valid ? 0
                                                                          : 1;
            nodeMap.push_back(endpoint->second * 2 + signal);
            continue;
          }
          auto ordinal = checked(nodeIndexContext, result_.nodeSignals_.size());
          if (!ordinal)
            return ordinal.takeError();
          nodeMap.push_back(*ordinal);
          result_.nodeSignals_.push_back(std::nullopt);
        }

        auto &arcPairs = modelArcPairs_[modelOrdinal];
        arcPairs.reserve(model.arcs().size());
        for (const HandshakeOwnerArc &arc : model.arcs()) {
          if (arc.source >= nodeMap.size() || arc.destination >= nodeMap.size())
            return invalid("handshake owner arc is out of range");
          ArcPair pair{nodeMap[arc.source], nodeMap[arc.destination]};
          arcPairs.push_back(pair);
          allArcs.push_back(pair);
        }
      }

      if (llvm::Error error = preflightPnrIndexCapacity(
              arcCountContext, static_cast<std::uint64_t>(allArcs.size())))
        return error;
      std::vector<ArcPair> scratch(allArcs.size());
      stableCountingSort(allArcs, scratch, result_.nodeSignals_.size(),
                         [](const ArcPair &arc) { return arc.destination; });
      stableCountingSort(scratch, allArcs, result_.nodeSignals_.size(),
                         [](const ArcPair &arc) { return arc.source; });
      allArcs.erase(std::unique(allArcs.begin(), allArcs.end()), allArcs.end());

      result_.arcs_.reserve(allArcs.size());
      result_.adjacencyOffsets_.assign(result_.nodeSignals_.size() + 1, 0);
      arcOrdinals_.reserve(allArcs.size());
      for (auto [ordinal, arc] : llvm::enumerate(allArcs)) {
        auto index = checked(arcIndexContext, ordinal);
        if (!index)
          return index.takeError();
        result_.arcs_.push_back({arc.source, arc.destination});
        arcOrdinals_.emplace(arc, *index);
        ++result_.adjacencyOffsets_[arc.source + 1];
      }
      for (std::size_t node = 1; node < result_.adjacencyOffsets_.size();
           ++node)
        result_.adjacencyOffsets_[node] += result_.adjacencyOffsets_[node - 1];

      result_.reverseAdjacencyOffsets_.assign(result_.nodeSignals_.size() + 1,
                                              0);
      for (const FrozenSpatialHandshakeArc &arc : result_.arcs_)
        ++result_.reverseAdjacencyOffsets_[arc.destination + 1];
      for (std::size_t node = 1; node < result_.reverseAdjacencyOffsets_.size();
           ++node)
        result_.reverseAdjacencyOffsets_[node] +=
            result_.reverseAdjacencyOffsets_[node - 1];
      result_.reverseArcOrdinals_.resize(result_.arcs_.size());
      std::vector<PnrIndex> cursor = result_.reverseAdjacencyOffsets_;
      cursor.pop_back();
      for (auto [ordinal, arc] : llvm::enumerate(result_.arcs_))
        result_.reverseArcOrdinals_[cursor[arc.destination]++] =
            static_cast<PnrIndex>(ordinal);
      return llvm::Error::success();
    }

    llvm::Error buildFragments() {
      modelFragmentOffsets_.reserve(models_.size() + 1);
      modelFragmentOffsets_.push_back(0);
      std::vector<std::vector<PnrIndex>> traversalFragments(
          routing_.traversals().size());
      std::vector<std::vector<PnrIndex>> traversalAllGroups(
          routing_.traversals().size());
      for (auto [ordinal, traversal] : llvm::enumerate(routing_.traversals()))
        traversalOrdinals_.try_emplace(refKey(traversal.reference),
                                       static_cast<PnrIndex>(ordinal));

      for (auto [modelOrdinal, model] : llvm::enumerate(models_)) {
        for (auto [localFragmentOrdinal, fragment] :
             llvm::enumerate(model.fragments())) {
          auto globalFragment =
              checked(fragmentIndexContext, result_.fragments_.size());
          auto contributionOffset = checked(
              incidenceCountContext, result_.fragmentArcOrdinals_.size());
          if (!globalFragment)
            return globalFragment.takeError();
          if (!contributionOffset)
            return contributionOffset.takeError();
          std::vector<PnrIndex> contributions;
          contributions.reserve(fragment.contributionCount);
          for (std::uint32_t index = 0; index < fragment.contributionCount;
               ++index) {
            const std::uint32_t localArc =
                model.fragmentContributionOrdinals()
                    [fragment.contributionOffset + index];
            if (localArc >= modelArcPairs_[modelOrdinal].size())
              return invalid("handshake fragment arc is out of range");
            auto found =
                arcOrdinals_.find(modelArcPairs_[modelOrdinal][localArc]);
            if (found == arcOrdinals_.end())
              return invalid("handshake fragment arc was not flattened");
            contributions.push_back(found->second);
          }
          llvm::sort(contributions);
          contributions.erase(
              std::unique(contributions.begin(), contributions.end()),
              contributions.end());
          auto contributionCount =
              checked(incidenceCountContext, contributions.size());
          if (!contributionCount)
            return contributionCount.takeError();
          result_.fragmentArcOrdinals_.insert(
              result_.fragmentArcOrdinals_.end(), contributions.begin(),
              contributions.end());
          result_.fragments_.push_back(
              {*contributionOffset, *contributionCount});

          switch (fragment.activationKind) {
          case HandshakeActivationKind::Always:
            result_.fixedFragments_.push_back(*globalFragment);
            break;
          case HandshakeActivationKind::AnyTraversal:
            for (std::uint32_t witness = 0; witness < fragment.witnessCount;
                 ++witness) {
              auto traversal = traversalIndex(
                  model.traversalWitnesses()[fragment.witnessOffset + witness]);
              if (!traversal)
                return traversal.takeError();
              traversalFragments[*traversal].push_back(*globalFragment);
            }
            break;
          case HandshakeActivationKind::AllTraversals: {
            auto witnessOffset =
                checked(incidenceCountContext,
                        result_.allTraversalGroupWitnesses_.size());
            if (!witnessOffset)
              return witnessOffset.takeError();
            std::vector<PnrIndex> witnesses;
            witnesses.reserve(fragment.witnessCount);
            for (std::uint32_t witness = 0; witness < fragment.witnessCount;
                 ++witness) {
              auto traversal = traversalIndex(
                  model.traversalWitnesses()[fragment.witnessOffset + witness]);
              if (!traversal)
                return traversal.takeError();
              witnesses.push_back(*traversal);
            }
            llvm::sort(witnesses);
            witnesses.erase(std::unique(witnesses.begin(), witnesses.end()),
                            witnesses.end());
            auto witnessCount =
                checked(incidenceCountContext, witnesses.size());
            auto group = checked(incidenceCountContext,
                                 result_.allTraversalGroups_.size());
            if (!witnessCount)
              return witnessCount.takeError();
            if (!group)
              return group.takeError();
            result_.allTraversalGroupWitnesses_.insert(
                result_.allTraversalGroupWitnesses_.end(), witnesses.begin(),
                witnesses.end());
            result_.allTraversalGroups_.push_back(
                {*witnessOffset, *witnessCount, *globalFragment});
            for (PnrIndex traversal : witnesses)
              traversalAllGroups[traversal].push_back(*group);
            break;
          }
          case HandshakeActivationKind::ExactOwnerSelection:
            break;
          }
        }
        auto end = checked(fragmentCountContext, result_.fragments_.size());
        if (!end)
          return end.takeError();
        modelFragmentOffsets_.push_back(*end);
      }

      for (auto &fragments : traversalFragments) {
        llvm::sort(fragments);
        fragments.erase(std::unique(fragments.begin(), fragments.end()),
                        fragments.end());
      }
      for (auto &groups : traversalAllGroups) {
        llvm::sort(groups);
        groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
      }
      if (llvm::Error error = flattenSlices(traversalFragments,
                                            result_.traversalFragmentOffsets_,
                                            result_.traversalFragments_))
        return error;
      return flattenSlices(traversalAllGroups,
                           result_.traversalAllGroupOffsets_,
                           result_.traversalAllGroups_);
    }

    llvm::Error buildComputeSelections(
        const dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping, const FabricArtifactView &fabric,
        const FrozenSpatialRealizationIndex &realizations) {
      std::vector<std::vector<FabricFuOperationHandshakeBinding>> bindings;
      bindings.reserve(techMapping.computeRealizations().size());
      for (const TechComputeRealizationView &realization :
           techMapping.computeRealizations()) {
        std::vector<FabricFuOperationHandshakeBinding> actorBindings;
        actorBindings.reserve(realization.actors.size());
        for (const TechComputeActorView &actor : realization.actors) {
          auto resolved = dataflow.resolve(actor.actor);
          if (!resolved)
            return resolved.takeError();
          auto projection =
              dataflow::projectRegisteredActorSchemaProjection(resolved->op);
          if (!projection)
            return projection.takeError();
          auto indexBitWidth = getIndexBitWidth(resolved->op);
          if (!indexBitWidth)
            return indexBitWidth.takeError();
          auto pointerLayout = pointerLayoutFor(dataflow, *projection);
          if (!pointerLayout)
            return pointerLayout.takeError();
          actorBindings.push_back({actor.fabricOperation,
                                   std::move(*projection), *indexBitWidth,
                                   std::move(*pointerLayout),
                                   actor.operandPorts, actor.resultPorts});
        }
        bindings.push_back(std::move(actorBindings));
      }

      std::vector<std::vector<PnrIndex>> placementFragments(
          realizations.computePlacements().size());
      for (auto [placementOrdinal, placement] :
           llvm::enumerate(realizations.computePlacements())) {
        if (placement.realization >= techMapping.computeRealizations().size())
          return invalid("compute placement realization is out of range");
        const TechComputeRealizationView &realization =
            techMapping.computeRealizations()[placement.realization];
        auto selection = makeFuHandshakeSelection(
            fabric, placement.fu, realization.capabilityTemplate,
            bindings[placement.realization]);
        if (!selection)
          return selection.takeError();
        auto model = modelIndex(FabricHandshakeOwner::fu(placement.fu));
        if (!model)
          return model.takeError();
        FabricHandshakeSelection exact;
        exact.fuCapabilities.push_back(std::move(*selection));
        auto activation = resolveSelectedHandshake(models_[*model], exact);
        if (!activation)
          return activation.takeError();
        appendResolvedFragments(*model, activation->fragmentOrdinals(),
                                placementFragments[placementOrdinal]);
      }
      return flattenSlices(placementFragments,
                           result_.computePlacementFragmentOffsets_,
                           result_.computePlacementFragments_);
    }

    llvm::Error buildMemorySelections(
        const dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping, const FabricArtifactView &fabric,
        const FrozenSpatialRealizationIndex &realizations,
        const FrozenSpatialResourceIndex &resources) {
      llvm::StringMap<PnrIndex> usePatternOrdinals;
      for (auto [ordinal, pattern] : llvm::enumerate(resources.usePatterns()))
        usePatternOrdinals.try_emplace(refKey(pattern.reference),
                                       static_cast<PnrIndex>(ordinal));

      result_.memoryPlacementDomainOffsets_.reserve(
          realizations.memoryPlacements().size() + 1);
      for (auto [placementOrdinal, placement] :
           llvm::enumerate(realizations.memoryPlacements())) {
        auto domainOffset = checked(memoryDomainOffsetContext,
                                    result_.memoryOperationDomains_.size());
        if (!domainOffset)
          return domainOffset.takeError();
        result_.memoryPlacementDomainOffsets_.push_back(*domainOffset);
        if (placement.realization >= techMapping.memoryRealizations().size())
          return invalid("memory placement realization is out of range");
        const TechMemoryRealizationView &realization =
            techMapping.memoryRealizations()[placement.realization];
        const FrozenSpatialMemoryRealization &frozenRealization =
            realizations.memoryRealizations()[placement.realization];
        auto schedule = fabric.memorySchedule(placement.memory);
        if (!schedule)
          return invalid("memory placement has no scheduling contract");
        auto model = modelIndex(FabricHandshakeOwner::memory(placement.memory));
        if (!model)
          return model.takeError();

        for (auto [localActorOrdinal, actor] :
             llvm::enumerate(realization.actors)) {
          auto resolved = dataflow.resolve(actor.actor);
          if (!resolved)
            return resolved.takeError();
          auto maskForm = memoryMaskForm(resolved->op);
          if (!maskForm)
            return maskForm.takeError();
          const FabricMemoryOperationPortRef port{placement.memory,
                                                  actor.operationPort.ordinal};
          const FabricMemoryCapabilityAlternativeRef capability{
              port, actor.capability.ordinal};
          const MemoryCapabilityAlternativeView *alternative =
              fabric.memoryCapabilityAlternative(capability);
          if (!alternative)
            return invalid("memory handshake capability does not resolve");

          auto planOffset =
              checked(planCountContext, result_.memoryOperationPlans_.size());
          if (!planOffset)
            return planOffset.takeError();
          for (::fabric::UsePatternKey pattern :
               alternative->admissibleUsePatterns) {
            const FabricUsePatternRef usePattern{
                FabricUsePatternOwnerRef(FabricInventoryOwnerRef::of(port)),
                pattern.ordinal()};
            FabricMemoryHandshakePlacement operationPlacement = port;
            if (*schedule == ::fabric::Schedule::Temporal) {
              if (fabric.memoryResidentContextCount(placement.memory) == 0)
                return invalid("Temporal memory has no resident context");
              operationPlacement = FabricMemoryOperationContextRef{port, 0};
            }
            auto selected = makeMemoryHandshakeSelection(
                fabric, operationPlacement, capability, usePattern, *maskForm);
            if (!selected)
              return selected.takeError();
            FabricHandshakeSelection exact;
            exact.memoryOperations.push_back(std::move(*selected));
            auto activation = resolveSelectedHandshake(models_[*model], exact);
            if (!activation)
              return activation.takeError();

            auto usePatternOrdinal =
                usePatternOrdinals.find(refKey(usePattern));
            if (usePatternOrdinal == usePatternOrdinals.end())
              return invalid("memory handshake plan has no frozen use pattern");
            auto fragmentOffset = checked(incidenceCountContext,
                                          result_.memoryPlanFragments_.size());
            if (!fragmentOffset)
              return fragmentOffset.takeError();
            std::vector<PnrIndex> fragments;
            appendResolvedFragments(*model, activation->fragmentOrdinals(),
                                    fragments);
            auto fragmentCount =
                checked(incidenceCountContext, fragments.size());
            if (!fragmentCount)
              return fragmentCount.takeError();
            result_.memoryPlanFragments_.insert(
                result_.memoryPlanFragments_.end(), fragments.begin(),
                fragments.end());
            result_.memoryOperationPlans_.push_back(
                {usePatternOrdinal->second, *fragmentOffset, *fragmentCount});
          }
          const std::size_t planCountValue =
              result_.memoryOperationPlans_.size() - *planOffset;
          if (planCountValue == 0)
            return infeasible("memory operation has no handshake plan");
          auto planCount = checked(planCountContext, planCountValue);
          if (!planCount)
            return planCount.takeError();
          result_.memoryOperationDomains_.push_back(
              {static_cast<PnrIndex>(placementOrdinal),
               frozenRealization.actorOffset +
                   static_cast<PnrIndex>(localActorOrdinal),
               *planOffset, *planCount});
        }
      }
      auto domainEnd = checked(memoryDomainOffsetContext,
                               result_.memoryOperationDomains_.size());
      if (!domainEnd)
        return domainEnd.takeError();
      result_.memoryPlacementDomainOffsets_.push_back(*domainEnd);
      return llvm::Error::success();
    }

  private:
    llvm::Expected<PnrIndex>
    traversalIndex(const FabricPhysicalTraversalRef &reference) const {
      auto found = traversalOrdinals_.find(refKey(reference));
      if (found == traversalOrdinals_.end())
        return invalid("handshake traversal witness is absent from routing");
      return found->second;
    }

    llvm::Expected<PnrIndex> modelIndex(FabricHandshakeOwner owner) const {
      auto found = modelOrdinals_.find(ownerKey(owner));
      if (found == modelOrdinals_.end())
        return invalid("selected handshake owner has no compiled model");
      return found->second;
    }

    void appendResolvedFragments(PnrIndex model,
                                 llvm::ArrayRef<std::uint32_t> localFragments,
                                 std::vector<PnrIndex> &destination) const {
      for (std::uint32_t fragment : localFragments)
        destination.push_back(modelFragmentOffsets_[model] + fragment);
      llvm::sort(destination);
      destination.erase(std::unique(destination.begin(), destination.end()),
                        destination.end());
    }

    FrozenSpatialHandshakeIndex &result_;
    llvm::ArrayRef<HandshakeOwnerModel> models_;
    const FrozenSpatialRoutingGraph &routing_;
    llvm::StringMap<PnrIndex> endpointOrdinals_;
    llvm::StringMap<PnrIndex> traversalOrdinals_;
    llvm::StringMap<PnrIndex> modelOrdinals_;
    std::vector<std::vector<PnrIndex>> modelNodes_;
    std::vector<std::vector<ArcPair>> modelArcPairs_;
    std::unordered_map<ArcPair, PnrIndex, ArcPairHash> arcOrdinals_;
    std::vector<PnrIndex> modelFragmentOffsets_;
  };
};

llvm::Expected<FrozenSpatialHandshakeIndex>
loom::pnr::detail::buildFrozenSpatialHandshakeIndex(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing) {
  return FrozenSpatialHandshakeIndexBuilder::build(
      dataflow, techMapping, fabric, realizations, resources, routing);
}

llvm::Error loom::pnr::detail::verifyFrozenSpatialHandshakeIndex(
    const FrozenSpatialHandshakeIndex &handshake,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing) {
  const std::size_t nodeCount = handshake.nodeSignals().size();
  if (handshake.adjacencyOffsets().size() != nodeCount + 1 ||
      handshake.reverseAdjacencyOffsets().size() != nodeCount + 1 ||
      handshake.adjacencyOffsets().front() != 0 ||
      handshake.reverseAdjacencyOffsets().front() != 0 ||
      handshake.adjacencyOffsets().back() != handshake.arcs().size() ||
      handshake.reverseAdjacencyOffsets().back() != handshake.arcs().size() ||
      handshake.reverseArcOrdinals().size() != handshake.arcs().size())
    return invalid("handshake CSR shape is inconsistent");
  for (auto [ordinal, arc] : llvm::enumerate(handshake.arcs())) {
    if (arc.source >= nodeCount || arc.destination >= nodeCount)
      return invalid("handshake arc endpoint is out of range");
    if (ordinal != 0) {
      const auto &previous = handshake.arcs()[ordinal - 1];
      if (std::tie(previous.source, previous.destination) >=
          std::tie(arc.source, arc.destination))
        return invalid("handshake arcs are not unique source-major records");
    }
  }
  for (PnrIndex arc : handshake.reverseArcOrdinals())
    if (arc >= handshake.arcs().size())
      return invalid("reverse handshake incidence is out of range");
  for (const FrozenSpatialHandshakeFragment &fragment : handshake.fragments()) {
    if (!rangeFits(fragment.contributionOffset, fragment.contributionCount,
                   handshake.fragmentArcOrdinals().size()))
      return invalid("handshake fragment contribution slice is inconsistent");
    for (PnrIndex arc : handshake.fragmentArcOrdinals().slice(
             fragment.contributionOffset, fragment.contributionCount))
      if (arc >= handshake.arcs().size())
        return invalid("handshake fragment contribution is out of range");
  }
  for (PnrIndex fragment : handshake.fixedFragments())
    if (fragment >= handshake.fragments().size())
      return invalid("fixed handshake fragment is out of range");
  if (handshake.traversalFragmentOffsets().size() !=
          routing.traversals().size() + 1 ||
      handshake.traversalAllGroupOffsets().size() !=
          routing.traversals().size() + 1)
    return invalid("traversal handshake reverse incidence is incomplete");
  for (const FrozenSpatialHandshakeAllTraversalGroup &group :
       handshake.allTraversalGroups()) {
    if (group.witnessCount == 0 ||
        !rangeFits(group.witnessOffset, group.witnessCount,
                   handshake.allTraversalGroupWitnesses().size()) ||
        group.fragment >= handshake.fragments().size())
      return invalid("all-traversal handshake group is inconsistent");
  }
  if (handshake.computePlacementFragmentOffsets().size() !=
      realizations.computePlacements().size() + 1)
    return invalid("compute handshake incidence is incomplete");
  if (handshake.memoryPlacementDomainOffsets().size() !=
          realizations.memoryPlacements().size() + 1 ||
      handshake.memoryPlacementDomainOffsets().empty() ||
      handshake.memoryPlacementDomainOffsets().front() != 0 ||
      handshake.memoryPlacementDomainOffsets().back() !=
          handshake.memoryOperationDomains().size())
    return invalid("memory-placement handshake CSR is inconsistent");
  for (auto [placementOrdinal, placement] :
       llvm::enumerate(realizations.memoryPlacements())) {
    if (placement.realization >= realizations.memoryRealizations().size())
      return invalid("memory placement realization is out of range");
    const FrozenSpatialMemoryRealization &realization =
        realizations.memoryRealizations()[placement.realization];
    const PnrIndex begin =
        handshake.memoryPlacementDomainOffsets()[placementOrdinal];
    const PnrIndex end =
        handshake.memoryPlacementDomainOffsets()[placementOrdinal + 1];
    if (begin > end || end > handshake.memoryOperationDomains().size() ||
        end - begin != realization.actorCount)
      return invalid("memory placement does not cover its exact actor domain");
    for (PnrIndex localActor = 0; localActor < realization.actorCount;
         ++localActor) {
      const FrozenSpatialMemoryOperationHandshakeDomain &domain =
          handshake.memoryOperationDomains()[begin + localActor];
      if (domain.placement != placementOrdinal ||
          domain.actor != realization.actorOffset + localActor ||
          domain.planCount == 0 ||
          !rangeFits(domain.planOffset, domain.planCount,
                     handshake.memoryOperationPlans().size()))
        return invalid("memory handshake plan domain is inconsistent");
    }
  }
  for (const FrozenSpatialMemoryOperationHandshakePlan &plan :
       handshake.memoryOperationPlans()) {
    if (plan.usePattern >= resources.usePatterns().size() ||
        !rangeFits(plan.fragmentOffset, plan.fragmentCount,
                   handshake.memoryPlanFragments().size()))
      return invalid("memory handshake plan is inconsistent");
  }
  return llvm::Error::success();
}
