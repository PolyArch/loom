#include "SpatialPnrHandshakeIndex.h"

#include "Common/IndexWidth.h"
#include "Common/PointerLayout.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "HandshakeProjectionInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialPnrProblem";
constexpr PnrCapacityContext ownerIndexContext{
    frozenArtifact, "handshake_owners", "handshake_owners",
    PnrCapacityMeasure::Index};
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
constexpr PnrCapacityContext projectionNodeIndexContext{
    frozenArtifact, "handshake_projection", "node", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext projectionArcIndexContext{
    frozenArtifact, "handshake_projection", "arc", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext projectionArcOffsetContext{
    frozenArtifact, "handshake_projection", "arc", PnrCapacityMeasure::Offset};

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
        const FabricHandshakeContext &handshakeContext,
        const FrozenSpatialRealizationIndex &realizations,
        const FrozenSpatialResourceIndex &resources,
        const FrozenSpatialRoutingGraph &routing,
        const FrozenSpatialActiveRoutingDomain &activeRouting) {
    auto activeModels = selectActiveModels(handshakeContext.ownerModels(), realizations,
                                           routing, activeRouting);
    if (!activeModels)
      return activeModels.takeError();
    FrozenSpatialHandshakeIndex result;
    result.fabricContext_ = handshakeContext;
    BuildState state{result,      *activeModels, dataflow,
                     techMapping, fabric,        realizations,
                     resources,   routing,       activeRouting};
    if (llvm::Error error = state.prepareActiveFragments())
      return std::move(error);
    if (llvm::Error error = state.buildFragments())
      return std::move(error);
    if (llvm::Error error = state.materializeExactSelections())
      return std::move(error);
    if (llvm::Error error = state.buildDenseProjectionIndex())
      return std::move(error);
    if (llvm::Error error = detail::verifyFrozenSpatialHandshakeIndex(
            result, realizations, resources, routing))
      return std::move(error);
    return result;
  }

private:
  static llvm::Expected<std::vector<const HandshakeOwnerModel *>>
  selectActiveModels(llvm::ArrayRef<HandshakeOwnerModel> models,
                     const FrozenSpatialRealizationIndex &realizations,
                     const FrozenSpatialRoutingGraph &routing,
                     const FrozenSpatialActiveRoutingDomain &activeRouting) {
    if (activeRouting.activeTraversals().size() != routing.traversals().size())
      return invalid("active traversal domain has the wrong width");
    llvm::StringMap<bool> selectedOwners;
    for (const FrozenSpatialComputePlacement &placement :
         realizations.computePlacements())
      selectedOwners.try_emplace(
          ownerKey(FabricHandshakeOwner::fu(placement.fu)), true);
    for (const FrozenSpatialMemoryPlacement &placement :
         realizations.memoryPlacements())
      selectedOwners.try_emplace(
          ownerKey(FabricHandshakeOwner::memory(placement.memory)), true);

    llvm::StringMap<PnrIndex> traversalOrdinals;
    for (auto [ordinal, traversal] : llvm::enumerate(routing.traversals()))
      traversalOrdinals.try_emplace(refKey(traversal.reference),
                                    static_cast<PnrIndex>(ordinal));

    std::vector<const HandshakeOwnerModel *> active;
    active.reserve(models.size());
    for (const HandshakeOwnerModel &model : models) {
      bool selected = selectedOwners.contains(ownerKey(model.owner()));
      if (!selected) {
        for (std::uint32_t witnessOrdinal = 0;
             witnessOrdinal != model.traversalWitnessCount();
             ++witnessOrdinal) {
          const FabricPhysicalTraversalRef witness =
              model.traversalWitness(witnessOrdinal);
          const auto found = traversalOrdinals.find(refKey(witness));
          if (found != traversalOrdinals.end() &&
              activeRouting.traversalIsActive(found->second)) {
            selected = true;
            break;
          }
        }
      }
      if (selected)
        active.push_back(&model);
    }
    return active;
  }

  class BuildState final {
    struct PendingMemoryPlan final {
      PnrIndex model = 0;
      PnrIndex usePattern = 0;
      bool temporalResident = false;
      std::optional<std::uint32_t> issueLatencyCycles;
      std::vector<std::uint32_t> localFragments;
    };

  public:
    BuildState(FrozenSpatialHandshakeIndex &result,
               llvm::ArrayRef<const HandshakeOwnerModel *> models,
               const dataflow::CanonicalDataflowProgramView &dataflow,
               const TechMappingView &techMapping,
               const FabricArtifactView &fabric,
               const FrozenSpatialRealizationIndex &realizations,
               const FrozenSpatialResourceIndex &resources,
               const FrozenSpatialRoutingGraph &routing,
               const FrozenSpatialActiveRoutingDomain &activeRouting)
        : result_(result), models_(models), dataflow_(dataflow),
          techMapping_(techMapping), fabric_(fabric),
          realizations_(realizations), resources_(resources), routing_(routing),
          activeRouting_(activeRouting) {}

    llvm::Error prepareActiveFragments() {
      if (activeRouting_.activeTraversals().size() !=
          routing_.traversals().size())
        return invalid("active traversal domain has the wrong width");

      for (auto [modelOrdinal, model] : llvm::enumerate(models_)) {
        auto ordinal = checked(ownerIndexContext, modelOrdinal);
        if (!ordinal)
          return ordinal.takeError();
        if (!modelOrdinals_.try_emplace(ownerKey(model->owner()), *ordinal)
                 .second)
          return invalid("active handshake owner is repeated");
        result_.ownerModels_.push_back(*model);
      }
      for (auto [ordinal, traversal] : llvm::enumerate(routing_.traversals()))
        traversalOrdinals_.try_emplace(refKey(traversal.reference),
                                       static_cast<PnrIndex>(ordinal));

      activeLocalFragments_.resize(models_.size());
      if (llvm::Error error = prepareComputeSelections())
        return error;
      if (llvm::Error error = prepareMemorySelections())
        return error;

      for (auto [modelOrdinal, modelPointer] : llvm::enumerate(models_)) {
        const HandshakeOwnerModel &model = *modelPointer;
        auto &active = activeLocalFragments_[modelOrdinal];
        for (std::uint32_t fragmentOrdinal = 0;
             fragmentOrdinal != model.fragmentCount(); ++fragmentOrdinal) {
          const HandshakeActivationFragment fragment =
              model.fragment(fragmentOrdinal);
          bool retain = false;
          switch (fragment.activationKind) {
          case HandshakeActivationKind::Always:
            retain = true;
            break;
          case HandshakeActivationKind::AnyTraversal:
          case HandshakeActivationKind::AnySwitchActivationTraversal:
          case HandshakeActivationKind::ExactSwitchActivationTraversal:
            for (std::uint32_t witness = 0; witness < fragment.witnessCount;
                 ++witness) {
              auto activeWitness = traversalIsActive(
                  model.traversalWitness(fragment.witnessOffset + witness));
              if (!activeWitness)
                return activeWitness.takeError();
              if (*activeWitness) {
                retain = true;
                break;
              }
            }
            break;
          case HandshakeActivationKind::AllTraversals:
            retain = fragment.witnessCount != 0;
            for (std::uint32_t witness = 0;
                 retain && witness < fragment.witnessCount; ++witness) {
              auto activeWitness = traversalIsActive(
                  model.traversalWitness(fragment.witnessOffset + witness));
              if (!activeWitness)
                return activeWitness.takeError();
              retain = *activeWitness;
            }
            break;
          case HandshakeActivationKind::ExactOwnerSelection:
            break;
          }
          if (retain)
            active.push_back(fragmentOrdinal);
        }
        llvm::sort(active);
        active.erase(std::unique(active.begin(), active.end()), active.end());
      }
      return llvm::Error::success();
    }

    llvm::Error buildFragments() {
      modelFragmentOrdinals_.resize(models_.size());
      std::vector<std::vector<PnrIndex>> traversalFragments(
          routing_.traversals().size());
      std::vector<std::vector<PnrIndex>> traversalAllGroups(
          routing_.traversals().size());
      using SwitchActivationKey =
          std::tuple<PnrIndex, FabricOrdinal, FabricOrdinal>;
      std::map<SwitchActivationKey, std::vector<PnrIndex>>
          switchActivationBaseFragments;
      std::map<std::pair<SwitchActivationKey, PnrIndex>, std::vector<PnrIndex>>
          switchTraversalFragments;

      const auto switchDomain = [&](FabricSwitchOccurrenceRef occurrence)
          -> llvm::Expected<PnrIndex> {
        const auto domains = routing_.tagContinuity().matchDomains();
        std::optional<PnrIndex> selected;
        for (auto [ordinal, domain] : llvm::enumerate(domains)) {
          if (domain.kind !=
                  FabricPhysicalTagMatchDomainKind::TemporalSwitchTable ||
              domain.owner != FabricInventoryOwnerRef::of(occurrence))
            continue;
          if (selected)
            return invalid("Temporal switch has multiple match domains");
          selected = static_cast<PnrIndex>(ordinal);
        }
        if (!selected)
          return invalid("Temporal switch handshake has no match domain");
        return *selected;
      };

      for (auto [modelOrdinal, modelPointer] : llvm::enumerate(models_)) {
        const HandshakeOwnerModel &model = *modelPointer;
        auto &fragmentOrdinals = modelFragmentOrdinals_[modelOrdinal];
        fragmentOrdinals.assign(model.fragmentCount(), getInvalidPnrIndex());
        for (std::uint32_t localFragmentOrdinal :
             activeLocalFragments_[modelOrdinal]) {
          if (localFragmentOrdinal >= model.fragmentCount())
            return invalid("active handshake fragment is out of range");
          const HandshakeActivationFragment fragment =
              model.fragment(localFragmentOrdinal);
          auto globalFragment =
              checked(fragmentIndexContext, result_.fragments_.size());
          if (!globalFragment)
            return globalFragment.takeError();
          if (fragment.contributionOffset > model.fragmentContributionCount() ||
              fragment.contributionCount > model.fragmentContributionCount() -
                                               fragment.contributionOffset)
            return invalid("handshake fragment contribution is out of range");
          for (std::uint32_t index = 0; index < fragment.contributionCount;
               ++index) {
            const std::uint32_t localArc = model.fragmentContributionOrdinal(
                fragment.contributionOffset + index);
            if (localArc >= model.arcCount())
              return invalid("handshake fragment arc is out of range");
            const HandshakeOwnerArc arc = model.arc(localArc);
            if (arc.source >= model.nodeCount() ||
                arc.destination >= model.nodeCount())
              return invalid("handshake fragment arc endpoint is out of range");
          }
          result_.fragments_.push_back({static_cast<PnrIndex>(modelOrdinal),
                                        localFragmentOrdinal,
                                        fragment.contributionCount});
          fragmentOrdinals[localFragmentOrdinal] = *globalFragment;

          switch (fragment.activationKind) {
          case HandshakeActivationKind::Always:
            result_.fixedFragments_.push_back(*globalFragment);
            break;
          case HandshakeActivationKind::AnyTraversal:
            for (std::uint32_t witness = 0; witness < fragment.witnessCount;
                 ++witness) {
              auto traversal = traversalIndex(
                  model.traversalWitness(fragment.witnessOffset + witness));
              if (!traversal)
                return traversal.takeError();
              if (activeRouting_.traversalIsActive(*traversal))
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
                  model.traversalWitness(fragment.witnessOffset + witness));
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
          case HandshakeActivationKind::AnySwitchActivationTraversal:
          case HandshakeActivationKind::ExactSwitchActivationTraversal: {
            if (!fragment.switchActivation)
              return invalid("Temporal switch fragment has no activation key");
            auto domain = switchDomain(fragment.switchActivation->occurrence);
            if (!domain)
              return domain.takeError();
            const SwitchActivationKey key{*domain,
                                          fragment.switchActivation->row,
                                          fragment.switchActivation->input};
            if (fragment.activationKind ==
                HandshakeActivationKind::AnySwitchActivationTraversal) {
              switchActivationBaseFragments[key].push_back(*globalFragment);
              break;
            }
            if (fragment.witnessCount != 1)
              return invalid("Temporal switch crosspoint fragment does not "
                             "have one traversal witness");
            auto traversal =
                traversalIndex(model.traversalWitness(fragment.witnessOffset));
            if (!traversal)
              return traversal.takeError();
            if (!activeRouting_.traversalIsActive(*traversal))
              return invalid(
                  "inactive Temporal switch crosspoint was retained");
            switchTraversalFragments[{key, *traversal}].push_back(
                *globalFragment);
            break;
          }
          case HandshakeActivationKind::ExactOwnerSelection:
            break;
          }
        }
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
      if (llvm::Error error = flattenSlices(traversalAllGroups,
                                            result_.traversalAllGroupOffsets_,
                                            result_.traversalAllGroups_))
        return error;

      for (auto &[key, baseFragments] : switchActivationBaseFragments) {
        llvm::sort(baseFragments);
        baseFragments.erase(
            std::unique(baseFragments.begin(), baseFragments.end()),
            baseFragments.end());
        auto baseOffset =
            checked(incidenceCountContext,
                    result_.switchActivationBaseFragments_.size());
        auto baseCount = checked(incidenceCountContext, baseFragments.size());
        auto selectionOffset = checked(
            incidenceCountContext, result_.switchTraversalSelections_.size());
        if (!baseOffset)
          return baseOffset.takeError();
        if (!baseCount)
          return baseCount.takeError();
        if (!selectionOffset)
          return selectionOffset.takeError();
        result_.switchActivationBaseFragments_.insert(
            result_.switchActivationBaseFragments_.end(), baseFragments.begin(),
            baseFragments.end());
        const std::size_t selectionBegin =
            result_.switchTraversalSelections_.size();
        auto selected = switchTraversalFragments.lower_bound({key, 0});
        while (selected != switchTraversalFragments.end() &&
               selected->first.first == key) {
          auto &fragments = selected->second;
          llvm::sort(fragments);
          fragments.erase(std::unique(fragments.begin(), fragments.end()),
                          fragments.end());
          auto fragmentOffset = checked(
              incidenceCountContext, result_.switchTraversalFragments_.size());
          auto fragmentCount = checked(incidenceCountContext, fragments.size());
          if (!fragmentOffset)
            return fragmentOffset.takeError();
          if (!fragmentCount)
            return fragmentCount.takeError();
          result_.switchTraversalFragments_.insert(
              result_.switchTraversalFragments_.end(), fragments.begin(),
              fragments.end());
          result_.switchTraversalSelections_.push_back(
              {selected->first.second, *fragmentOffset, *fragmentCount});
          ++selected;
        }
        auto selectionCount =
            checked(incidenceCountContext,
                    result_.switchTraversalSelections_.size() - selectionBegin);
        if (!selectionCount)
          return selectionCount.takeError();
        result_.switchActivations_.push_back(
            {std::get<0>(key), std::get<1>(key), std::get<2>(key), *baseOffset,
             *baseCount, *selectionOffset, *selectionCount});
      }
      if (result_.switchTraversalSelections_.size() !=
          switchTraversalFragments.size())
        return invalid(
            "Temporal switch traversal fragment has no activation base");
      return llvm::Error::success();
    }

    llvm::Error prepareComputeSelections() {
      std::vector<std::vector<FabricFuOperationHandshakeBinding>> bindings;
      bindings.reserve(techMapping_.computeRealizations().size());
      for (const TechComputeRealizationView &realization :
           techMapping_.computeRealizations()) {
        std::vector<FabricFuOperationHandshakeBinding> actorBindings;
        actorBindings.reserve(realization.actors.size());
        for (const TechComputeActorView &actor : realization.actors) {
          auto resolved = dataflow_.resolve(actor.actor);
          if (!resolved)
            return resolved.takeError();
          auto projection =
              dataflow::projectRegisteredActorSchemaProjection(resolved->op);
          if (!projection)
            return projection.takeError();
          auto indexBitWidth = getIndexBitWidth(resolved->op);
          if (!indexBitWidth)
            return indexBitWidth.takeError();
          auto pointerLayout = pointerLayoutFor(dataflow_, *projection);
          if (!pointerLayout)
            return pointerLayout.takeError();
          actorBindings.push_back({actor.fabricOperation,
                                   std::move(*projection), *indexBitWidth,
                                   std::move(*pointerLayout),
                                   actor.operandPorts, actor.resultPorts});
        }
        bindings.push_back(std::move(actorBindings));
      }

      computePlacementLocalFragments_.resize(
          realizations_.computePlacements().size());
      for (auto [placementOrdinal, placement] :
           llvm::enumerate(realizations_.computePlacements())) {
        if (placement.realization >= techMapping_.computeRealizations().size())
          return invalid("compute placement realization is out of range");
        const TechComputeRealizationView &realization =
            techMapping_.computeRealizations()[placement.realization];
        auto selection = makeFuHandshakeSelection(
            fabric_, placement.fu, realization.capabilityTemplate,
            bindings[placement.realization]);
        if (!selection)
          return selection.takeError();
        auto model = modelIndex(FabricHandshakeOwner::fu(placement.fu));
        if (!model)
          return model.takeError();
        FabricHandshakeSelection exact;
        exact.fuCapabilities.push_back(std::move(*selection));
        auto activation = resolveSelectedHandshake(*models_[*model], exact);
        if (!activation)
          return activation.takeError();
        auto &placementFragments =
            computePlacementLocalFragments_[placementOrdinal];
        placementFragments.assign(activation->fragmentOrdinals().begin(),
                                  activation->fragmentOrdinals().end());
        auto &active = activeLocalFragments_[*model];
        active.insert(active.end(), placementFragments.begin(),
                      placementFragments.end());
      }
      return llvm::Error::success();
    }

    llvm::Error prepareMemorySelections() {
      llvm::StringMap<PnrIndex> usePatternOrdinals;
      for (auto [ordinal, pattern] : llvm::enumerate(resources_.usePatterns()))
        usePatternOrdinals.try_emplace(refKey(pattern.reference),
                                       static_cast<PnrIndex>(ordinal));

      result_.memoryPlacementDomainOffsets_.reserve(
          realizations_.memoryPlacements().size() + 1);
      for (auto [placementOrdinal, placement] :
           llvm::enumerate(realizations_.memoryPlacements())) {
        auto domainOffset = checked(memoryDomainOffsetContext,
                                    result_.memoryOperationDomains_.size());
        if (!domainOffset)
          return domainOffset.takeError();
        result_.memoryPlacementDomainOffsets_.push_back(*domainOffset);
        if (placement.realization >= techMapping_.memoryRealizations().size())
          return invalid("memory placement realization is out of range");
        const TechMemoryRealizationView &realization =
            techMapping_.memoryRealizations()[placement.realization];
        const FrozenSpatialMemoryRealization &frozenRealization =
            realizations_.memoryRealizations()[placement.realization];
        auto schedule = fabric_.memorySchedule(placement.memory);
        if (!schedule)
          return invalid("memory placement has no scheduling contract");
        auto model = modelIndex(FabricHandshakeOwner::memory(placement.memory));
        if (!model)
          return model.takeError();
        auto roleDemands = ::loom::mapping::deriveSpatialMemoryActorRoleDemands(
            dataflow_, techMapping_, fabric_, realization, placement.memory);
        if (!roleDemands)
          return roleDemands.takeError();

        for (auto [localActorOrdinal, actor] :
             llvm::enumerate(realization.actors)) {
          auto resolved = dataflow_.resolve(actor.actor);
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
              fabric_.memoryCapabilityAlternative(capability);
          const ::fabric::MemoryOperationPortRecord *operationPort =
              fabric_.memoryOperationPort(port);
          if (!alternative || !operationPort)
            return invalid("memory handshake capability does not resolve");
          const ::dataflow::ActorRef actorRef = actor.actor;
          const auto roleDemand = llvm::find_if(
              *roleDemands,
              [&](const ::loom::mapping::SpatialMemoryActorRoleDemandView
                      &candidate) { return candidate.actor == actorRef; });
          if (roleDemand == roleDemands->end())
            return invalid("memory handshake actor has no role demand");

          auto planOffset =
              checked(planCountContext, pendingMemoryPlans_.size());
          if (!planOffset)
            return planOffset.takeError();
          // Every resident context of one Temporal operation port exposes the
          // same capability, use patterns, and handshake fragments; only the
          // ordinal differs. Enumerating them multiplied this decision domain
          // by the context count without offering the search one distinguishing
          // fact, and left nothing that forced two actors on one port apart.
          // The domain therefore covers use patterns alone and materialization
          // derives each ordinal from the canonical order of the actors that
          // resolve to the exact port. Resolution here uses the first context
          // as the representative placement.
          const bool temporalResident =
              *schedule == ::fabric::Schedule::Temporal;
          if (temporalResident &&
              fabric_.memoryResidentContextCount(placement.memory) == 0)
            return invalid("Temporal memory has no resident context");
          const FabricMemoryHandshakePlacement operationPlacement =
              temporalResident ? FabricMemoryHandshakePlacement(
                                     FabricMemoryOperationContextRef{port, 0})
                               : FabricMemoryHandshakePlacement(port);
          {
            for (::fabric::UsePatternKey pattern :
                 alternative->admissibleUsePatterns) {
              auto issueLatency = ::fabric::projectMemoryOperationIssueLatency(
                  *operationPort, pattern);
              if (!issueLatency)
                return issueLatency.takeError();
              const FabricUsePatternRef usePattern{
                  FabricUsePatternOwnerRef(FabricInventoryOwnerRef::of(port)),
                  pattern.ordinal()};
              auto selected = makeMemoryHandshakeSelection(
                  fabric_, operationPlacement, capability, usePattern,
                  *maskForm, roleDemand->sources, roleDemand->destinations);
              if (!selected)
                return selected.takeError();
              FabricHandshakeSelection exact;
              exact.memoryOperations.push_back(std::move(*selected));
              auto activation =
                  resolveSelectedHandshake(*models_[*model], exact);
              if (!activation)
                return activation.takeError();

              auto usePatternOrdinal =
                  usePatternOrdinals.find(refKey(usePattern));
              if (usePatternOrdinal == usePatternOrdinals.end())
                return invalid(
                    "memory handshake plan has no frozen use pattern");
              PendingMemoryPlan pending;
              pending.model = *model;
              pending.usePattern = usePatternOrdinal->second;
              pending.temporalResident = temporalResident;
              pending.issueLatencyCycles = *issueLatency;
              pending.localFragments.assign(
                  activation->fragmentOrdinals().begin(),
                  activation->fragmentOrdinals().end());
              auto &active = activeLocalFragments_[*model];
              active.insert(active.end(), pending.localFragments.begin(),
                            pending.localFragments.end());
              pendingMemoryPlans_.push_back(std::move(pending));
            }
          }
          const std::size_t planCountValue =
              pendingMemoryPlans_.size() - *planOffset;
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

    llvm::Error materializeExactSelections() {
      std::vector<std::vector<PnrIndex>> placementFragments(
          computePlacementLocalFragments_.size());
      for (auto [placement, localFragments] :
           llvm::enumerate(computePlacementLocalFragments_)) {
        if (placement >= realizations_.computePlacements().size())
          return invalid("compute handshake placement is out of range");
        auto model = modelIndex(FabricHandshakeOwner::fu(
            realizations_.computePlacements()[placement].fu));
        if (!model)
          return model.takeError();
        if (llvm::Error error = appendResolvedFragments(
                *model, localFragments, placementFragments[placement]))
          return error;
      }
      if (llvm::Error error = flattenSlices(
              placementFragments, result_.computePlacementFragmentOffsets_,
              result_.computePlacementFragments_))
        return error;

      result_.memoryOperationPlans_.reserve(pendingMemoryPlans_.size());
      for (const PendingMemoryPlan &pending : pendingMemoryPlans_) {
        auto fragmentOffset =
            checked(incidenceCountContext, result_.memoryPlanFragments_.size());
        if (!fragmentOffset)
          return fragmentOffset.takeError();
        std::vector<PnrIndex> fragments;
        if (llvm::Error error = appendResolvedFragments(
                pending.model, pending.localFragments, fragments))
          return error;
        auto fragmentCount = checked(incidenceCountContext, fragments.size());
        if (!fragmentCount)
          return fragmentCount.takeError();
        result_.memoryPlanFragments_.insert(result_.memoryPlanFragments_.end(),
                                            fragments.begin(), fragments.end());
        result_.memoryOperationPlans_.push_back(
            {pending.usePattern, pending.temporalResident, *fragmentOffset,
             *fragmentCount, pending.issueLatencyCycles});
      }
      return llvm::Error::success();
    }

    llvm::Error buildDenseProjectionIndex() {
      using ProjectionArcKey = std::pair<std::string, std::string>;

      std::set<std::string> nodeKeys;
      std::map<std::string, std::optional<HandshakeSignalRef>> nodeSignals;
      std::set<ProjectionArcKey> arcKeys;
      std::vector<ProjectionArcKey> fixedArcKeys;
      std::vector<std::vector<ProjectionArcKey>> fragmentArcKeys(
          result_.fragments_.size());

      const auto retainArc = [&](detail::HandshakeArcIdentity identity,
                                 std::vector<ProjectionArcKey> &destination) {
        ProjectionArcKey key{detail::nodeKey(identity.source),
                             detail::nodeKey(identity.destination)};
        nodeKeys.insert(key.first);
        nodeKeys.insert(key.second);
        nodeSignals.try_emplace(key.first, identity.source.boundarySignal);
        nodeSignals.try_emplace(key.second,
                                identity.destination.boundarySignal);
        arcKeys.insert(key);
        destination.push_back(std::move(key));
      };

      if (!result_.fabricContext_)
        return invalid("handshake projection has no Fabric static context");
      for (const HandshakeDependencyArc &arc :
           result_.fabricContext_->unconditionalDependencyArcs()) {
        detail::HandshakeNodeIdentity source;
        source.boundarySignal = arc.source;
        detail::HandshakeNodeIdentity destination;
        destination.boundarySignal = arc.destination;
        retainArc({std::move(source), std::move(destination)}, fixedArcKeys);
      }

      for (auto [fragmentOrdinal, fragment] :
           llvm::enumerate(result_.fragments_)) {
        if (fragment.owner >= result_.ownerModels_.size())
          return invalid("handshake projection fragment owner is out of range");
        const HandshakeOwnerModel &model = result_.ownerModels_[fragment.owner];
        if (fragment.localFragment >= model.fragmentCount())
          return invalid("handshake projection local fragment is out of range");
        const HandshakeActivationFragment local =
            model.fragment(fragment.localFragment);
        if (local.contributionCount != fragment.contributionCount ||
            local.contributionOffset > model.fragmentContributionCount() ||
            local.contributionCount >
                model.fragmentContributionCount() - local.contributionOffset)
          return invalid("handshake projection fragment is stale");
        auto &retained = fragmentArcKeys[fragmentOrdinal];
        retained.reserve(local.contributionCount);
        for (std::uint32_t contribution = 0;
             contribution < local.contributionCount; ++contribution) {
          const std::uint32_t localArc = model.fragmentContributionOrdinal(
              local.contributionOffset + contribution);
          if (localArc >= model.arcCount())
            return invalid("handshake projection arc is out of range");
          auto identity =
              detail::arcIdentity(fragment.owner, model, model.arc(localArc));
          if (!identity) {
            llvm::consumeError(identity.takeError());
            return invalid("handshake projection arc endpoint is invalid");
          }
          retainArc(std::move(*identity), retained);
        }
        llvm::sort(retained);
        retained.erase(std::unique(retained.begin(), retained.end()),
                       retained.end());
      }

      llvm::sort(fixedArcKeys);
      fixedArcKeys.erase(std::unique(fixedArcKeys.begin(), fixedArcKeys.end()),
                         fixedArcKeys.end());

      std::map<std::string, PnrIndex> nodeOrdinals;
      result_.projectionNodeSignals_.reserve(nodeKeys.size());
      for (const std::string &key : nodeKeys) {
        auto ordinal = checked(projectionNodeIndexContext, nodeOrdinals.size());
        if (!ordinal)
          return ordinal.takeError();
        nodeOrdinals.emplace(key, *ordinal);
        const auto signal = nodeSignals.find(key);
        if (signal == nodeSignals.end())
          return invalid("handshake projection node has no retained signal");
        result_.projectionNodeSignals_.push_back(signal->second);
      }
      auto nodeCount = checked(projectionNodeIndexContext, nodeOrdinals.size());
      if (!nodeCount)
        return nodeCount.takeError();
      result_.projectionNodeCount_ = *nodeCount;

      std::map<ProjectionArcKey, PnrIndex> arcOrdinals;
      result_.projectionArcs_.reserve(arcKeys.size());
      for (const ProjectionArcKey &key : arcKeys) {
        const auto source = nodeOrdinals.find(key.first);
        const auto destination = nodeOrdinals.find(key.second);
        if (source == nodeOrdinals.end() || destination == nodeOrdinals.end())
          return invalid("handshake projection arc has no canonical node");
        auto ordinal =
            checked(projectionArcIndexContext, result_.projectionArcs_.size());
        if (!ordinal)
          return ordinal.takeError();
        result_.projectionArcs_.push_back(
            {source->second, destination->second});
        arcOrdinals.emplace(key, *ordinal);
      }

      result_.projectionFixedArcs_.reserve(fixedArcKeys.size());
      for (const ProjectionArcKey &key : fixedArcKeys) {
        const auto arc = arcOrdinals.find(key);
        if (arc == arcOrdinals.end())
          return invalid("fixed handshake projection arc is absent");
        result_.projectionFixedArcs_.push_back(arc->second);
      }

      std::vector<std::vector<PnrIndex>> fragmentArcs(fragmentArcKeys.size());
      for (auto [fragmentOrdinal, keys] : llvm::enumerate(fragmentArcKeys)) {
        auto &retained = fragmentArcs[fragmentOrdinal];
        retained.reserve(keys.size());
        for (const ProjectionArcKey &key : keys) {
          const auto arc = arcOrdinals.find(key);
          if (arc == arcOrdinals.end())
            return invalid("fragment handshake projection arc is absent");
          retained.push_back(arc->second);
        }
      }
      if (llvm::Error error =
              flattenSlices(fragmentArcs, result_.projectionFragmentArcOffsets_,
                            result_.projectionFragmentArcs_))
        return error;

      result_.projectionOutgoingArcOffsets_.clear();
      result_.projectionOutgoingArcOffsets_.reserve(
          static_cast<std::size_t>(result_.projectionNodeCount_) + 1);
      std::size_t arcCursor = 0;
      for (PnrIndex node = 0; node < result_.projectionNodeCount_; ++node) {
        auto offset = checked(projectionArcOffsetContext, arcCursor);
        if (!offset)
          return offset.takeError();
        result_.projectionOutgoingArcOffsets_.push_back(*offset);
        while (arcCursor < result_.projectionArcs_.size() &&
               result_.projectionArcs_[arcCursor].source == node)
          ++arcCursor;
      }
      auto end = checked(projectionArcOffsetContext, arcCursor);
      if (!end)
        return end.takeError();
      result_.projectionOutgoingArcOffsets_.push_back(*end);
      if (arcCursor != result_.projectionArcs_.size())
        return invalid("handshake projection arcs are not source-major");
      return llvm::Error::success();
    }

  private:
    llvm::Expected<bool>
    traversalIsActive(const FabricPhysicalTraversalRef &reference) const {
      auto traversal = traversalIndex(reference);
      if (!traversal)
        return traversal.takeError();
      return activeRouting_.traversalIsActive(*traversal);
    }

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

    llvm::Error
    appendResolvedFragments(PnrIndex model,
                            llvm::ArrayRef<std::uint32_t> localFragments,
                            std::vector<PnrIndex> &destination) const {
      if (model >= modelFragmentOrdinals_.size())
        return invalid("resolved handshake owner is out of range");
      for (std::uint32_t fragment : localFragments) {
        if (fragment >= modelFragmentOrdinals_[model].size() ||
            modelFragmentOrdinals_[model][fragment] == getInvalidPnrIndex())
          return invalid("resolved handshake fragment was not retained");
        destination.push_back(modelFragmentOrdinals_[model][fragment]);
      }
      llvm::sort(destination);
      destination.erase(std::unique(destination.begin(), destination.end()),
                        destination.end());
      return llvm::Error::success();
    }

    FrozenSpatialHandshakeIndex &result_;
    llvm::ArrayRef<const HandshakeOwnerModel *> models_;
    const dataflow::CanonicalDataflowProgramView &dataflow_;
    const TechMappingView &techMapping_;
    const FabricArtifactView &fabric_;
    const FrozenSpatialRealizationIndex &realizations_;
    const FrozenSpatialResourceIndex &resources_;
    const FrozenSpatialRoutingGraph &routing_;
    const FrozenSpatialActiveRoutingDomain &activeRouting_;
    llvm::StringMap<PnrIndex> traversalOrdinals_;
    llvm::StringMap<PnrIndex> modelOrdinals_;
    std::vector<std::vector<std::uint32_t>> activeLocalFragments_;
    std::vector<std::vector<std::uint32_t>> computePlacementLocalFragments_;
    std::vector<PendingMemoryPlan> pendingMemoryPlans_;
    std::vector<std::vector<PnrIndex>> modelFragmentOrdinals_;
  };
};

llvm::Expected<FrozenSpatialHandshakeIndex>
loom::pnr::detail::buildFrozenSpatialHandshakeIndex(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const FabricHandshakeContext &handshakeContext,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing,
    const FrozenSpatialActiveRoutingDomain &activeRouting) {
  return FrozenSpatialHandshakeIndexBuilder::build(
      dataflow, techMapping, fabric, handshakeContext, realizations,
      resources, routing, activeRouting);
}

llvm::Error loom::pnr::detail::verifyFrozenSpatialHandshakeIndex(
    const FrozenSpatialHandshakeIndex &handshake,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing) {
  (void)resources;
  if (!handshake.fabricContext())
    return invalid("handshake index has no Fabric static context");
  const auto models = handshake.ownerModels();
  llvm::StringMap<bool> owners;
  for (const HandshakeOwnerModel &model : models)
    if (!owners.try_emplace(ownerKey(model.owner()), true).second)
      return invalid("handshake owner model is repeated");
  for (const FrozenSpatialHandshakeFragment &fragment : handshake.fragments()) {
    if (fragment.owner >= models.size())
      return invalid("handshake fragment owner is out of range");
    const HandshakeOwnerModel &model = models[fragment.owner];
    if (fragment.localFragment >= model.fragmentCount())
      return invalid("local handshake fragment is out of range");
    const HandshakeActivationFragment local =
        model.fragment(fragment.localFragment);
    if (local.contributionCount != fragment.contributionCount ||
        local.contributionOffset > model.fragmentContributionCount() ||
        local.contributionCount >
            model.fragmentContributionCount() - local.contributionOffset)
      return invalid("handshake fragment contribution is inconsistent");
    for (std::uint32_t index = 0; index < local.contributionCount; ++index) {
      const std::uint32_t arcOrdinal =
          model.fragmentContributionOrdinal(local.contributionOffset + index);
      if (arcOrdinal >= model.arcCount())
        return invalid("handshake fragment arc is out of range");
      const HandshakeOwnerArc arc = model.arc(arcOrdinal);
      if (arc.source >= model.nodeCount() ||
          arc.destination >= model.nodeCount())
        return invalid("handshake fragment arc endpoint is out of range");
    }
  }
  for (PnrIndex fragment : handshake.fixedFragments())
    if (fragment >= handshake.fragments().size())
      return invalid("fixed handshake fragment is out of range");

  const auto projectionArcs = handshake.projectionArcs();
  const auto outgoingOffsets = handshake.projectionOutgoingArcOffsets();
  if (outgoingOffsets.size() !=
          static_cast<std::size_t>(handshake.projectionNodeCount()) + 1 ||
      outgoingOffsets.empty() || outgoingOffsets.front() != 0 ||
      outgoingOffsets.back() != projectionArcs.size())
    return invalid("handshake projection outgoing CSR is incomplete");
  for (auto [ordinal, arc] : llvm::enumerate(projectionArcs)) {
    if (arc.source >= handshake.projectionNodeCount() ||
        arc.destination >= handshake.projectionNodeCount())
      return invalid("handshake projection arc endpoint is out of range");
    if (ordinal != 0) {
      const FrozenSpatialHandshakeArc previous = projectionArcs[ordinal - 1];
      if (std::tie(previous.source, previous.destination) >=
          std::tie(arc.source, arc.destination))
        return invalid("handshake projection arcs are noncanonical");
    }
  }
  for (PnrIndex node = 0; node < handshake.projectionNodeCount(); ++node) {
    const PnrIndex begin = outgoingOffsets[node];
    const PnrIndex end = outgoingOffsets[node + 1];
    if (begin > end || end > projectionArcs.size())
      return invalid("handshake projection outgoing range is invalid");
    for (PnrIndex arc = begin; arc < end; ++arc)
      if (projectionArcs[arc].source != node)
        return invalid("handshake projection outgoing range is stale");
  }

  std::vector<std::uint8_t> referencedProjectionArcs(projectionArcs.size(), 0);
  const auto verifyArcOrdinals = [&](llvm::ArrayRef<PnrIndex> ordinals,
                                     llvm::StringRef subject) -> llvm::Error {
    PnrIndex previous = 0;
    bool hasPrevious = false;
    for (PnrIndex arc : ordinals) {
      if (arc >= projectionArcs.size())
        return invalid(subject + " arc is out of range");
      if (hasPrevious && arc <= previous)
        return invalid(subject + " arcs are noncanonical");
      previous = arc;
      hasPrevious = true;
      referencedProjectionArcs[arc] = 1;
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = verifyArcOrdinals(handshake.projectionFixedArcs(),
                                            "fixed handshake projection"))
    return error;

  const auto fragmentArcOffsets = handshake.projectionFragmentArcOffsets();
  const auto fragmentArcs = handshake.projectionFragmentArcs();
  if (fragmentArcOffsets.size() != handshake.fragments().size() + 1 ||
      fragmentArcOffsets.empty() || fragmentArcOffsets.front() != 0 ||
      fragmentArcOffsets.back() != fragmentArcs.size())
    return invalid("handshake projection fragment CSR is incomplete");
  for (PnrIndex fragment = 0; fragment < handshake.fragments().size();
       ++fragment) {
    const PnrIndex begin = fragmentArcOffsets[fragment];
    const PnrIndex end = fragmentArcOffsets[fragment + 1];
    if (begin > end || end > fragmentArcs.size())
      return invalid("handshake projection fragment range is invalid");
    if (llvm::Error error =
            verifyArcOrdinals(fragmentArcs.slice(begin, end - begin),
                              "fragment handshake projection"))
      return error;
  }
  if (llvm::find(referencedProjectionArcs, 0) != referencedProjectionArcs.end())
    return invalid("handshake projection contains an unowned arc");

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
  for (auto [ordinal, activation] :
       llvm::enumerate(handshake.switchActivations())) {
    const auto domains = routing.tagContinuity().matchDomains();
    if (activation.matchDomain >= domains.size() ||
        domains[activation.matchDomain].kind !=
            FabricPhysicalTagMatchDomainKind::TemporalSwitchTable ||
        activation.baseFragmentCount == 0 ||
        !rangeFits(activation.baseFragmentOffset, activation.baseFragmentCount,
                   handshake.switchActivationBaseFragments().size()) ||
        !rangeFits(activation.traversalSelectionOffset,
                   activation.traversalSelectionCount,
                   handshake.switchTraversalSelections().size()))
      return invalid("Temporal switch handshake activation is inconsistent");
    if (ordinal != 0) {
      const auto &previous = handshake.switchActivations()[ordinal - 1];
      if (std::tie(previous.matchDomain, previous.row, previous.input) >=
          std::tie(activation.matchDomain, activation.row, activation.input))
        return invalid(
            "Temporal switch handshake activations are noncanonical");
    }
    for (PnrIndex fragment : handshake.switchActivationBaseFragments().slice(
             activation.baseFragmentOffset, activation.baseFragmentCount))
      if (fragment >= handshake.fragments().size())
        return invalid("Temporal switch activation fragment is out of range");
    PnrIndex previousTraversal = getInvalidPnrIndex();
    for (const FrozenSpatialSwitchHandshakeTraversalSelection &selection :
         handshake.switchTraversalSelections().slice(
             activation.traversalSelectionOffset,
             activation.traversalSelectionCount)) {
      if (selection.traversal >= routing.traversals().size() ||
          !rangeFits(selection.fragmentOffset, selection.fragmentCount,
                     handshake.switchTraversalFragments().size()) ||
          selection.fragmentCount == 0 ||
          (previousTraversal != getInvalidPnrIndex() &&
           selection.traversal <= previousTraversal))
        return invalid("Temporal switch traversal selection is inconsistent");
      previousTraversal = selection.traversal;
      for (PnrIndex fragment : handshake.switchTraversalFragments().slice(
               selection.fragmentOffset, selection.fragmentCount))
        if (fragment >= handshake.fragments().size())
          return invalid("Temporal switch traversal fragment is out of range");
    }
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
