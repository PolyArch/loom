#include "SpatialBindingRelationModel.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
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
         projection == Projection::MemoryPlacement;
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

llvm::Expected<PnrIndex> relationDecision(
    Projection projection, const SpatialConstraintSubject &subject,
    const llvm::DenseMap<std::uint64_t, PnrIndex> &computeDecisions,
    const llvm::DenseMap<std::uint64_t, PnrIndex> &memoryDecisions) {
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
  llvm_unreachable("deferred projection requested a binding decision");
}

} // namespace

llvm::Expected<std::shared_ptr<const SpatialBindingRelationModel>>
SpatialBindingRelationModel::create(
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenConstraintIndex &constraints) {
  std::vector<PnrIndex> computeChoiceOffsets;
  std::vector<SpatialComputeBindingChoice> computeChoices;
  std::vector<PnrIndex> computeContextChoiceOrdinals(
      realizations.computeInstructionContexts().size(), getInvalidPnrIndex());
  std::vector<PnrIndex> memoryChoiceOffsets;
  std::vector<SpatialMemoryBindingChoice> memoryChoices;
  std::vector<PnrIndex> memoryPlacementChoiceOrdinals(
      realizations.memoryPlacements().size(), getInvalidPnrIndex());
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
  const PnrIndex memoryDecisionOffset =
      static_cast<PnrIndex>(realizations.computeRealizations().size());
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

  std::vector<InitializerRelationInput> relationInputs;
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
          auto decision =
              relationDecision(*projection, shard.subjects()[subjectOrdinal],
                               computeDecisions, memoryDecisions);
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
          } else {
            const PnrIndex realization = *decision - memoryDecisionOffset;
            const auto choices =
                llvm::ArrayRef(memoryChoices)
                    .slice(memoryChoiceOffsets[realization],
                           memoryChoiceOffsets[realization + 1] -
                               memoryChoiceOffsets[realization]);
            keys.reserve(choices.size());
            for (const SpatialMemoryBindingChoice &choice : choices)
              keys.push_back(memoryProjectionKey(realizations, choice));
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

  auto relations = InitializerRelationModel::create(
      std::move(decisionChoiceCounts), std::move(relationInputs));
  if (!relations)
    return relations.takeError();
  return std::shared_ptr<const SpatialBindingRelationModel>(
      new SpatialBindingRelationModel(
          std::move(*relations), std::move(computeChoiceOffsets),
          std::move(computeChoices), std::move(computeContextChoiceOrdinals),
          std::move(memoryChoiceOffsets), std::move(memoryChoices),
          std::move(memoryPlacementChoiceOrdinals), deferredProjection));
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
SpatialBindingRelationModel::decisionRelations(PnrIndex decision) const {
  assert(decision < decisionCount());
  const auto offsets = relations_.decisionRelationOffsets();
  return relations_.decisionRelations().slice(
      offsets[decision], offsets[decision + 1] - offsets[decision]);
}
