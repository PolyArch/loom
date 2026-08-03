#include "SpatialPnrCapacityIndex.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialPnrProblem";

PnrCapacityContext capacityContext(llvm::StringLiteral table,
                                   llvm::StringLiteral domain,
                                   PnrCapacityMeasure measure) {
  return PnrCapacityContext{frozenArtifact, table, domain, measure};
}

llvm::Expected<PnrIndex> checkedIndex(llvm::StringLiteral table,
                                      llvm::StringLiteral domain,
                                      PnrCapacityMeasure measure,
                                      std::size_t value) {
  return checkedPnrIndex(capacityContext(table, domain, measure), value);
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

std::string refKey(const FabricUsePatternRef &reference) {
  const std::vector<std::uint8_t> bytes = canonicalFabricBytes(reference);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Expected<std::string> eventKey(const ArtifactIdentity &dataflowIdentity,
                                     PnrIndex realization,
                                     const SpatialActivityEventRef &event) {
  std::string result;
  auto appendU64 = [&](std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte)
      result.push_back(static_cast<char>(value >> (8 * (7 - byte))));
  };
  appendU64(realization);
  auto encoded = encodeSpatialActivityEventKey(dataflowIdentity, event);
  if (!encoded)
    return encoded.takeError();
  appendU64(encoded->size());
  result.append(reinterpret_cast<const char *>(encoded->data()),
                encoded->size());
  return result;
}

llvm::Expected<PnrIndex>
patternOrdinal(const llvm::StringMap<PnrIndex> &patternByRef,
               const FabricUsePatternRef &reference) {
  const auto found = patternByRef.find(refKey(reference));
  if (found == patternByRef.end())
    return invalid("selected ResourceUse names absent Fabric UsePattern " +
                   printFabricRef(reference));
  return found->second;
}

llvm::Expected<std::uint64_t>
atomicEnvelopeOveruse(const FrozenSpatialResourceIndex &resources,
                      llvm::ArrayRef<PnrIndex> patterns) {
  llvm::SmallDenseMap<PnrIndex, std::uint64_t, 8> demand;
  for (PnrIndex patternOrdinal : patterns) {
    if (patternOrdinal >= resources.usePatterns().size())
      return invalid("atomic capacity envelope contains an invalid pattern");
    const FrozenSpatialUsePattern &pattern =
        resources.usePatterns()[patternOrdinal];
    for (const FrozenSpatialResourceClaim &claim :
         resources.claims().slice(pattern.claimOffset, pattern.claimCount)) {
      if (claim.state >= resources.resourceStates().size())
        return invalid("atomic capacity envelope contains an invalid state");
      const FrozenSpatialResourceState &state =
          resources.resourceStates()[claim.state];
      if (claim.dimension >= state.capacityCount)
        return invalid(
            "atomic capacity envelope contains an invalid dimension");
      const PnrIndex dimension = state.capacityOffset + claim.dimension;
      std::uint64_t &amount = demand[dimension];
      if (claim.amount > std::numeric_limits<std::uint64_t>::max() - amount)
        return invalid("atomic capacity demand overflows u64");
      amount += claim.amount;
    }
  }

  std::uint64_t total = 0;
  for (const auto &entry : demand) {
    if (entry.first >= resources.capacityDimensions().size())
      return invalid("atomic capacity envelope resolved a foreign dimension");
    const FrozenSpatialCapacityDimension &dimension =
        resources.capacityDimensions()[entry.first];
    if (entry.second >
        std::numeric_limits<std::uint64_t>::max() - dimension.initialOccupancy)
      return invalid("atomic capacity usage overflows u64");
    const std::uint64_t usage = dimension.initialOccupancy + entry.second;
    const std::uint64_t overuse =
        usage > dimension.capacity ? usage - dimension.capacity : 0;
    if (overuse > std::numeric_limits<std::uint64_t>::max() - total)
      return invalid("atomic capacity overuse overflows u64");
    total += overuse;
  }
  return total;
}

struct BoundaryChange final {
  PnrIndex dimension = 0;
  std::uint64_t rank = 0;
  std::uint64_t added = 0;
  std::uint64_t removed = 0;
};

llvm::Error checkedAccumulate(std::uint64_t value, std::uint64_t &total,
                              llvm::StringRef subject);

llvm::Expected<std::uint64_t>
appendTimedEnvelope(const FrozenSpatialResourceIndex &resources,
                    llvm::ArrayRef<PnrIndex> patterns,
                    std::vector<FrozenSpatialResourceTimeSegment> &segments) {
  llvm::SmallVector<BoundaryChange, 16> changes;
  for (PnrIndex patternOrdinal : patterns) {
    if (patternOrdinal >= resources.usePatterns().size())
      return invalid("resource-time envelope contains an invalid pattern");
    const FrozenSpatialUsePattern &pattern =
        resources.usePatterns()[patternOrdinal];
    if (pattern.timingContract >= resources.timingContracts().size())
      return invalid("resource-time envelope has an invalid timing contract");
    const FrozenSpatialTimingContract &timing =
        resources.timingContracts()[pattern.timingContract];
    if (pattern.acquireEvent >= timing.eventRankCount ||
        pattern.releaseEvent >= timing.eventRankCount)
      return invalid("resource-time envelope has an invalid event rank");
    const auto ranks = resources.eventRanks().slice(timing.eventRankOffset,
                                                    timing.eventRankCount);
    const std::uint64_t begin = ranks[pattern.acquireEvent];
    const std::uint64_t release = ranks[pattern.releaseEvent];
    const std::uint64_t end = release > begin ? release : begin + 1;
    for (const FrozenSpatialResourceClaim &claim :
         resources.claims().slice(pattern.claimOffset, pattern.claimCount)) {
      if (claim.state >= resources.resourceStates().size())
        return invalid("resource-time envelope contains an invalid state");
      const FrozenSpatialResourceState &state =
          resources.resourceStates()[claim.state];
      if (claim.dimension >= state.capacityCount)
        return invalid("resource-time envelope contains an invalid dimension");
      const PnrIndex dimension = state.capacityOffset + claim.dimension;
      changes.push_back({dimension, begin, claim.amount, 0});
      changes.push_back({dimension, end, 0, claim.amount});
    }
  }

  llvm::sort(changes, [](const BoundaryChange &lhs, const BoundaryChange &rhs) {
    return std::tie(lhs.dimension, lhs.rank) <
           std::tie(rhs.dimension, rhs.rank);
  });
  std::uint64_t totalOveruse = 0;
  for (std::size_t dimensionBegin = 0; dimensionBegin < changes.size();) {
    const PnrIndex dimension = changes[dimensionBegin].dimension;
    if (dimension >= resources.capacityDimensions().size())
      return invalid("resource-time envelope resolved a foreign dimension");
    const FrozenSpatialCapacityDimension &capacity =
        resources.capacityDimensions()[dimension];
    std::uint64_t usage = capacity.initialOccupancy;
    std::uint64_t maximumOveruse = 0;
    std::size_t cursor = dimensionBegin;
    while (cursor < changes.size() && changes[cursor].dimension == dimension) {
      const std::uint64_t rank = changes[cursor].rank;
      std::uint64_t added = 0;
      std::uint64_t removed = 0;
      while (cursor < changes.size() &&
             changes[cursor].dimension == dimension &&
             changes[cursor].rank == rank) {
        if (llvm::Error error = checkedAccumulate(changes[cursor].added, added,
                                                  "resource-time addition"))
          return std::move(error);
        if (llvm::Error error = checkedAccumulate(
                changes[cursor].removed, removed, "resource-time removal"))
          return std::move(error);
        ++cursor;
      }
      if (removed > usage)
        return invalid("resource-time removal exceeds current usage");
      usage -= removed;
      if (added > std::numeric_limits<std::uint64_t>::max() - usage)
        return invalid("resource-time usage overflows u64");
      usage += added;
      const std::uint64_t nextRank =
          cursor < changes.size() && changes[cursor].dimension == dimension
              ? changes[cursor].rank
              : rank;
      if (nextRank > rank && usage != capacity.initialOccupancy) {
        const std::uint64_t overuse =
            usage > capacity.capacity ? usage - capacity.capacity : 0;
        segments.push_back({dimension, rank, nextRank, usage, overuse});
        maximumOveruse = std::max(maximumOveruse, overuse);
      }
    }
    if (usage != capacity.initialOccupancy)
      return invalid("resource-time envelope does not release every claim");
    if (llvm::Error error = checkedAccumulate(maximumOveruse, totalOveruse,
                                              "resource-time capacity overuse"))
      return std::move(error);
    dimensionBegin = cursor;
  }
  return totalOveruse;
}

llvm::Error checkedAccumulate(std::uint64_t value, std::uint64_t &total,
                              llvm::StringRef subject) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total)
    return invalid(subject + " overflows u64");
  total += value;
  return llvm::Error::success();
}

} // namespace

class loom::pnr::FrozenSpatialCapacityIndexBuilder final {
public:
  static llvm::Expected<FrozenSpatialCapacityIndex>
  build(const dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping, const FabricArtifactView &fabric,
        const FrozenSpatialRealizationIndex &realizations,
        const FrozenSpatialResourceIndex &resources,
        const FrozenSpatialRoutingGraph &routing,
        const FrozenSpatialHandshakeIndex &handshake) {
    llvm::StringMap<PnrIndex> patternByRef;
    for (auto [ordinal, pattern] : llvm::enumerate(resources.usePatterns())) {
      const auto inserted = patternByRef.try_emplace(
          refKey(pattern.reference), static_cast<PnrIndex>(ordinal));
      if (!inserted.second)
        return invalid("frozen Fabric UsePattern inventory is not unique");
    }

    FrozenSpatialCapacityIndex result;
    llvm::StringMap<PnrIndex> eventByKey;
    const auto contexts = realizations.computeInstructionContexts();
    if (contexts.size() >= std::numeric_limits<PnrIndex>::max())
      return invalid("compute context envelope-offset domain exceeds PnrIndex");
    result.computeInstructionContextEnvelopeOffsets_.assign(
        contexts.size() + 1, getInvalidPnrIndex());
    result.computeInstructionContextEnvelopeOffsets_.front() = 0;
    result.computeInstructionContextOveruse_.assign(contexts.size(), 0);
    for (auto [realizationOrdinal, frozenRealization] :
         llvm::enumerate(realizations.computeRealizations())) {
      if (realizationOrdinal >= techMapping.computeRealizations().size())
        return invalid("compute capacity owner is absent from TechMapping");
      const TechComputeRealizationView &realization =
          techMapping.computeRealizations()[realizationOrdinal];

      for (const FrozenSpatialComputePlacement &placement :
           realizations.computePlacements().slice(
               frozenRealization.placementOffset,
               frozenRealization.placementCount)) {
        for (PnrIndex contextOrdinal = placement.contextOffset;
             contextOrdinal != placement.contextOffset + placement.contextCount;
             ++contextOrdinal) {
          const InstructionContextRef context = contexts[contextOrdinal];
          auto envelopeOffset = checkedIndex(
              "compute_context_envelope_offsets", "resource_time_envelopes",
              PnrCapacityMeasure::Offset, result.envelopes_.size());
          if (!envelopeOffset)
            return envelopeOffset.takeError();
          PnrIndex &storedOffset =
              result.computeInstructionContextEnvelopeOffsets_[contextOrdinal];
          if (storedOffset != getInvalidPnrIndex() &&
              storedOffset != *envelopeOffset)
            return invalid("compute context envelope offsets are not ordered");
          storedOffset = *envelopeOffset;
          std::uint64_t placementOveruse = 0;

          const SpatialComputeBindingView selected{
              realization.entityId, placement.fu, context, {}};
          auto requirements = deriveSpatialComputeBindingUseRequirements(
              dataflow, realization, fabric, selected);
          if (!requirements)
            return requirements.takeError();
          for (std::size_t begin = 0; begin < requirements->size();) {
            std::size_t end = begin + 1;
            while (end < requirements->size() &&
                   (*requirements)[end].trigger ==
                       (*requirements)[begin].trigger)
              ++end;
            auto key = eventKey(dataflow.identity(),
                                static_cast<PnrIndex>(realizationOrdinal),
                                (*requirements)[begin].trigger);
            if (!key)
              return key.takeError();
            PnrIndex eventOrdinal = 0;
            auto found = eventByKey.find(*key);
            if (found == eventByKey.end()) {
              auto checkedEvent = checkedIndex(
                  "resource_events", "resource_events",
                  PnrCapacityMeasure::Index, result.events_.size());
              if (!checkedEvent)
                return checkedEvent.takeError();
              eventOrdinal = *checkedEvent;
              eventByKey.try_emplace(*key, eventOrdinal);
              result.events_.push_back(
                  {static_cast<PnrIndex>(realizationOrdinal),
                   (*requirements)[begin].trigger});
            } else {
              eventOrdinal = found->second;
            }

            auto useOffset =
                checkedIndex("resource_time_envelopes", "resource_uses",
                             PnrCapacityMeasure::Offset, result.uses_.size());
            if (!useOffset)
              return useOffset.takeError();
            llvm::SmallVector<PnrIndex, 8> patterns;
            patterns.reserve(end - begin);
            for (std::size_t use = begin; use < end; ++use) {
              auto dense =
                  patternOrdinal(patternByRef, (*requirements)[use].pattern);
              if (!dense)
                return dense.takeError();
              patterns.push_back(*dense);
              result.uses_.push_back({eventOrdinal, *dense});
            }
            auto useCount =
                checkedIndex("resource_time_envelopes", "resource_uses",
                             PnrCapacityMeasure::Count, end - begin);
            if (!useCount)
              return useCount.takeError();
            auto segmentOffset = checkedIndex(
                "resource_time_envelopes", "resource_time_segments",
                PnrCapacityMeasure::Offset, result.segments_.size());
            if (!segmentOffset)
              return segmentOffset.takeError();
            auto overuse =
                appendTimedEnvelope(resources, patterns, result.segments_);
            if (!overuse)
              return overuse.takeError();
            auto segmentCount = checkedIndex(
                "resource_time_envelopes", "resource_time_segments",
                PnrCapacityMeasure::Count,
                result.segments_.size() - *segmentOffset);
            if (!segmentCount)
              return segmentCount.takeError();
            result.envelopes_.push_back({eventOrdinal, *useOffset, *useCount,
                                         *segmentOffset, *segmentCount,
                                         *overuse});
            if (llvm::Error error =
                    checkedAccumulate(*overuse, placementOveruse,
                                      "compute atomic capacity overuse"))
              return std::move(error);
            begin = end;
          }
          result.computeInstructionContextOveruse_[contextOrdinal] =
              placementOveruse;
          auto envelopeEnd = checkedIndex(
              "compute_context_envelope_offsets", "resource_time_envelopes",
              PnrCapacityMeasure::Offset, result.envelopes_.size());
          if (!envelopeEnd)
            return envelopeEnd.takeError();
          PnrIndex &storedEnd =
              result.computeInstructionContextEnvelopeOffsets_[contextOrdinal +
                                                               1];
          if (storedEnd != getInvalidPnrIndex() && storedEnd != *envelopeEnd)
            return invalid("compute context envelope offsets are not ordered");
          storedEnd = *envelopeEnd;
        }
      }
    }
    const auto missing = llvm::find(
        result.computeInstructionContextEnvelopeOffsets_, getInvalidPnrIndex());
    if (missing != result.computeInstructionContextEnvelopeOffsets_.end())
      return invalid(
          "compute context envelope projection is incomplete at offset " +
          llvm::Twine(std::distance(
              result.computeInstructionContextEnvelopeOffsets_.begin(),
              missing)));

    result.memoryOperationPlanOveruse_.reserve(
        handshake.memoryOperationPlans().size());
    for (const FrozenSpatialMemoryOperationHandshakePlan &plan :
         handshake.memoryOperationPlans()) {
      const PnrIndex selected[] = {plan.usePattern};
      auto overuse = atomicEnvelopeOveruse(resources, selected);
      if (!overuse)
        return overuse.takeError();
      result.memoryOperationPlanOveruse_.push_back(*overuse);
    }

    // A traversal activation group is one owner-normalized physical use. It
    // must be individually feasible, while contention between independent
    // message events remains negotiated routing or ResourceTimeOverbooking.
    for (const FrozenSpatialRouteClaim &claim : routing.routeClaims()) {
      if (claim.capacityDimension >= resources.capacityDimensions().size())
        return invalid("route claim names an invalid capacity dimension");
      const FrozenSpatialCapacityDimension &dimension =
          resources.capacityDimensions()[claim.capacityDimension];
      const std::uint64_t usage =
          static_cast<std::uint64_t>(dimension.initialOccupancy) + claim.amount;
      if (usage > dimension.capacity)
        return invalid(
            "one traversal activation group exceeds Fabric capacity");
    }
    return result;
  }
};

llvm::Expected<FrozenSpatialCapacityIndex>
loom::pnr::detail::buildFrozenSpatialCapacityIndex(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing,
    const FrozenSpatialHandshakeIndex &handshake) {
  return FrozenSpatialCapacityIndexBuilder::build(dataflow, techMapping, fabric,
                                                  realizations, resources,
                                                  routing, handshake);
}
