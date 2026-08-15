#include "SpatialPnrResourceIndex.h"

#include "PnR/PnrIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

using namespace loom;
using namespace loom::fabric;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialPnrProblem";

PnrCapacityContext capacityContext(llvm::StringLiteral table,
                                   llvm::StringLiteral domain,
                                   PnrCapacityMeasure measure) {
  return PnrCapacityContext{frozenArtifact, table, domain, measure};
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext capacity,
                                 std::size_t value) {
  return checkedPnrIndex(capacity, static_cast<std::uint64_t>(value));
}

llvm::Error appendChecked(PnrCapacityContext capacity, std::size_t current,
                          std::uint64_t added) {
  auto end = checkedPnrIndexAdd(capacity, current, added);
  if (!end)
    return end.takeError();
  return llvm::Error::success();
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

} // namespace

class loom::pnr::FrozenSpatialResourceIndexBuilder final {
public:
  static llvm::Expected<FrozenSpatialResourceIndex>
  build(const FabricArtifactView &fabric) {
    FrozenSpatialResourceIndex result;
    const auto owners = fabric.moduleResourceOwners();
    if (llvm::Error error = preflightPnrIndexCapacity(
            capacityContext("resource_owners", "resource_owners",
                            PnrCapacityMeasure::Count),
            owners.size()))
      return std::move(error);
    result.owners_.reserve(owners.size());

    for (const FabricInventoryOwnerRef &owner : owners) {
      const ::fabric::ResourceContract *contract =
          fabric.resourceContract(owner);
      if (!contract)
        return invalid("physical resource owner has no ResourceContract");

      auto stateOffset =
          checked(capacityContext("resource_owners", "resource_states",
                                  PnrCapacityMeasure::Offset),
                  result.states_.size());
      if (!stateOffset)
        return stateOffset.takeError();
      if (llvm::Error error = appendChecked(
              capacityContext("resource_states", "resource_states",
                              PnrCapacityMeasure::Count),
              result.states_.size(), contract->stateCount()))
        return std::move(error);
      for (std::uint32_t stateOrdinal = 0;
           stateOrdinal < contract->stateCount(); ++stateOrdinal) {
        auto capacityOffset =
            checked(capacityContext("resource_states", "capacity_dimensions",
                                    PnrCapacityMeasure::Offset),
                    result.capacityDimensions_.size());
        if (!capacityOffset)
          return capacityOffset.takeError();
        const auto dimensions =
            contract->capacityDimensions(::fabric::StateKey(stateOrdinal));
        if (llvm::Error error = appendChecked(
                capacityContext("capacity_dimensions", "capacity_dimensions",
                                PnrCapacityMeasure::Count),
                result.capacityDimensions_.size(), dimensions.size()))
          return std::move(error);
        for (const ::fabric::CapacityDimension &dimension : dimensions)
          result.capacityDimensions_.push_back(
              {dimension.capacity.value(), dimension.initialOccupancy.value()});
        auto capacityCount = checked(capacityContext("capacity_dimensions",
                                                     "capacity_dimensions",
                                                     PnrCapacityMeasure::Count),
                                     dimensions.size());
        if (!capacityCount)
          return capacityCount.takeError();
        result.states_.push_back(
            {FabricResourceStateRef{FabricResourceStateOwnerRef(owner),
                                    stateOrdinal},
             *capacityOffset, *capacityCount});
      }
      auto stateCount =
          checked(capacityContext("resource_states", "resource_states",
                                  PnrCapacityMeasure::Count),
                  contract->stateCount());
      if (!stateCount)
        return stateCount.takeError();

      auto timingOffset =
          checked(capacityContext("resource_owners", "timing_contracts",
                                  PnrCapacityMeasure::Offset),
                  result.timingContracts_.size());
      if (!timingOffset)
        return timingOffset.takeError();
      if (llvm::Error error = appendChecked(
              capacityContext("timing_contracts", "timing_contracts",
                              PnrCapacityMeasure::Count),
              result.timingContracts_.size(), contract->timingContractCount()))
        return std::move(error);
      for (std::uint32_t timingOrdinal = 0;
           timingOrdinal < contract->timingContractCount(); ++timingOrdinal) {
        auto eventRankOffset =
            checked(capacityContext("timing_contracts", "event_ranks",
                                    PnrCapacityMeasure::Offset),
                    result.eventRanks_.size());
        if (!eventRankOffset)
          return eventRankOffset.takeError();
        const auto ranks =
            contract->eventOrder(::fabric::TimingContractKey(timingOrdinal));
        if (llvm::Error error =
                appendChecked(capacityContext("event_ranks", "event_ranks",
                                              PnrCapacityMeasure::Count),
                              result.eventRanks_.size(), ranks.size()))
          return std::move(error);
        result.eventRanks_.insert(result.eventRanks_.end(), ranks.begin(),
                                  ranks.end());
        auto eventRankCount =
            checked(capacityContext("event_ranks", "event_ranks",
                                    PnrCapacityMeasure::Count),
                    ranks.size());
        if (!eventRankCount)
          return eventRankCount.takeError();
        result.timingContracts_.push_back({*eventRankOffset, *eventRankCount});
      }
      auto timingCount =
          checked(capacityContext("timing_contracts", "timing_contracts",
                                  PnrCapacityMeasure::Count),
                  contract->timingContractCount());
      if (!timingCount)
        return timingCount.takeError();

      auto patternOffset =
          checked(capacityContext("resource_owners", "use_patterns",
                                  PnrCapacityMeasure::Offset),
                  result.patterns_.size());
      if (!patternOffset)
        return patternOffset.takeError();
      if (llvm::Error error = appendChecked(
              capacityContext("use_patterns", "use_patterns",
                              PnrCapacityMeasure::Count),
              result.patterns_.size(), contract->usePatternCount()))
        return std::move(error);
      for (std::uint32_t patternOrdinal = 0;
           patternOrdinal < contract->usePatternCount(); ++patternOrdinal) {
        const ::fabric::UsePattern pattern =
            contract->usePattern(::fabric::UsePatternKey(patternOrdinal));
        auto claimOffset =
            checked(capacityContext("use_patterns", "resource_claims",
                                    PnrCapacityMeasure::Offset),
                    result.claims_.size());
        if (!claimOffset)
          return claimOffset.takeError();
        if (llvm::Error error = appendChecked(
                capacityContext("resource_claims", "resource_claims",
                                PnrCapacityMeasure::Count),
                result.claims_.size(), pattern.claims.size()))
          return std::move(error);
        for (const ::fabric::Claim &claim : pattern.claims) {
          if (claim.state.ordinal() >= contract->stateCount())
            return invalid("use pattern names an out-of-range resource state");
          auto state = checkedPnrIndexAdd(
              capacityContext("resource_claims", "resource_states",
                              PnrCapacityMeasure::Index),
              *stateOffset, claim.state.ordinal());
          if (!state)
            return state.takeError();
          result.claims_.push_back(
              {*state, claim.dimension.ordinal(), claim.amount.value()});
        }
        auto claimCount =
            checked(capacityContext("resource_claims", "resource_claims",
                                    PnrCapacityMeasure::Count),
                    pattern.claims.size());
        if (!claimCount)
          return claimCount.takeError();

        auto transactionOffset =
            checked(capacityContext("use_patterns", "internal_transactions",
                                    PnrCapacityMeasure::Offset),
                    result.internalTransactions_.size());
        if (!transactionOffset)
          return transactionOffset.takeError();
        for (std::uint32_t transaction = 0;
             transaction < pattern.internalTransactionCount; ++transaction) {
          auto transactionClaimOffset = checked(
              capacityContext("internal_transactions", "transaction_claims",
                              PnrCapacityMeasure::Offset),
              result.transactionClaims_.size());
          if (!transactionClaimOffset)
            return transactionClaimOffset.takeError();
          const auto claims = contract->internalTransaction(
              ::fabric::UsePatternKey(patternOrdinal), transaction);
          if (llvm::Error error = appendChecked(
                  capacityContext("transaction_claims", "transaction_claims",
                                  PnrCapacityMeasure::Count),
                  result.transactionClaims_.size(), claims.size()))
            return std::move(error);
          for (::fabric::ClaimKey claim : claims) {
            if (claim.ordinal() >= pattern.claims.size())
              return invalid("internal transaction selects an invalid claim");
            auto globalClaim = checkedPnrIndexAdd(
                capacityContext("transaction_claims", "resource_claims",
                                PnrCapacityMeasure::Index),
                *claimOffset, claim.ordinal());
            if (!globalClaim)
              return globalClaim.takeError();
            result.transactionClaims_.push_back(*globalClaim);
          }
          auto transactionClaimCount = checked(
              capacityContext("transaction_claims", "transaction_claims",
                              PnrCapacityMeasure::Count),
              claims.size());
          if (!transactionClaimCount)
            return transactionClaimCount.takeError();
          result.internalTransactions_.push_back(
              {*transactionClaimOffset, *transactionClaimCount});
        }
        auto transactionCount = checked(
            capacityContext("internal_transactions", "internal_transactions",
                            PnrCapacityMeasure::Count),
            pattern.internalTransactionCount);
        if (!transactionCount)
          return transactionCount.takeError();

        std::optional<FrozenSpatialResourceCommit> commit;
        if (pattern.commit)
          commit =
              FrozenSpatialResourceCommit{pattern.commit->event.ordinal(),
                                          pattern.commit->transition.ordinal()};
        auto timing = checkedPnrIndexAdd(
            capacityContext("use_patterns", "timing_contracts",
                            PnrCapacityMeasure::Index),
            *timingOffset, pattern.timingAndProgress.ordinal());
        if (!timing)
          return timing.takeError();
        const ::fabric::UsePatternTiming intrinsicTiming =
            contract->usePatternTiming(
                ::fabric::UsePatternKey(patternOrdinal));
        result.patterns_.push_back(
            {FabricUsePatternRef{FabricUsePatternOwnerRef(owner),
                                 patternOrdinal},
             pattern.requester.ordinal(), pattern.eligibility.ordinal(),
             pattern.acquire.ordinal(), pattern.release.ordinal(), commit,
             *timing, intrinsicTiming.releaseLatencyCycles,
             intrinsicTiming.minimumInitiationIntervalCycles, *claimOffset,
             *claimCount, *transactionOffset, *transactionCount});
      }
      auto patternCount =
          checked(capacityContext("use_patterns", "use_patterns",
                                  PnrCapacityMeasure::Count),
                  contract->usePatternCount());
      if (!patternCount)
        return patternCount.takeError();

      auto grantOrderOffset =
          checked(capacityContext("resource_owners", "grant_requester_order",
                                  PnrCapacityMeasure::Offset),
                  result.grantRequesterOrder_.size());
      if (!grantOrderOffset)
        return grantOrderOffset.takeError();
      FrozenSpatialGrantPolicyKind grantPolicy =
          FrozenSpatialGrantPolicyKind::None;
      std::optional<std::uint32_t> roundRobinResetRequester;
      if (const auto policy = contract->grantPolicy())
        std::visit(
            [&](const auto &typed) {
              using Policy = std::decay_t<decltype(typed)>;
              if constexpr (std::is_same_v<Policy,
                                           ::fabric::FixedPriorityView>) {
                grantPolicy = FrozenSpatialGrantPolicyKind::FixedPriority;
                for (::fabric::RequesterKey requester : typed.requesterOrder())
                  result.grantRequesterOrder_.push_back(requester.ordinal());
              } else {
                grantPolicy = FrozenSpatialGrantPolicyKind::RoundRobin;
                for (::fabric::RequesterKey requester : typed.requesterCycle())
                  result.grantRequesterOrder_.push_back(requester.ordinal());
                roundRobinResetRequester = typed.resetCursor().ordinal();
              }
            },
            *policy);
      const std::size_t grantOrderSize =
          result.grantRequesterOrder_.size() - *grantOrderOffset;
      auto grantOrderCount = checked(capacityContext("grant_requester_order",
                                                     "grant_requester_order",
                                                     PnrCapacityMeasure::Count),
                                     grantOrderSize);
      if (!grantOrderCount)
        return grantOrderCount.takeError();

      result.owners_.push_back(
          {owner, *stateOffset, *stateCount, *patternOffset, *patternCount,
           *timingOffset, *timingCount, *grantOrderOffset, *grantOrderCount,
           grantPolicy, roundRobinResetRequester,
           contract->resourceTransitionCount(), contract->requesterCount(),
           contract->eligibilityCount(), contract->eventCount()});
    }
    return result;
  }
};

llvm::Expected<FrozenSpatialResourceIndex>
loom::pnr::detail::buildFrozenSpatialResourceIndex(
    const FabricArtifactView &fabric) {
  return FrozenSpatialResourceIndexBuilder::build(fabric);
}
