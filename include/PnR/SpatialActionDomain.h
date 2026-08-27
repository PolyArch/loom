#ifndef LOOM_PNR_SPATIALACTIONDOMAIN_H
#define LOOM_PNR_SPATIALACTIONDOMAIN_H

#include "PnR/SpatialAction.h"
#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace loom::pnr {

namespace detail {
class SpatialMemoryConstraintScratch;
}

class FrozenSpatialPnrProblem;
class SpatialCandidateState;

/// Worker-local storage for the exact dynamic Action domain of one candidate.
/// Frozen problem tables own every choice; this removable projection retains
/// capacity across proposals and performs no allocation after prepare().
class SpatialActionDomainScratch final {
public:
  SpatialActionDomainScratch();
  ~SpatialActionDomainScratch();

  llvm::Error prepare(const FrozenSpatialPnrProblem &problem);
  llvm::Error rebuild(const SpatialCandidateState &candidate);

  SpatialActionProposalDomain view() const;
  llvm::Expected<std::optional<SpatialMappingAction>>
  propose(const ResolvedPnrActionProposalPolicy &policy,
          DeterministicPnrRandomStream &proposalStream) const;
  std::uint64_t movableDecisionCount() const;
  std::uint64_t selectableMovableDecisionCount(
      const ResolvedPnrActionProposalPolicy &policy) const;
  std::uint64_t examinedRealizationChoiceCount() const {
    return examinedRealizationChoiceCount_;
  }
  std::uint64_t fixedRelationPrunedRealizationChoiceCount() const {
    return fixedRelationPrunedRealizationChoiceCount_;
  }
  std::size_t retainedStorageBytes() const;

private:
  struct RelationDecisionMember final {
    std::size_t choiceValueOrdinalOffset = 0;
    std::uint64_t demand = 1;
  };

  std::vector<SpatialActionChoiceRange> realizationAnchors_;
  std::vector<SpatialRealizationBindingAction> realizationChoices_;
  std::vector<SpatialActionChoiceRange> transportAnchors_;
  std::vector<SpatialTransportRoutingAction> transportChoices_;
  std::vector<PnrIndex> routeRootEndpoints_;
  std::vector<PnrIndex> routeSubtreeSlots_;
  std::vector<std::uint8_t> routeSubtreeHasSink_;
  std::vector<PnrIndex> hardProgressWitnessOwners_;
  std::vector<SpatialActionChoiceRange> resourceAnchors_;
  std::vector<SpatialResourceAllocationAction> resourceChoices_;
  std::vector<PnrIndex> relationChoices_;
  std::vector<std::size_t> relationValueOffsets_;
  std::vector<PnrIndex> relationValues_;
  std::vector<std::uint64_t> relationValueLoads_;
  std::vector<PnrIndex> relationDistinctValueCounts_;
  std::vector<std::uint8_t> rootClosedRelations_;
  std::vector<std::size_t> relationDecisionMemberOffsets_;
  std::vector<RelationDecisionMember> relationDecisionMembers_;
  std::vector<PnrIndex> relationDecisionMemberChoiceValueOrdinals_;
  std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryChoices_;
  std::unique_ptr<detail::SpatialMemoryConstraintScratch>
      memoryConstraintScratch_;
  std::uint64_t realizationMovableDecisionCount_ = 0;
  std::uint64_t transportMovableDecisionCount_ = 0;
  std::uint64_t resourceMovableDecisionCount_ = 0;
  std::uint64_t examinedRealizationChoiceCount_ = 0;
  std::uint64_t fixedRelationPrunedRealizationChoiceCount_ = 0;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALACTIONDOMAIN_H
