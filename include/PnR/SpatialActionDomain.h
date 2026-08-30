#ifndef LOOM_PNR_SPATIALACTIONDOMAIN_H
#define LOOM_PNR_SPATIALACTIONDOMAIN_H

#include "PnR/SpatialAction.h"
#include "PnR/SpatialCandidateState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
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
  /// Rebuilds only the projection state one committed Action can invalidate:
  /// the relation loads and realization segments the changed decisions'
  /// relation values touch, the transport segments of the touched logical
  /// nets, and the wholesale witness and resource tails. Unaffected segments
  /// are copied forward, so the arrays equal a full rebuild exactly. Falls
  /// back to rebuild() when a change is outside the modeled dependency cone.
  llvm::Error applyCommitted(
      const SpatialCandidateState &candidate,
      llvm::ArrayRef<std::pair<SpatialCandidateScratch::DecisionKind, PnrIndex>>
          changedDecisions,
      llvm::ArrayRef<PnrIndex> touchedLogicalNets);

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

  llvm::Error rebuildRelationLoads(const SpatialCandidateState &candidate);
  bool relationChoiceIsLegal(PnrIndex decision, PnrIndex localChoice,
                             bool constraintsOnly) const;
  llvm::Error appendRealizationRange(std::size_t offset);
  llvm::Error appendTransportRange(std::size_t offset);
  llvm::Error appendResourceRange(std::size_t offset);
  llvm::Error emitRealizationSegment(const SpatialCandidateState &candidate,
                                     PnrIndex decision);
  llvm::Error emitTransportNetSegment(const SpatialCandidateState &candidate,
                                      PnrIndex logicalNet);
  llvm::Error emitTransportWitnessTail(const SpatialCandidateState &candidate,
                                       std::uint64_t externalNetCount);
  llvm::Error rebuildResourceSection(const SpatialCandidateState &candidate);

  std::vector<SpatialActionChoiceRange> realizationAnchors_;
  std::vector<SpatialRealizationBindingAction> realizationChoices_;
  std::vector<SpatialActionChoiceRange> transportAnchors_;
  std::vector<SpatialTransportRoutingAction> transportChoices_;
  std::vector<PnrIndex> routeRootEndpoints_;
  std::vector<PnrIndex> routeSubtreeSlots_;
  std::vector<std::uint8_t> routeSubtreeHasSink_;
  std::vector<PnrIndex> progressShortfallWitnessOwners_;
  std::vector<PnrIndex> progressDebtWitnessOwners_;
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
  /// Copy-forward buffers and per-decision/per-net segment locations that let
  /// applyCommitted() reuse unaffected segments byte for byte.
  std::vector<SpatialRealizationBindingAction> previousRealizationChoices_;
  std::vector<SpatialTransportRoutingAction> previousTransportChoices_;
  std::vector<SpatialActionChoiceRange> realizationSegments_;
  std::vector<SpatialActionChoiceRange> previousRealizationSegments_;
  std::vector<SpatialActionChoiceRange> transportNetSegments_;
  std::vector<SpatialActionChoiceRange> previousTransportNetSegments_;
  /// Global relation-value ordinal to realization decisions with a choice on
  /// that value under a constraint or root-closed relation.
  std::vector<std::size_t> valueRealizationOffsets_;
  std::vector<PnrIndex> valueRealizationDecisions_;
  /// Relation to distinct realization member decisions, for equal-kind
  /// relations whose legality is relation-wide.
  std::vector<std::size_t> relationRealizationOffsets_;
  std::vector<PnrIndex> relationRealizationDecisions_;
  std::vector<std::uint64_t> realizationAffectedMarks_;
  std::vector<std::pair<PnrIndex, PnrIndex>> touchedRelationValues_;
  std::vector<PnrIndex> touchedEqualRelations_;
  std::vector<PnrIndex> sortedTouchedNets_;
  std::uint64_t affectedEpoch_ = 0;
  bool segmentsValid_ = false;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALACTIONDOMAIN_H
