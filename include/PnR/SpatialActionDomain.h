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
  std::uint64_t movableDecisionCount() const { return movableDecisionCount_; }
  std::size_t retainedStorageBytes() const;

private:
  std::vector<SpatialActionChoiceRange> realizationAnchors_;
  std::vector<SpatialRealizationBindingAction> realizationChoices_;
  std::vector<SpatialActionChoiceRange> transportAnchors_;
  std::vector<SpatialTransportRoutingAction> transportChoices_;
  std::vector<PnrIndex> routeRootEndpoints_;
  std::vector<PnrIndex> routeSubtreeSlots_;
  std::vector<std::uint8_t> routeSubtreeHasSink_;
  std::vector<SpatialActionChoiceRange> resourceAnchors_;
  std::vector<SpatialResourceAllocationAction> resourceChoices_;
  std::vector<PnrIndex> relationChoices_;
  std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryChoices_;
  std::unique_ptr<detail::SpatialMemoryConstraintScratch>
      memoryConstraintScratch_;
  std::uint64_t movableDecisionCount_ = 0;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALACTIONDOMAIN_H
