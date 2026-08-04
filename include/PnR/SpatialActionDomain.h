#ifndef LOOM_PNR_SPATIALACTIONDOMAIN_H
#define LOOM_PNR_SPATIALACTIONDOMAIN_H

#include "PnR/SpatialAction.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace loom::pnr {

class FrozenSpatialPnrProblem;
class SpatialCandidateState;

/// Worker-local storage for the exact dynamic Action domain of one candidate.
/// Frozen problem tables own every choice; this removable projection retains
/// capacity across proposals and performs no allocation after prepare().
class SpatialActionDomainScratch final {
public:
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
  std::vector<SpatialActionChoiceRange> resourceAnchors_;
  std::vector<SpatialResourceAllocationAction> resourceChoices_;
  std::vector<PnrIndex> relationChoices_;
  std::uint64_t movableDecisionCount_ = 0;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALACTIONDOMAIN_H
