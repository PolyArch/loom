#ifndef LOOM_PNR_SYSTEM_SYSTEMACTIONDOMAIN_H
#define LOOM_PNR_SYSTEM_SYSTEMACTIONDOMAIN_H

#include "PnR/System/SystemAction.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr {

class SystemCandidateState;

class SystemActionDomainScratch final {
public:
  llvm::Error rebuild(const SystemCandidateState &candidate);

  SystemActionProposalDomain view() const;
  std::uint64_t movableDecisionCount() const;
  std::uint64_t selectableMovableDecisionCount(
      const ResolvedPnrActionProposalPolicy &policy) const;

private:
  std::vector<SystemActionChoiceRange> bindingAnchors_;
  std::vector<SystemExecutionBindingAction> bindingChoices_;
  std::vector<SystemActionChoiceRange> routingAnchors_;
  std::vector<SystemTransportRoutingAction> routingChoices_;
  std::vector<SystemActionChoiceRange> resourceAnchors_;
  std::vector<SystemResourceAllocationAction> resourceChoices_;
  std::uint64_t bindingMovableDecisionCount_ = 0;
  std::uint64_t routingMovableDecisionCount_ = 0;
  std::uint64_t resourceMovableDecisionCount_ = 0;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMACTIONDOMAIN_H
