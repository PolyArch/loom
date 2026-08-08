#include "PnR/System/SystemAction.h"

using namespace loom::pnr;

llvm::Expected<std::optional<SystemExecutionBindingAction>>
loom::pnr::proposeSystemAction(const ResolvedPnrActionProposalPolicy &policy,
                               SystemActionProposalDomain domain,
                               DeterministicPnrRandomStream &stream) {
  if (llvm::Error error = validateResolvedPnrActionProposalPolicy(policy))
    return std::move(error);
  if (domain.bindingChoices.empty() || policy.realizationBindingWeight == 0)
    return std::optional<SystemExecutionBindingAction>();
  auto selected = stream.nextBounded(domain.bindingChoices.size());
  if (!selected)
    return selected.takeError();
  return std::optional<SystemExecutionBindingAction>(
      domain.bindingChoices[*selected]);
}
