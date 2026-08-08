#include "PnR/System/SystemActionDomain.h"

#include "SystemCandidateMutation.h"

#include "PnR/PnrIndex.h"
#include "PnR/System/SystemCandidateState.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <system_error>

using namespace loom;
using namespace loom::pnr;

namespace {

llvm::Error invalid(llvm::StringRef detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_action_domain_invalid: %s", detail.str().c_str());
}

llvm::Expected<PnrIndex> actionIndex(std::size_t value, llvm::StringRef table,
                                     PnrCapacityMeasure measure) {
  return checkedPnrIndex({"SystemActionDomain", table, "Action", measure},
                         value);
}

template <typename Action>
llvm::Error appendRange(std::size_t offset, const std::vector<Action> &choices,
                        std::vector<SystemActionChoiceRange> &anchors,
                        std::uint64_t &movableDecisionCount,
                        llvm::StringRef table) {
  if (choices.size() == offset)
    return llvm::Error::success();
  auto checkedOffset = actionIndex(offset, table, PnrCapacityMeasure::Offset);
  if (!checkedOffset)
    return checkedOffset.takeError();
  auto checkedCount =
      actionIndex(choices.size() - offset, table, PnrCapacityMeasure::Count);
  if (!checkedCount)
    return checkedCount.takeError();
  anchors.push_back({*checkedOffset, *checkedCount});
  if (movableDecisionCount == std::numeric_limits<std::uint64_t>::max())
    return invalid("movable decision count overflows u64");
  ++movableDecisionCount;
  return llvm::Error::success();
}

} // namespace

llvm::Error
SystemActionDomainScratch::rebuild(const SystemCandidateState &candidate) {
  bindingAnchors_.clear();
  bindingChoices_.clear();
  routingAnchors_.clear();
  routingChoices_.clear();
  resourceAnchors_.clear();
  resourceChoices_.clear();
  movableDecisionCount_ = 0;

  const FrozenSystemPnrProblem &problem = candidate.problem();
  for (PnrIndex decision = 0; decision < problem.threadDecisions().size();
       ++decision) {
    const std::size_t offset = bindingChoices_.size();
    for (PnrIndex choice = 0;
         choice < problem.threadChoiceCatalogOrdinals(decision).size();
         ++choice)
      if (choice != candidate.threadChoice(decision))
        bindingChoices_.push_back({decision, choice});
    if (llvm::Error error =
            appendRange(offset, bindingChoices_, bindingAnchors_,
                        movableDecisionCount_, "bindingChoices"))
      return error;
  }
  const PnrIndex threadCount = problem.threadDecisions().size();
  for (PnrIndex decision = 0; decision < problem.graphDecisions().size();
       ++decision) {
    const std::size_t offset = bindingChoices_.size();
    for (PnrIndex choice = 0;
         choice < problem.graphChoiceCatalogOrdinals(decision).size(); ++choice)
      if (choice != candidate.graphChoice(decision))
        bindingChoices_.push_back({threadCount + decision, choice});
    if (llvm::Error error =
            appendRange(offset, bindingChoices_, bindingAnchors_,
                        movableDecisionCount_, "bindingChoices"))
      return error;
  }

  for (const SystemServiceRouteSelection &route : candidate.serviceRoutes()) {
    const std::size_t offset = routingChoices_.size();
    const auto nodes =
        candidate.serviceRouteNodes().slice(route.nodeOffset, route.nodeCount);
    if (llvm::any_of(nodes, [](const SystemServiceRouteNodeSelection &node) {
          return node.incomingTraversal != getInvalidPnrIndex();
        }))
      routingChoices_.emplace_back(SystemWholeLegRoutingAction{route.leg});
    if (llvm::Error error =
            appendRange(offset, routingChoices_, routingAnchors_,
                        movableDecisionCount_, "routingChoices"))
      return error;
  }

  for (PnrIndex context = 0; context < candidate.serviceTargets().size();
       ++context) {
    const std::size_t offset = resourceChoices_.size();
    auto choices = detail::systemServiceTargetChoices(candidate, context);
    if (!choices)
      return choices.takeError();
    for (PnrIndex choice = 0; choice < choices->size(); ++choice)
      if (!((*choices)[choice] == candidate.serviceTarget(context)))
        resourceChoices_.emplace_back(
            SystemServiceTargetAction{context, choice});
    if (llvm::Error error =
            appendRange(offset, resourceChoices_, resourceAnchors_,
                        movableDecisionCount_, "resourceChoices"))
      return error;
  }
  for (PnrIndex use = 0; use < candidate.instructionResourceUses().size();
       ++use) {
    const std::size_t offset = resourceChoices_.size();
    auto choices = detail::systemInstructionUsePatternChoices(candidate, use);
    if (!choices)
      return choices.takeError();
    for (PnrIndex choice = 0; choice < choices->size(); ++choice)
      if ((*choices)[choice] !=
          candidate.instructionResourceUses()[use].pattern)
        resourceChoices_.emplace_back(
            SystemInstructionUsePatternAction{use, choice});
    if (llvm::Error error =
            appendRange(offset, resourceChoices_, resourceAnchors_,
                        movableDecisionCount_, "resourceChoices"))
      return error;
  }
  for (PnrIndex use = 0; use < candidate.serviceResourceUses().size(); ++use) {
    const std::size_t offset = resourceChoices_.size();
    auto choices = detail::systemServiceUsePatternChoices(candidate, use);
    if (!choices)
      return choices.takeError();
    for (PnrIndex choice = 0; choice < choices->size(); ++choice)
      if ((*choices)[choice] != candidate.serviceResourceUses()[use].pattern)
        resourceChoices_.emplace_back(
            SystemServiceUsePatternAction{use, choice});
    if (llvm::Error error =
            appendRange(offset, resourceChoices_, resourceAnchors_,
                        movableDecisionCount_, "resourceChoices"))
      return error;
  }
  return llvm::Error::success();
}

SystemActionProposalDomain SystemActionDomainScratch::view() const {
  return {bindingAnchors_, bindingChoices_,  routingAnchors_,
          routingChoices_, resourceAnchors_, resourceChoices_};
}
