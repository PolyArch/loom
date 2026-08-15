#include "SpatialCandidateLocalTransferPreference.h"

#include "SpatialBindingRelationModel.h"
#include "SpatialRouteConstraintModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <limits>
#include <system_error>

using namespace loom::pnr;

namespace {

llvm::Error preferenceError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial local-transfer preference: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

} // namespace

llvm::Expected<detail::SpatialCandidateLocalTransferPreference>
detail::SpatialCandidateLocalTransferPreference::create(
    const FrozenSpatialPnrProblem &problem) {
  SpatialCandidateLocalTransferPreference result(problem);
  const PnrIndex computeCount =
      problem.bindingRelations().computeDecisionCount();
  const auto domains = problem.localTransfers().domains();
  const auto options = problem.localTransfers().options();
  if (domains.size() != problem.transfers().logicalNets().size())
    return preferenceError("local-transfer domains do not cover logical nets");

  result.logicalNetsByRealization_.resize(computeCount);
  for (PnrIndex logicalNet = 0; logicalNet < domains.size(); ++logicalNet) {
    if (problem.routeConstraints().netHasConstraints(logicalNet))
      continue;
    const FrozenSpatialRegisterFifoTransferDomain &domain = domains[logicalNet];
    if (domain.optionOffset > options.size() ||
        domain.optionCount > options.size() - domain.optionOffset)
      return preferenceError("a local-transfer domain is out of range");
    if (domain.optionCount == 0)
      continue;

    const auto domainOptions =
        options.slice(domain.optionOffset, domain.optionCount);
    const auto &first = domainOptions.front();
    if (first.logicalNet != logicalNet ||
        first.producerRealization >= computeCount ||
        first.consumerRealization >= computeCount)
      return preferenceError("a local-transfer option has a foreign owner");
    if (llvm::any_of(domainOptions, [&](const auto &option) {
          return option.logicalNet != logicalNet ||
                 option.producerRealization != first.producerRealization ||
                 option.consumerRealization != first.consumerRealization;
        }))
      return preferenceError("one local-transfer domain changes its owners");

    result.logicalNetsByRealization_[first.producerRealization].push_back(
        logicalNet);
    if (first.consumerRealization != first.producerRealization)
      result.logicalNetsByRealization_[first.consumerRealization].push_back(
          logicalNet);
  }
  return result;
}

llvm::Expected<detail::SpatialCandidateLocalTransferScores>
detail::SpatialCandidateLocalTransferPreference::scoreChoices(
    PnrIndex realization, llvm::ArrayRef<PnrIndex> choicePlacements,
    llvm::ArrayRef<PnrIndex> selectedChoiceOrdinals) const {
  SpatialCandidateLocalTransferScores result;
  const auto &bindings = problem_->bindingRelations();
  const PnrIndex computeCount = bindings.computeDecisionCount();
  if (realization >= computeCount ||
      realization >= logicalNetsByRealization_.size() ||
      choicePlacements.size() != bindings.computeChoices(realization).size() ||
      selectedChoiceOrdinals.size() < computeCount)
    return preferenceError("compute choice dimensions are inconsistent");
  result.matchedNets.assign(choicePlacements.size(), 0);
  result.unmatchedNets.assign(choicePlacements.size(), 0);

  const auto domains = problem_->localTransfers().domains();
  const auto options = problem_->localTransfers().options();
  for (PnrIndex logicalNet : logicalNetsByRealization_[realization]) {
    if (logicalNet >= domains.size())
      return preferenceError("a realization names a foreign logical net");
    const auto &domain = domains[logicalNet];
    if (domain.optionCount == 0 || domain.optionOffset > options.size() ||
        domain.optionCount > options.size() - domain.optionOffset)
      return preferenceError("an active local-transfer domain is malformed");
    const auto domainOptions =
        options.slice(domain.optionOffset, domain.optionCount);
    const auto &first = domainOptions.front();

    const PnrIndex peer = first.producerRealization == realization
                              ? first.consumerRealization
                              : first.producerRealization;
    PnrIndex peerPlacement = getInvalidPnrIndex();
    if (peer != realization) {
      if (peer >= computeCount)
        return preferenceError("a local-transfer peer is out of range");
      const PnrIndex selected = selectedChoiceOrdinals[peer];
      if (selected == getInvalidPnrIndex())
        continue;
      const auto peerChoices = bindings.computeChoices(peer);
      if (selected >= peerChoices.size())
        return preferenceError("a local-transfer peer choice is out of range");
      peerPlacement = peerChoices[selected].placement;
    }

    if (result.activeNets == std::numeric_limits<std::uint64_t>::max())
      return preferenceError("active local-transfer count exceeds u64");
    ++result.activeNets;
    for (std::size_t local = 0; local < choicePlacements.size(); ++local) {
      const PnrIndex candidatePlacement = choicePlacements[local];
      const bool matched = llvm::any_of(domainOptions, [&](const auto &option) {
        const PnrIndex producerPlacement =
            option.producerRealization == realization ? candidatePlacement
                                                      : peerPlacement;
        const PnrIndex consumerPlacement =
            option.consumerRealization == realization ? candidatePlacement
                                                      : peerPlacement;
        return option.producerPlacement == producerPlacement &&
               option.consumerPlacement == consumerPlacement;
      });
      std::uint64_t &count =
          matched ? result.matchedNets[local] : result.unmatchedNets[local];
      if (count == std::numeric_limits<std::uint64_t>::max())
        return preferenceError("local-transfer score exceeds u64");
      ++count;
    }
  }
  return result;
}
