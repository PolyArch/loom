#include "FabricHandshakeInternal.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_handshake_invalid: " + message);
}

template <typename T>
llvm::Expected<std::uint32_t> checkedSize(const T &container,
                                          llvm::StringRef description) {
  if (container.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid(description + " exceeds the owner-model index domain");
  return static_cast<std::uint32_t>(container.size());
}

std::vector<std::uint8_t>
junctionKey(llvm::ArrayRef<std::uint8_t> ownerLocalKey) {
  std::vector<std::uint8_t> key;
  key.reserve(ownerLocalKey.size() + 1);
  key.push_back(1);
  key.insert(key.end(), ownerLocalKey.begin(), ownerLocalKey.end());
  return key;
}

} // namespace

std::vector<std::uint8_t> handshakeSignalKey(const HandshakeSignalRef &signal) {
  std::vector<std::uint8_t> key = canonicalFabricBytes(signal.endpoint);
  key.insert(key.begin(), static_cast<std::uint8_t>(signal.signal));
  key.insert(key.begin(), 0);
  return key;
}

HandshakeOwnerModelBuilder::HandshakeOwnerModelBuilder(
    FabricHandshakeOwner owner)
    : model_(std::move(owner)) {}

std::uint32_t
HandshakeOwnerModelBuilder::boundarySignal(HandshakeSignalRef signal) {
  const std::vector<std::uint8_t> key = handshakeSignalKey(signal);
  auto found = nodes_.find(key);
  if (found != nodes_.end())
    return found->second;
  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(model_.nodes_.size());
  model_.nodes_.push_back(
      {HandshakeOwnerNodeKind::BoundarySignal, std::move(signal)});
  nodes_.emplace(key, ordinal);
  return ordinal;
}

std::uint32_t HandshakeOwnerModelBuilder::junction(
    llvm::ArrayRef<std::uint8_t> ownerLocalKey) {
  const std::vector<std::uint8_t> key = junctionKey(ownerLocalKey);
  auto found = nodes_.find(key);
  if (found != nodes_.end())
    return found->second;
  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(model_.nodes_.size());
  model_.nodes_.push_back(
      {HandshakeOwnerNodeKind::OwnerLocalJunction, std::nullopt});
  nodes_.emplace(key, ordinal);
  return ordinal;
}

void HandshakeOwnerModelBuilder::addFragment(
    HandshakeFragmentSelector selector,
    std::vector<std::pair<std::uint32_t, std::uint32_t>> arcs) {
  pending_.push_back({std::move(selector), std::move(arcs)});
}

llvm::Expected<HandshakeOwnerModel> HandshakeOwnerModelBuilder::finish() {
  if (auto count = checkedSize(model_.nodes_, "handshake node count"); !count)
    return count.takeError();

  std::set<std::pair<std::uint32_t, std::uint32_t>> uniqueArcs;
  for (const PendingFragment &fragment : pending_)
    uniqueArcs.insert(fragment.arcs.begin(), fragment.arcs.end());
  if (uniqueArcs.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("handshake arc count exceeds the owner-model index domain");

  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> arcOrdinals;
  model_.arcs_.reserve(uniqueArcs.size());
  for (const auto &arc : uniqueArcs) {
    const std::uint32_t ordinal =
        static_cast<std::uint32_t>(model_.arcs_.size());
    arcOrdinals.emplace(arc, ordinal);
    model_.arcs_.push_back({arc.first, arc.second});
  }

  model_.fragments_.reserve(pending_.size());
  model_.fragmentSelectors_.reserve(pending_.size());
  for (PendingFragment &pending : pending_) {
    std::vector<std::uint32_t> contributions;
    contributions.reserve(pending.arcs.size());
    for (const auto &arc : pending.arcs)
      contributions.push_back(arcOrdinals.at(arc));
    llvm::sort(contributions);
    contributions.erase(std::unique(contributions.begin(), contributions.end()),
                        contributions.end());
    auto offset = checkedSize(model_.fragmentContributionOrdinals_,
                              "handshake contribution offset");
    auto count = checkedSize(contributions, "handshake contribution count");
    if (!offset)
      return offset.takeError();
    if (!count)
      return count.takeError();

    HandshakeActivationKind activationKind =
        HandshakeActivationKind::ExactOwnerSelection;
    switch (pending.selector.kind) {
    case HandshakeFragmentSelectorKind::Always:
      activationKind = HandshakeActivationKind::Always;
      break;
    case HandshakeFragmentSelectorKind::AnyTraversal:
      activationKind = HandshakeActivationKind::AnyTraversal;
      break;
    case HandshakeFragmentSelectorKind::AllTraversals:
      activationKind = HandshakeActivationKind::AllTraversals;
      break;
    case HandshakeFragmentSelectorKind::AnySwitchActivationTraversal:
      activationKind = HandshakeActivationKind::AnySwitchActivationTraversal;
      break;
    case HandshakeFragmentSelectorKind::ExactSwitchActivationTraversal:
      activationKind = HandshakeActivationKind::ExactSwitchActivationTraversal;
      break;
    case HandshakeFragmentSelectorKind::FuCapability:
    case HandshakeFragmentSelectorKind::FuOperationCase:
    case HandshakeFragmentSelectorKind::FuOperationInputActive:
    case HandshakeFragmentSelectorKind::FuOperationResultActive:
    case HandshakeFragmentSelectorKind::MemoryOperationPlan:
      break;
    }

    std::vector<FabricPhysicalTraversalRef> witnesses;
    if (activationKind == HandshakeActivationKind::AnyTraversal ||
        activationKind == HandshakeActivationKind::AllTraversals ||
        activationKind ==
            HandshakeActivationKind::AnySwitchActivationTraversal ||
        activationKind ==
            HandshakeActivationKind::ExactSwitchActivationTraversal) {
      witnesses = pending.selector.traversalWitnesses;
      llvm::sort(witnesses, [](const auto &lhs, const auto &rhs) {
        return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
      });
      witnesses.erase(std::unique(witnesses.begin(), witnesses.end()),
                      witnesses.end());
      if (witnesses.empty())
        return invalid("traversal-selected fragment has no witness");
    }
    auto witnessOffset =
        checkedSize(model_.traversalWitnesses_, "handshake witness offset");
    auto witnessCount = checkedSize(witnesses, "handshake witness count");
    if (!witnessOffset)
      return witnessOffset.takeError();
    if (!witnessCount)
      return witnessCount.takeError();
    if ((activationKind ==
             HandshakeActivationKind::AnySwitchActivationTraversal ||
         activationKind ==
             HandshakeActivationKind::ExactSwitchActivationTraversal) !=
        pending.selector.switchActivation.has_value())
      return invalid("switch activation fragment has no exact key");
    model_.fragments_.push_back({*offset, *count, activationKind,
                                 *witnessOffset, *witnessCount,
                                 pending.selector.switchActivation});
    model_.fragmentContributionOrdinals_.insert(
        model_.fragmentContributionOrdinals_.end(), contributions.begin(),
        contributions.end());
    model_.traversalWitnesses_.insert(model_.traversalWitnesses_.end(),
                                      witnesses.begin(), witnesses.end());
    model_.fragmentSelectors_.push_back(std::move(pending.selector));
  }
  return std::move(model_);
}

} // namespace loom::fabric::detail
