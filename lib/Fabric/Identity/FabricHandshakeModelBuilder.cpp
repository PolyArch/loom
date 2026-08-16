#include "FabricHandshakeInternal.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <unordered_map>
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

std::uint64_t packedArc(std::uint32_t source, std::uint32_t destination) {
  return (static_cast<std::uint64_t>(source) << 32) | destination;
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
    : model_(std::move(owner)) {
  boundaryNodes_.reserve(32);
  junctionNodes_.reserve(64);
}

std::uint32_t
HandshakeOwnerModelBuilder::boundarySignal(HandshakeSignalRef signal) {
  std::vector<std::uint8_t> key = canonicalFabricBytes(signal.endpoint);
  key.push_back(static_cast<std::uint8_t>(signal.signal));
  auto found = boundaryNodes_.find(key);
  if (found != boundaryNodes_.end())
    return found->second;
  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(model_.nodes_.size());
  model_.nodes_.push_back(
      {HandshakeOwnerNodeKind::BoundarySignal, std::move(signal)});
  boundaryNodes_.emplace(std::move(key), ordinal);
  return ordinal;
}

std::uint32_t HandshakeOwnerModelBuilder::junction(
    std::vector<std::uint8_t> ownerLocalKey) {
  const auto found = junctionNodes_.find(ownerLocalKey);
  if (found != junctionNodes_.end())
    return found->second;
  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(model_.nodes_.size());
  model_.nodes_.push_back(
      {HandshakeOwnerNodeKind::OwnerLocalJunction, std::nullopt});
  junctionNodes_.emplace(std::move(ownerLocalKey), ordinal);
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

  std::vector<std::uint64_t> uniqueArcs;
  for (const PendingFragment &fragment : pending_)
    for (const auto &[source, destination] : fragment.arcs)
      uniqueArcs.push_back(packedArc(source, destination));
  llvm::sort(uniqueArcs);
  uniqueArcs.erase(std::unique(uniqueArcs.begin(), uniqueArcs.end()),
                   uniqueArcs.end());
  if (uniqueArcs.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("handshake arc count exceeds the owner-model index domain");

  std::unordered_map<std::uint64_t, std::uint32_t> arcOrdinals;
  arcOrdinals.reserve(uniqueArcs.size());
  model_.arcs_.reserve(uniqueArcs.size());
  for (const std::uint64_t arc : uniqueArcs) {
    const std::uint32_t ordinal =
        static_cast<std::uint32_t>(model_.arcs_.size());
    arcOrdinals.emplace(arc, ordinal);
    model_.arcs_.push_back(
        {static_cast<std::uint32_t>(arc >> 32),
         static_cast<std::uint32_t>(arc)});
  }

  model_.fragments_.reserve(pending_.size());
  model_.fragmentSelectors_.reserve(pending_.size());
  for (PendingFragment &pending : pending_) {
    std::vector<std::uint32_t> contributions;
    contributions.reserve(pending.arcs.size());
    for (const auto &[source, destination] : pending.arcs)
      contributions.push_back(
          arcOrdinals.at(packedArc(source, destination)));
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
      struct KeyedTraversal final {
        std::vector<std::uint8_t> key;
        FabricPhysicalTraversalRef traversal;
      };
      std::vector<KeyedTraversal> keyed;
      keyed.reserve(pending.selector.traversalWitnesses.size());
      for (const FabricPhysicalTraversalRef &traversal :
           pending.selector.traversalWitnesses)
        keyed.push_back({canonicalFabricBytes(traversal), traversal});
      llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
        return lhs.key < rhs.key;
      });
      keyed.erase(std::unique(keyed.begin(), keyed.end(),
                              [](const auto &lhs, const auto &rhs) {
                                return lhs.traversal == rhs.traversal;
                              }),
                  keyed.end());
      witnesses.reserve(keyed.size());
      for (KeyedTraversal &entry : keyed)
        witnesses.push_back(std::move(entry.traversal));
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
