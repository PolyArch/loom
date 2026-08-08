#include "PnR/System/SystemAction.h"

#include <array>
#include <limits>
#include <numeric>
#include <system_error>
#include <type_traits>

using namespace loom;
using namespace loom::pnr;

namespace {

struct ActionKey final {
  std::array<std::uint64_t, 3> fields{};

  friend bool operator==(const ActionKey &lhs, const ActionKey &rhs) {
    return lhs.fields == rhs.fields;
  }
  friend bool operator<(const ActionKey &lhs, const ActionKey &rhs) {
    return lhs.fields < rhs.fields;
  }
};

llvm::Error invalid(llvm::StringRef detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_action_domain_invalid: %s", detail.str().c_str());
}

ActionKey anchorKey(const SystemExecutionBindingAction &action) {
  return {{action.decision}};
}

ActionKey choiceKey(const SystemExecutionBindingAction &action) {
  return {{action.choice}};
}

ActionKey anchorKey(const SystemTransportRoutingAction &action) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SystemWholeLegRoutingAction> ||
                      std::is_same_v<T, SystemSingleSinkRoutingAction> ||
                      std::is_same_v<T, SystemRootedSubtreeRoutingAction>)
          return ActionKey{{0, value.leg}};
        else if constexpr (std::is_same_v<T, SystemWitnessRegionRoutingAction>)
          return ActionKey{{1, static_cast<std::uint32_t>(value.witnessKind),
                            value.witnessOrdinal}};
        else
          return ActionKey{{2}};
      },
      action);
}

ActionKey choiceKey(const SystemTransportRoutingAction &action) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SystemWholeLegRoutingAction>)
          return ActionKey{{0}};
        else if constexpr (std::is_same_v<T, SystemSingleSinkRoutingAction>)
          return ActionKey{{1, value.sinkObligation}};
        else if constexpr (std::is_same_v<T, SystemRootedSubtreeRoutingAction>)
          return ActionKey{{2, value.rootEndpoint}};
        else
          return ActionKey{};
      },
      action);
}

ActionKey anchorKey(const SystemResourceAllocationAction &action) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SystemServiceTargetAction>)
          return ActionKey{{0, value.context}};
        else if constexpr (std::is_same_v<T, SystemInstructionUsePatternAction>)
          return ActionKey{{1, value.use}};
        else
          return ActionKey{{2, value.use}};
      },
      action);
}

ActionKey choiceKey(const SystemResourceAllocationAction &action) {
  return std::visit([](const auto &value) { return ActionKey{{value.choice}}; },
                    action);
}

template <typename Action>
llvm::Error validateDomain(llvm::ArrayRef<SystemActionChoiceRange> anchors,
                           llvm::ArrayRef<Action> choices) {
  std::size_t nextOffset = 0;
  ActionKey previousAnchor;
  bool firstAnchor = true;
  for (const SystemActionChoiceRange &range : anchors) {
    if (range.choiceCount == 0 || range.choiceOffset != nextOffset ||
        range.choiceOffset > choices.size() ||
        range.choiceCount > choices.size() - range.choiceOffset)
      return invalid("choice ranges are not a contiguous nonempty partition");
    const auto rangeChoices =
        choices.slice(range.choiceOffset, range.choiceCount);
    const ActionKey anchor = anchorKey(rangeChoices.front());
    if (!firstAnchor && !(previousAnchor < anchor))
      return invalid("anchors are not in canonical unique order");
    firstAnchor = false;
    previousAnchor = anchor;
    ActionKey previousChoice;
    bool firstChoice = true;
    for (const Action &choice : rangeChoices) {
      if (!(anchorKey(choice) == anchor))
        return invalid("one choice range crosses typed anchors");
      const ActionKey key = choiceKey(choice);
      if (!firstChoice && !(previousChoice < key))
        return invalid("choices are not in canonical unique order");
      firstChoice = false;
      previousChoice = key;
    }
    nextOffset += range.choiceCount;
  }
  if (nextOffset != choices.size())
    return invalid("choices are not completely owned by anchor ranges");
  return llvm::Error::success();
}

enum class LiveKind : std::uint8_t {
  Binding,
  Routing,
  Resource,
};

struct LiveKindRecord final {
  LiveKind kind;
  std::uint64_t weight;
};

} // namespace

llvm::Expected<std::optional<SystemMappingAction>>
loom::pnr::proposeSystemAction(const ResolvedPnrActionProposalPolicy &policy,
                               SystemActionProposalDomain domain,
                               DeterministicPnrRandomStream &stream) {
  if (llvm::Error error = validateResolvedPnrActionProposalPolicy(policy))
    return std::move(error);
  if (llvm::Error error =
          validateDomain(domain.bindingAnchors, domain.bindingChoices))
    return std::move(error);
  if (llvm::Error error =
          validateDomain(domain.routingAnchors, domain.routingChoices))
    return std::move(error);
  if (llvm::Error error =
          validateDomain(domain.resourceAnchors, domain.resourceChoices))
    return std::move(error);

  std::array<LiveKindRecord, 3> live{};
  std::size_t liveCount = 0;
  const auto addLive = [&](LiveKind kind, std::uint64_t weight,
                           std::size_t anchorCount) {
    if (weight != 0 && anchorCount != 0)
      live[liveCount++] = {kind, weight};
  };
  addLive(LiveKind::Binding, policy.realizationBindingWeight,
          domain.bindingAnchors.size());
  addLive(LiveKind::Routing, policy.transportRoutingWeight,
          domain.routingAnchors.size());
  addLive(LiveKind::Resource, policy.resourceAllocationWeight,
          domain.resourceAnchors.size());
  if (liveCount == 0)
    return std::optional<SystemMappingAction>{};

  std::uint64_t divisor = 0;
  for (std::size_t index = 0; index != liveCount; ++index)
    divisor = std::gcd(divisor, live[index].weight);
  std::uint64_t totalWeight = 0;
  for (std::size_t index = 0; index != liveCount; ++index) {
    live[index].weight /= divisor;
    if (live[index].weight >
        std::numeric_limits<std::uint64_t>::max() - totalWeight)
      return invalid("normalized Action kind weight sum overflows u64");
    totalWeight += live[index].weight;
  }
  auto selectedWeight = stream.nextBounded(totalWeight);
  if (!selectedWeight)
    return selectedWeight.takeError();
  const LiveKindRecord *selectedKind = nullptr;
  for (std::size_t index = 0; index != liveCount; ++index) {
    if (*selectedWeight < live[index].weight) {
      selectedKind = &live[index];
      break;
    }
    *selectedWeight -= live[index].weight;
  }
  if (!selectedKind)
    return invalid("weighted kind selection escaped the live domain");

  const auto select = [&](const auto &anchors,
                          const auto &choices) -> llvm::Expected<std::size_t> {
    auto anchor = stream.nextBounded(anchors.size());
    if (!anchor)
      return anchor.takeError();
    const SystemActionChoiceRange &range = anchors[*anchor];
    auto choice = stream.nextBounded(range.choiceCount);
    if (!choice)
      return choice.takeError();
    return static_cast<std::size_t>(range.choiceOffset + *choice);
  };

  switch (selectedKind->kind) {
  case LiveKind::Binding: {
    auto choice = select(domain.bindingAnchors, domain.bindingChoices);
    if (!choice)
      return choice.takeError();
    return SystemMappingAction(domain.bindingChoices[*choice]);
  }
  case LiveKind::Routing: {
    auto choice = select(domain.routingAnchors, domain.routingChoices);
    if (!choice)
      return choice.takeError();
    return SystemMappingAction(domain.routingChoices[*choice]);
  }
  case LiveKind::Resource: {
    auto choice = select(domain.resourceAnchors, domain.resourceChoices);
    if (!choice)
      return choice.takeError();
    return SystemMappingAction(domain.resourceChoices[*choice]);
  }
  }
  llvm_unreachable("unknown System Action kind");
}
