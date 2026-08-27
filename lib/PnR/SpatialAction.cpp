#include "PnR/SpatialAction.h"

#include "PnR/SpatialCandidateState.h"

#include "SpatialActionProposalInternal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <system_error>
#include <type_traits>

using namespace loom;
using namespace loom::pnr;

namespace {

struct ActionKey final {
  std::array<std::uint64_t, 5> fields{};

  friend bool operator==(const ActionKey &left, const ActionKey &right) {
    return left.fields == right.fields;
  }
  friend bool operator<(const ActionKey &left, const ActionKey &right) {
    return left.fields < right.fields;
  }
};

llvm::Error invalid(llvm::StringRef detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_action_domain_invalid: %s", detail.str().c_str());
}

llvm::Error invalidBatch(llvm::StringRef detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_action_batch_invalid: %s", detail.str().c_str());
}

ActionKey anchorKey(const SpatialRealizationBindingAction &action) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SpatialComputeBindingAction>)
          return ActionKey{{0, value.realization}};
        else
          return ActionKey{{1, value.realization}};
      },
      action);
}

ActionKey choiceKey(const SpatialRealizationBindingAction &action) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SpatialComputeBindingAction>)
          return ActionKey{{value.placement, value.instructionContext}};
        else
          return ActionKey{{value.placement}};
      },
      action);
}

ActionKey anchorKey(const SpatialTransportRoutingAction &action) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SpatialWholeNetRoutingAction> ||
                      std::is_same_v<T, SpatialSingleSinkRoutingAction> ||
                      std::is_same_v<T, SpatialRootedSubtreeRoutingAction>)
          return ActionKey{{0, value.logicalNet}};
        else if constexpr (std::is_same_v<T, SpatialWitnessRegionRoutingAction>)
          return ActionKey{{1, static_cast<std::uint32_t>(value.witnessKind),
                            value.witnessOrdinal}};
        else
          return ActionKey{{2}};
      },
      action);
}

ActionKey choiceKey(const SpatialTransportRoutingAction &action) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SpatialWholeNetRoutingAction>)
          return ActionKey{{0, static_cast<std::uint8_t>(value.disposition),
                            value.registerFifoTransfer}};
        else if constexpr (std::is_same_v<T, SpatialSingleSinkRoutingAction>)
          return ActionKey{{1, value.sinkObligation}};
        else if constexpr (std::is_same_v<T, SpatialRootedSubtreeRoutingAction>)
          return ActionKey{{2, value.rootEndpoint}};
        else
          return ActionKey{};
      },
      action);
}

ActionKey anchorKey(const SpatialResourceAllocationAction &action) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SpatialPortAttachmentAction>)
          return ActionKey{{0, value.demand}};
        else if constexpr (std::is_same_v<T,
                                          SpatialGraphBoundaryAttachmentAction>)
          return ActionKey{{1, value.boundary}};
        else if constexpr (std::is_same_v<T, SpatialMemoryOperationPlanAction>)
          return ActionKey{{2, value.actor}};
        else if constexpr (std::is_same_v<T, SpatialLogicalMemoryBindingAction>)
          return ActionKey{{3, value.binding}};
        else if constexpr (std::is_same_v<T, SpatialMemoryUseDispatchAction>)
          return ActionKey{{4, value.use}};
        else
          return ActionKey{{5, value.exposure}};
      },
      action);
}

ActionKey choiceKey(const SpatialResourceAllocationAction &action) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SpatialPortAttachmentAction> ||
                      std::is_same_v<T, SpatialGraphBoundaryAttachmentAction>)
          return ActionKey{{value.attachmentOption}};
        else if constexpr (std::is_same_v<T, SpatialMemoryOperationPlanAction>)
          return ActionKey{{value.plan}};
        else if constexpr (std::is_same_v<T, SpatialLogicalMemoryBindingAction>)
          return ActionKey{{value.target, value.physicalOffsetBytes}};
        else if constexpr (std::is_same_v<T, SpatialMemoryUseDispatchAction>)
          return ActionKey{{value.dispatchOption}};
        else
          return ActionKey{{value.exposureOption}};
      },
      action);
}

ActionKey batchAnchorKey(const SpatialMappingAction &action) {
  return std::visit(
      [](const auto &category) {
        ActionKey nested = anchorKey(category);
        ActionKey result;
        using Category = std::decay_t<decltype(category)>;
        if constexpr (std::is_same_v<Category, SpatialRealizationBindingAction>)
          result.fields[0] = 0;
        else if constexpr (std::is_same_v<Category,
                                          SpatialTransportRoutingAction>)
          result.fields[0] = 1;
        else
          result.fields[0] = 2;
        for (std::size_t index = 0; index + 1 < result.fields.size(); ++index)
          result.fields[index + 1] = nested.fields[index];
        return result;
      },
      action);
}

template <typename Action>
llvm::Error validateKindDomain(llvm::ArrayRef<SpatialActionChoiceRange> anchors,
                               llvm::ArrayRef<Action> choices) {
  std::size_t nextOffset = 0;
  ActionKey previousAnchor;
  bool firstAnchor = true;
  for (const SpatialActionChoiceRange &range : anchors) {
    if (range.choiceCount == 0 || range.choiceOffset != nextOffset ||
        range.choiceOffset > choices.size() ||
        range.choiceCount > choices.size() - range.choiceOffset)
      return invalid("choice ranges are not a contiguous nonempty partition");
    const auto rangeChoices =
        choices.slice(range.choiceOffset, range.choiceCount);
    const ActionKey currentAnchor = anchorKey(rangeChoices.front());
    if (!firstAnchor && !(previousAnchor < currentAnchor))
      return invalid("anchors are not in canonical unique order");
    firstAnchor = false;
    previousAnchor = currentAnchor;

    ActionKey previousChoice;
    bool firstChoice = true;
    for (const Action &choice : rangeChoices) {
      if (!(anchorKey(choice) == currentAnchor))
        return invalid("one choice range crosses typed anchors");
      const ActionKey currentChoice = choiceKey(choice);
      if (!firstChoice && !(previousChoice < currentChoice))
        return invalid("choices are not in canonical unique order");
      firstChoice = false;
      previousChoice = currentChoice;
    }
    nextOffset += range.choiceCount;
  }
  if (nextOffset != choices.size())
    return invalid("choices are not completely owned by anchor ranges");
  return llvm::Error::success();
}

enum class LiveKind : std::uint8_t {
  Realization,
  Transport,
  Resource,
};

struct LiveKindRecord final {
  LiveKind kind;
  std::uint64_t weight;
};

llvm::Expected<std::optional<SpatialMappingAction>>
selectSpatialAction(const ResolvedPnrActionProposalPolicy &policy,
                    SpatialActionProposalDomain domain,
                    DeterministicPnrRandomStream &proposalStream) {
  std::array<LiveKindRecord, 3> live{};
  std::size_t liveCount = 0;
  auto addLive = [&](LiveKind kind, std::uint64_t weight, std::size_t anchors) {
    if (weight != 0 && anchors != 0)
      live[liveCount++] = {kind, weight};
  };
  addLive(LiveKind::Realization, policy.realizationBindingWeight,
          domain.realizationAnchors.size());
  addLive(LiveKind::Transport, policy.transportRoutingWeight,
          domain.transportAnchors.size());
  addLive(LiveKind::Resource, policy.resourceAllocationWeight,
          domain.resourceAnchors.size());
  if (liveCount == 0)
    return std::optional<SpatialMappingAction>{};

  std::uint64_t divisor = 0;
  for (std::size_t index = 0; index < liveCount; ++index)
    divisor = std::gcd(divisor, live[index].weight);
  std::uint64_t totalWeight = 0;
  for (std::size_t index = 0; index < liveCount; ++index) {
    live[index].weight /= divisor;
    totalWeight += live[index].weight;
  }
  auto selectedWeight = proposalStream.nextBounded(totalWeight);
  if (!selectedWeight)
    return selectedWeight.takeError();
  const LiveKindRecord *selectedKind = nullptr;
  for (std::size_t index = 0; index < liveCount; ++index) {
    if (*selectedWeight < live[index].weight) {
      selectedKind = &live[index];
      break;
    }
    *selectedWeight -= live[index].weight;
  }
  if (!selectedKind)
    return invalid("weighted kind selection escaped the live domain");

  auto select = [&](const auto &anchors,
                    const auto &choices) -> llvm::Expected<std::size_t> {
    auto anchor = proposalStream.nextBounded(anchors.size());
    if (!anchor)
      return anchor.takeError();
    const SpatialActionChoiceRange &range = anchors[*anchor];
    auto choice = proposalStream.nextBounded(range.choiceCount);
    if (!choice)
      return choice.takeError();
    return static_cast<std::size_t>(range.choiceOffset + *choice);
  };

  switch (selectedKind->kind) {
  case LiveKind::Realization: {
    auto choice = select(domain.realizationAnchors, domain.realizationChoices);
    if (!choice)
      return choice.takeError();
    return SpatialMappingAction(domain.realizationChoices[*choice]);
  }
  case LiveKind::Transport: {
    auto choice = select(domain.transportAnchors, domain.transportChoices);
    if (!choice)
      return choice.takeError();
    return SpatialMappingAction(domain.transportChoices[*choice]);
  }
  case LiveKind::Resource: {
    auto choice = select(domain.resourceAnchors, domain.resourceChoices);
    if (!choice)
      return choice.takeError();
    return SpatialMappingAction(domain.resourceChoices[*choice]);
  }
  }
  llvm_unreachable("all live Spatial Action kinds are handled");
}

} // namespace

SpatialActionKey
loom::pnr::spatialActionKey(const SpatialMappingAction &action) {
  return std::visit(
      [](const auto &category) -> SpatialActionKey {
        using Category = std::decay_t<decltype(category)>;
        if constexpr (std::is_same_v<Category,
                                     SpatialRealizationBindingAction>) {
          return std::visit(
              [](const auto &choice) -> SpatialActionKey {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SpatialComputeBindingAction>)
                  return {{0, 0, choice.realization, choice.placement,
                           choice.instructionContext, 0}};
                else
                  return {{0, 1, choice.realization, choice.placement, 0, 0}};
              },
              category);
        } else if constexpr (std::is_same_v<Category,
                                            SpatialTransportRoutingAction>) {
          return std::visit(
              [](const auto &choice) -> SpatialActionKey {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SpatialWholeNetRoutingAction>)
                  return {{1, 0, choice.logicalNet,
                           static_cast<std::uint8_t>(choice.disposition),
                           choice.registerFifoTransfer, 0}};
                else if constexpr (std::is_same_v<
                                       Choice, SpatialSingleSinkRoutingAction>)
                  return {
                      {1, 1, choice.logicalNet, choice.sinkObligation, 0, 0}};
                else if constexpr (std::is_same_v<
                                       Choice,
                                       SpatialRootedSubtreeRoutingAction>)
                  return {{1, 2, choice.logicalNet, choice.rootEndpoint, 0, 0}};
                else if constexpr (std::is_same_v<
                                       Choice,
                                       SpatialWitnessRegionRoutingAction>)
                  return {{1, 3, static_cast<std::uint64_t>(choice.witnessKind),
                           choice.witnessOrdinal, 0, 0}};
                else
                  return {{1, 4, 0, 0, 0, 0}};
              },
              category);
        } else {
          return std::visit(
              [](const auto &choice) -> SpatialActionKey {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SpatialPortAttachmentAction>)
                  return {{2, 0, choice.demand, choice.attachmentOption, 0, 0}};
                else if constexpr (std::is_same_v<
                                       Choice,
                                       SpatialGraphBoundaryAttachmentAction>)
                  return {
                      {2, 1, choice.boundary, choice.attachmentOption, 0, 0}};
                else if constexpr (std::is_same_v<
                                       Choice,
                                       SpatialMemoryOperationPlanAction>)
                  return {{2, 2, choice.actor, choice.plan, 0, 0}};
                else if constexpr (std::is_same_v<
                                       Choice,
                                       SpatialLogicalMemoryBindingAction>)
                  return {{2, 3, choice.binding, choice.target,
                           choice.physicalOffsetBytes, 0}};
                else if constexpr (std::is_same_v<
                                       Choice, SpatialMemoryUseDispatchAction>)
                  return {{2, 4, choice.use, choice.dispatchOption, 0, 0}};
                else
                  return {{2, 5, choice.exposure, choice.exposureOption, 0, 0}};
              },
              category);
        }
      },
      action);
}

llvm::Error loom::pnr::validateCanonicalSpatialActionBatch(
    llvm::ArrayRef<SpatialMappingAction> actions) {
  if (actions.empty())
    return invalidBatch("ActionBatch is empty");

  ActionKey previous;
  bool first = true;
  for (const SpatialMappingAction &action : actions) {
    const ActionKey current = batchAnchorKey(action);
    if (!first) {
      if (current == previous)
        return invalidBatch("ActionBatch anchors are not unique");
      if (!(previous < current))
        return invalidBatch("ActionBatch anchors are not in canonical order");
    }
    previous = current;
    first = false;
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<SpatialMappingAction>>
loom::pnr::proposeSpatialAction(const ResolvedPnrActionProposalPolicy &policy,
                                SpatialActionProposalDomain domain,
                                DeterministicPnrRandomStream &proposalStream) {
  if (llvm::Error error = validateResolvedPnrActionProposalPolicy(policy))
    return std::move(error);
  if (llvm::Error error = validateKindDomain(domain.realizationAnchors,
                                             domain.realizationChoices))
    return std::move(error);
  if (llvm::Error error =
          validateKindDomain(domain.transportAnchors, domain.transportChoices))
    return std::move(error);
  if (llvm::Error error =
          validateKindDomain(domain.resourceAnchors, domain.resourceChoices))
    return std::move(error);
  return selectSpatialAction(policy, domain, proposalStream);
}

llvm::Expected<std::optional<SpatialMappingAction>>
loom::pnr::detail::proposeCanonicalSpatialAction(
    const ResolvedPnrActionProposalPolicy &policy,
    SpatialActionProposalDomain domain,
    DeterministicPnrRandomStream &proposalStream) {
  return selectSpatialAction(policy, domain, proposalStream);
}

bool loom::pnr::isIdentitySpatialAction(const SpatialCandidateState &candidate,
                                        const SpatialMappingAction &action) {
  if (const auto *realization =
          std::get_if<SpatialRealizationBindingAction>(&action)) {
    if (const auto *compute =
            std::get_if<SpatialComputeBindingAction>(realization)) {
      const auto current = candidate.computeBinding(compute->realization);
      return current.placement == compute->placement &&
             current.instructionContext == compute->instructionContext;
    }
    const auto &memory = std::get<SpatialMemoryBindingAction>(*realization);
    return candidate.memoryBinding(memory.realization).placement ==
           memory.placement;
  }
  if (const auto *resource =
          std::get_if<SpatialResourceAllocationAction>(&action)) {
    if (const auto *port = std::get_if<SpatialPortAttachmentAction>(resource))
      return candidate.portAttachment(port->demand) == port->attachmentOption;
    if (const auto *boundary =
            std::get_if<SpatialGraphBoundaryAttachmentAction>(resource))
      return candidate.graphBoundaryAttachment(boundary->boundary) ==
             boundary->attachmentOption;
    if (const auto *plan =
            std::get_if<SpatialMemoryOperationPlanAction>(resource))
      return candidate.memoryOperationPlan(plan->actor) == plan->plan;
    if (const auto *binding =
            std::get_if<SpatialLogicalMemoryBindingAction>(resource)) {
      const auto &current = candidate.logicalMemoryBinding(binding->binding);
      return current.target == binding->target &&
             current.physicalOffsetBytes == binding->physicalOffsetBytes;
    }
    if (const auto *dispatch =
            std::get_if<SpatialMemoryUseDispatchAction>(resource))
      return candidate.memoryUseDispatch(dispatch->use) ==
             dispatch->dispatchOption;
    const auto &exposure = std::get<SpatialMemoryExposureAction>(*resource);
    return candidate.memoryExposureSelection(exposure.exposure) ==
           exposure.exposureOption;
  }
  return false;
}
