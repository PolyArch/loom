#include "Fabric/IR/SwitchResourceContract.h"

#include "Fabric/IR/Crosspoint.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>

using namespace fabric;

namespace {

constexpr CapacityDimensionKey serviceSlot{0};
constexpr EventKey transferEvent{0};
constexpr TimingContractKey sameCycleTransfer{0};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid switch resource contract: " +
                                     message);
}

StateKey inputStateKey(std::uint32_t input) { return StateKey(input); }

StateKey outputStateKey(std::uint32_t inputCount, std::uint32_t output) {
  return StateKey(inputCount + output);
}

std::vector<RequesterKey> requesterKeys(std::uint32_t inputCount) {
  std::vector<RequesterKey> requesters;
  requesters.reserve(inputCount);
  for (std::uint32_t input = 0; input != inputCount; ++input)
    requesters.emplace_back(input);
  return requesters;
}

struct SwitchPolicyProjection final {
  std::vector<std::uint32_t> requesterOrder;
  std::optional<std::uint32_t> roundRobinResetPosition;
};

SwitchPolicyProjection projectSwitchPolicy(const ResourceContract &contract) {
  SwitchPolicyProjection result;
  const auto policy = contract.grantPolicy();
  if (!policy)
    return result;
  if (const auto *fixed = std::get_if<FixedPriorityView>(&*policy)) {
    result.requesterOrder.reserve(fixed->requesterOrder().size());
    for (RequesterKey requester : fixed->requesterOrder())
      result.requesterOrder.push_back(requester.ordinal());
    return result;
  }
  const auto &roundRobin = std::get<RoundRobinView>(*policy);
  result.requesterOrder.reserve(roundRobin.requesterCycle().size());
  for (RequesterKey requester : roundRobin.requesterCycle())
    result.requesterOrder.push_back(requester.ordinal());
  const auto reset =
      llvm::find(result.requesterOrder, roundRobin.resetCursor().ordinal());
  assert(reset != result.requesterOrder.end() &&
         "validated RoundRobin reset requester is absent");
  result.roundRobinResetPosition =
      static_cast<std::uint32_t>(reset - result.requesterOrder.begin());
  return result;
}

} // namespace

llvm::Expected<std::uint64_t>
fabric::validatedSwitchCrosspointCount(std::uint64_t inputCount,
                                       std::uint64_t outputCount) {
  auto crosspoints = checkedCrosspointCount(inputCount, outputCount);
  if (!crosspoints)
    return crosspoints.takeError();
  if (*crosspoints > kSwitchCrosspointLimit)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "switch crossbar has %llu crosspoints, exceeding maximum %llu",
        static_cast<unsigned long long>(*crosspoints),
        static_cast<unsigned long long>(kSwitchCrosspointLimit));
  return *crosspoints;
}

llvm::Expected<std::vector<SwitchArbitrationComponent>>
fabric::deriveSwitchArbitrationComponents(
    Schedule schedule, std::uint32_t inputCount, std::uint32_t outputCount,
    llvm::ArrayRef<std::vector<std::uint32_t>> sourcesByOutput,
    const ResourceContract &contract) {
  if (inputCount == 0 || outputCount == 0 ||
      sourcesByOutput.size() != outputCount)
    return invalid("arbitration topology has an invalid shape");

  std::vector<std::uint32_t> parent(inputCount + outputCount);
  for (std::uint32_t index = 0; index != parent.size(); ++index)
    parent[index] = index;
  const auto root = [&](std::uint32_t value) {
    std::uint32_t result = value;
    while (parent[result] != result)
      result = parent[result];
    while (parent[value] != value) {
      const std::uint32_t next = parent[value];
      parent[value] = result;
      value = next;
    }
    return result;
  };
  for (std::uint32_t output = 0; output != outputCount; ++output) {
    if (sourcesByOutput[output].empty())
      return invalid("arbitration topology has an unconnected output");
    for (std::uint32_t input : sourcesByOutput[output]) {
      if (input >= inputCount)
        return invalid("arbitration topology input is outside its domain");
      std::uint32_t inputRoot = root(input);
      std::uint32_t outputRoot = root(inputCount + output);
      if (inputRoot != outputRoot)
        parent[outputRoot] = inputRoot;
    }
  }

  std::map<std::uint32_t, SwitchArbitrationComponent> byRoot;
  for (std::uint32_t input = 0; input != inputCount; ++input)
    byRoot[root(input)].inputs.push_back(input);
  for (std::uint32_t output = 0; output != outputCount; ++output)
    byRoot[root(inputCount + output)].outputs.push_back(output);

  const SwitchPolicyProjection policy = projectSwitchPolicy(contract);
  const std::vector<std::uint32_t> &fullOrder = policy.requesterOrder;
  const bool roundRobin = policy.roundRobinResetPosition.has_value();

  std::vector<SwitchArbitrationComponent> result;
  result.reserve(byRoot.size());
  for (auto &[componentRoot, component] : byRoot) {
    (void)componentRoot;
    if (schedule == Schedule::Temporal && !fullOrder.empty()) {
      for (std::uint32_t requester : fullOrder)
        if (llvm::is_contained(component.inputs, requester))
          component.requesterOrder.push_back(requester);
      if (component.requesterOrder.size() != component.inputs.size())
        return invalid("GrantPolicy omits a switch component requester");
      if (roundRobin && component.inputs.size() > 1) {
        const std::uint32_t resetOrdinal = *policy.roundRobinResetPosition;
        std::uint32_t bestDistance = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t bestPosition = 0;
        for (auto [position, requester] :
             llvm::enumerate(component.requesterOrder)) {
          const auto found = llvm::find(fullOrder, requester);
          const std::uint32_t ordinal =
              static_cast<std::uint32_t>(found - fullOrder.begin());
          const std::uint32_t distance =
              ordinal >= resetOrdinal
                  ? ordinal - resetOrdinal
                  : static_cast<std::uint32_t>(fullOrder.size()) -
                        resetOrdinal + ordinal;
          if (distance < bestDistance) {
            bestDistance = distance;
            bestPosition = static_cast<std::uint32_t>(position);
          }
        }
        component.roundRobinResetPosition = bestPosition;
      }
    } else {
      component.requesterOrder = component.inputs;
    }
    result.push_back(std::move(component));
  }
  llvm::sort(result, [](const SwitchArbitrationComponent &lhs,
                        const SwitchArbitrationComponent &rhs) {
    return lhs.inputs < rhs.inputs;
  });
  return result;
}

llvm::Expected<SwitchResourceContract>
SwitchResourceContract::create(SwitchResourceDeclaration declaration) {
  if (declaration.inputCount == 0 || declaration.outputCount == 0)
    return invalid("input and output domains must be non-empty");
  if (auto crosspoints = validatedSwitchCrosspointCount(
          declaration.inputCount, declaration.outputCount);
      !crosspoints)
    return crosspoints.takeError();
  if (declaration.sourcesByOutput.size() != declaration.outputCount)
    return invalid("connectivity row count does not match the output domain");
  if (declaration.inputCount >
      std::numeric_limits<std::uint32_t>::max() - declaration.outputCount)
    return invalid("input and output state domains exceed u32");

  std::vector<std::vector<std::uint32_t>> outputsByInput(
      declaration.inputCount);
  bool hasFanIn = false;
  std::uint64_t traversalCount = 0;
  for (std::uint32_t output = 0; output != declaration.outputCount; ++output) {
    auto &sources = declaration.sourcesByOutput[output];
    if (sources.empty())
      return invalid("an output has no admitted input");
    llvm::sort(sources);
    if (std::adjacent_find(sources.begin(), sources.end()) != sources.end())
      return invalid("a connectivity row contains a duplicate input");
    hasFanIn |= sources.size() > 1;
    traversalCount += sources.size();
    if (traversalCount > std::numeric_limits<std::uint32_t>::max())
      return invalid("the traversal domain exceeds u32");
    for (std::uint32_t input : sources) {
      if (input >= declaration.inputCount)
        return invalid("a connectivity input is outside the input domain");
      outputsByInput[input].push_back(output);
    }
  }
  for (llvm::ArrayRef<std::uint32_t> outputs : outputsByInput)
    if (outputs.empty())
      return invalid("an input has no admitted output");
  if (declaration.schedule == Schedule::Spatial) {
    if (declaration.grantPolicy)
      return invalid("a spatial switch cannot declare a grant policy");
  } else if (declaration.schedule == Schedule::Temporal) {
    if (hasFanIn != declaration.grantPolicy.has_value())
      return invalid(hasFanIn ? "temporal fan-in requires an exact grant policy"
                              : "a grant policy is forbidden without fan-in");
  } else {
    return invalid("schedule is outside the closed switch domain");
  }

  std::vector<std::uint32_t> inputOffsets(declaration.inputCount + 1, 0);
  std::vector<std::uint32_t> flatOutputs;
  flatOutputs.reserve(static_cast<std::size_t>(traversalCount));
  for (std::uint32_t input = 0; input != declaration.inputCount; ++input) {
    inputOffsets[input] = static_cast<std::uint32_t>(flatOutputs.size());
    flatOutputs.insert(flatOutputs.end(), outputsByInput[input].begin(),
                       outputsByInput[input].end());
  }
  inputOffsets.back() = static_cast<std::uint32_t>(flatOutputs.size());

  ResourceContractDeclaration resource;
  resource.states.reserve(declaration.inputCount + declaration.outputCount);
  for (std::uint32_t state = 0;
       state != declaration.inputCount + declaration.outputCount; ++state)
    resource.states.push_back(ResourceStateDeclaration{
        StateKey(state),
        {CapacityDimensionDeclaration{serviceSlot, CapacityUnits(1),
                                      CapacityUnits(0)}}});
  resource.timingContracts = {
      TimingContractDeclaration{sameCycleTransfer, {0}}};
  resource.requesters = requesterKeys(
      declaration.schedule == Schedule::Spatial ? 1 : declaration.inputCount);
  resource.eligibilityCount = static_cast<std::uint32_t>(traversalCount);
  resource.eventCount = 1;
  resource.usePatterns.reserve(static_cast<std::size_t>(traversalCount));

  std::uint32_t pattern = 0;
  for (std::uint32_t input = 0; input != declaration.inputCount; ++input) {
    for (std::uint32_t output : outputsByInput[input]) {
      resource.usePatterns.push_back(UsePatternDeclaration{
          UsePatternKey(pattern),
          RequesterKey(declaration.schedule == Schedule::Spatial ? 0 : input),
          EligibilityKey(pattern),
          transferEvent,
          transferEvent,
          std::nullopt,
          sameCycleTransfer,
          {{ClaimKey(0), inputStateKey(input), serviceSlot, CapacityUnits(1)},
           {ClaimKey(1), outputStateKey(declaration.inputCount, output),
            serviceSlot, CapacityUnits(1)}},
          {{{ClaimKey(0), ClaimKey(1)}}}});
      ++pattern;
    }
  }

  if (declaration.grantPolicy) {
    resource.grantPolicy = std::visit(
        [](auto &&policy) -> GrantPolicyDeclaration {
          using Policy = std::decay_t<decltype(policy)>;
          if constexpr (std::is_same_v<Policy, TemporalSwitchFixedPriority>) {
            std::vector<RequesterKey> order;
            order.reserve(policy.requesterOrder.size());
            for (std::uint32_t requester : policy.requesterOrder)
              order.emplace_back(requester);
            return FixedPriorityDeclaration{std::move(order)};
          } else {
            std::vector<RequesterKey> cycle;
            cycle.reserve(policy.requesterCycle.size());
            for (std::uint32_t requester : policy.requesterCycle)
              cycle.emplace_back(requester);
            return RoundRobinDeclaration{std::move(cycle),
                                         RequesterKey(policy.resetRequester)};
          }
        },
        *declaration.grantPolicy);
  }

  auto contract = ResourceContract::create(resource);
  if (!contract)
    return contract.takeError();
  return SwitchResourceContract(
      declaration.schedule, declaration.inputCount, declaration.outputCount,
      std::move(inputOffsets), std::move(flatOutputs), std::move(*contract));
}

StateKey SwitchResourceContract::inputState(std::uint32_t input) const {
  assert(input < inputCount_ && "input ordinal outside switch domain");
  return inputStateKey(input);
}

StateKey SwitchResourceContract::outputState(std::uint32_t output) const {
  assert(output < outputCount_ && "output ordinal outside switch domain");
  return outputStateKey(inputCount_, output);
}

RequesterKey SwitchResourceContract::inputRequester(std::uint32_t input) const {
  assert(input < inputCount_ && "input ordinal outside switch domain");
  return RequesterKey(schedule_ == Schedule::Spatial ? 0 : input);
}

llvm::Expected<UsePatternKey>
SwitchResourceContract::traversalPattern(std::uint32_t input,
                                         std::uint32_t output) const {
  if (input >= inputCount_ || output >= outputCount_)
    return invalid("traversal endpoint is outside the switch domain");
  const std::uint32_t begin = inputOffsets_[input];
  const std::uint32_t end = inputOffsets_[input + 1];
  auto first = outputsByInput_.begin() + begin;
  auto last = outputsByInput_.begin() + end;
  auto found = std::lower_bound(first, last, output);
  if (found == last || *found != output)
    return invalid("traversal is not admitted by switch connectivity");
  return UsePatternKey(begin +
                       static_cast<std::uint32_t>(std::distance(first, found)));
}

llvm::Expected<UsePatternKey> fabric::resolveSwitchTraversalPattern(
    const ResourceContract &contract, std::uint32_t inputCount,
    std::uint32_t input, std::uint32_t output) {
  if (input >= inputCount)
    return invalid("traversal input is outside the switch domain");
  if (output > std::numeric_limits<std::uint32_t>::max() - inputCount)
    return invalid("traversal output state exceeds u32");
  const StateKey inputState(input);
  const StateKey outputState(inputCount + output);
  std::optional<UsePatternKey> resolved;
  for (std::uint32_t ordinal = 0; ordinal != contract.usePatternCount();
       ++ordinal) {
    const UsePattern pattern = contract.usePattern(UsePatternKey(ordinal));
    if (pattern.claims.size() != 2)
      continue;
    bool hasInput = false;
    bool hasOutput = false;
    bool exact = true;
    for (const Claim &claim : pattern.claims) {
      exact &=
          claim.dimension == serviceSlot && claim.amount == CapacityUnits(1);
      hasInput |= claim.state == inputState;
      hasOutput |= claim.state == outputState;
    }
    if (!exact || !hasInput || !hasOutput)
      continue;
    if (resolved)
      return invalid("traversal resolves to multiple switch use patterns");
    resolved = UsePatternKey(ordinal);
  }
  if (!resolved)
    return invalid("traversal has no switch use pattern");
  return *resolved;
}
