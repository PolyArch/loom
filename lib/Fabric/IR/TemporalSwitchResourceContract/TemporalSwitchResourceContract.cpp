#include "Fabric/IR/TemporalSwitchResourceContract.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>
#include <utility>

using namespace fabric;

namespace {

constexpr CapacityDimensionKey serviceSlot{0};
constexpr EventKey transferEvent{0};
constexpr TimingContractKey sameCycleTransfer{0};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid temporal switch resource contract: " +
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

} // namespace

llvm::Expected<TemporalSwitchResourceContract>
TemporalSwitchResourceContract::create(
    TemporalSwitchResourceDeclaration declaration) {
  if (declaration.inputCount == 0 || declaration.outputCount == 0)
    return invalid("input and output domains must be non-empty");
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
  if (hasFanIn != declaration.grantPolicy.has_value())
    return invalid(hasFanIn ? "fan-in requires an exact grant policy"
                            : "a grant policy is forbidden without fan-in");

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
  resource.requesters = requesterKeys(declaration.inputCount);
  resource.eligibilityCount = static_cast<std::uint32_t>(traversalCount);
  resource.eventCount = 1;
  resource.usePatterns.reserve(static_cast<std::size_t>(traversalCount));

  std::uint32_t pattern = 0;
  for (std::uint32_t input = 0; input != declaration.inputCount; ++input) {
    for (std::uint32_t output : outputsByInput[input]) {
      resource.usePatterns.push_back(UsePatternDeclaration{
          UsePatternKey(pattern),
          RequesterKey(input),
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
  return TemporalSwitchResourceContract(
      declaration.inputCount, declaration.outputCount, std::move(inputOffsets),
      std::move(flatOutputs), std::move(*contract));
}

StateKey TemporalSwitchResourceContract::inputState(std::uint32_t input) const {
  assert(input < inputCount_ && "input ordinal outside switch domain");
  return inputStateKey(input);
}

StateKey
TemporalSwitchResourceContract::outputState(std::uint32_t output) const {
  assert(output < outputCount_ && "output ordinal outside switch domain");
  return outputStateKey(inputCount_, output);
}

RequesterKey
TemporalSwitchResourceContract::inputRequester(std::uint32_t input) const {
  assert(input < inputCount_ && "input ordinal outside switch domain");
  return RequesterKey(input);
}

llvm::Expected<UsePatternKey>
TemporalSwitchResourceContract::traversalPattern(std::uint32_t input,
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

llvm::Expected<UsePatternKey> fabric::resolveTemporalSwitchTraversalPattern(
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
    if (pattern.requester != RequesterKey(input) || pattern.claims.size() != 2)
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
