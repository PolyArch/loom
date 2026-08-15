#include "Fabric/IR/TemporalPeResourceContract.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

using namespace fabric;

namespace {

constexpr CapacityDimensionKey occupiedEntry{0};
constexpr CapacityDimensionKey firstPortService{1};
constexpr CapacityDimensionKey secondPortService{2};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid temporal PE resource contract: " +
                                     message);
}

llvm::Expected<std::uint32_t> checkedAdd(std::uint32_t lhs, std::uint32_t rhs,
                                         llvm::StringRef domain) {
  if (lhs > std::numeric_limits<std::uint32_t>::max() - rhs)
    return invalid(domain + " exceeds the owner-local u32 key domain");
  return lhs + rhs;
}

void appendRequesters(GrantPolicyDeclaration &policy,
                      std::uint32_t firstRequester,
                      std::uint32_t requesterCount) {
  std::visit(
      [&](auto &declaration) {
        auto &order = [&]() -> std::vector<RequesterKey> & {
          using Declaration = std::decay_t<decltype(declaration)>;
          if constexpr (std::is_same_v<Declaration, FixedPriorityDeclaration>)
            return declaration.requesterOrder;
          else
            return declaration.requesterCycle;
        }();
        order.reserve(order.size() + requesterCount);
        for (std::uint32_t ordinal = 0; ordinal != requesterCount; ++ordinal)
          order.emplace_back(firstRequester + ordinal);
      },
      policy);
}

} // namespace

llvm::Expected<TemporalPeResourceContract> TemporalPeResourceContract::create(
    const TemporalPeResourceDeclaration &declaration) {
  if (declaration.registerFifoPorts != 1 && declaration.registerFifoPorts != 2)
    return invalid("register FIFO port count must be one or two");
  if (declaration.registerFifoCount != 0 && declaration.registerFifoDepth == 0)
    return invalid("a non-empty register FIFO domain requires positive depth");
  if (declaration.fuInputCounts.empty())
    return invalid("context dispatch requires at least one FU occurrence");
  if (declaration.fuInputCounts.size() >
      std::numeric_limits<std::uint32_t>::max())
    return invalid("FU occurrence inventory exceeds the owner key domain");

  auto operandBuffer =
      TemporalOperandBufferContract::create(TemporalOperandBufferDeclaration{
          declaration.pe, declaration.contextCount, declaration.fuInputCounts,
          declaration.operandBufferMode,
          declaration.operandEntriesPerAllocationUnit});
  if (!operandBuffer)
    return operandBuffer.takeError();

  const std::uint32_t fuCount =
      static_cast<std::uint32_t>(declaration.fuInputCounts.size());
  const std::uint64_t dispatchCandidateCount64 =
      static_cast<std::uint64_t>(declaration.contextCount) * fuCount;
  if (dispatchCandidateCount64 > std::numeric_limits<std::uint32_t>::max())
    return invalid("context-dispatch candidate domain exceeds u32");
  const std::uint32_t dispatchCandidateCount =
      static_cast<std::uint32_t>(dispatchCandidateCount64);
  const std::uint32_t dispatchUnitCount =
      declaration.operandBufferMode == OperandBufferMode::AllFuShare ? 1
                                                                     : fuCount;

  std::vector<TemporalPeDispatchCandidate> dispatchCandidates;
  dispatchCandidates.reserve(dispatchCandidateCount);
  std::vector<std::uint32_t> unitCounts(dispatchUnitCount, 0);
  for (std::uint32_t context = 0; context != declaration.contextCount;
       ++context)
    for (std::uint32_t fu = 0; fu != fuCount; ++fu) {
      const std::uint32_t unit =
          declaration.operandBufferMode == OperandBufferMode::AllFuShare ? 0
                                                                         : fu;
      dispatchCandidates.push_back(
          {{declaration.pe, context},
           static_cast<loom::fabric::FabricOrdinal>(fu),
           unit});
      ++unitCounts[unit];
    }
  std::vector<Span> dispatchUnitSpans(dispatchUnitCount);
  std::uint32_t unitOffset = 0;
  for (std::uint32_t unit = 0; unit != dispatchUnitCount; ++unit) {
    dispatchUnitSpans[unit] = {unitOffset, unitCounts[unit]};
    unitOffset += unitCounts[unit];
  }
  std::vector<std::uint32_t> dispatchUnitCandidates(dispatchCandidateCount);
  std::vector<std::uint32_t> unitFilled(dispatchUnitCount, 0);
  for (std::uint32_t candidate = 0; candidate != dispatchCandidateCount;
       ++candidate) {
    const std::uint32_t unit = dispatchCandidates[candidate].allocationUnit;
    dispatchUnitCandidates[dispatchUnitSpans[unit].first + unitFilled[unit]++] =
        candidate;
  }

  ResourceContractDeclaration combined =
      operandBuffer->resourceContract().declaration();
  const std::uint32_t dispatchStateOffset =
      operandBuffer->resourceContract().stateCount();
  const std::uint32_t registerTransitionOffset =
      operandBuffer->resourceContract().resourceTransitionCount();
  const std::uint32_t dispatchTimingOffset =
      operandBuffer->resourceContract().timingContractCount();
  const std::uint32_t dispatchPatternOffset =
      operandBuffer->resourceContract().usePatternCount();
  const std::uint32_t dispatchRequesterOffset =
      operandBuffer->resourceContract().requesterCount();
  const std::uint32_t dispatchEligibilityOffset =
      operandBuffer->resourceContract().eligibilityCount();
  const std::uint32_t dispatchEventOffset =
      operandBuffer->resourceContract().eventCount();

  auto registerStateOffset = checkedAdd(dispatchStateOffset, dispatchUnitCount,
                                        "context-dispatch state inventory");
  auto registerPatternOffset =
      checkedAdd(dispatchPatternOffset, dispatchCandidateCount,
                 "context-dispatch use-pattern inventory");
  auto registerRequesterOffset =
      checkedAdd(dispatchRequesterOffset, dispatchCandidateCount,
                 "context-dispatch requester inventory");
  auto registerEligibilityOffset = checkedAdd(
      dispatchEligibilityOffset, 1, "context-dispatch eligibility inventory");
  auto registerEventOffset =
      checkedAdd(dispatchEventOffset, 2, "context-dispatch event inventory");
  auto registerTimingOffset =
      checkedAdd(dispatchTimingOffset, 1, "context-dispatch timing inventory");
  if (!registerStateOffset)
    return registerStateOffset.takeError();
  if (!registerPatternOffset)
    return registerPatternOffset.takeError();
  if (!registerRequesterOffset)
    return registerRequesterOffset.takeError();
  if (!registerEligibilityOffset)
    return registerEligibilityOffset.takeError();
  if (!registerEventOffset)
    return registerEventOffset.takeError();
  if (!registerTimingOffset)
    return registerTimingOffset.takeError();

  auto stateCount =
      checkedAdd(*registerStateOffset, declaration.registerFifoCount,
                 "resource-state inventory");
  auto registerActionCount =
      checkedAdd(declaration.registerFifoCount, declaration.registerFifoCount,
                 "register FIFO action inventory");
  if (!registerActionCount)
    return registerActionCount.takeError();
  auto transitionCount =
      checkedAdd(registerTransitionOffset, *registerActionCount,
                 "resource-transition inventory");
  auto patternCount = checkedAdd(*registerPatternOffset, *registerActionCount,
                                 "use-pattern inventory");
  auto requesterCount =
      checkedAdd(*registerRequesterOffset, declaration.registerFifoCount,
                 "requester inventory");
  auto eligibilityCount = checkedAdd(*registerEligibilityOffset,
                                     declaration.registerFifoCount ? 2 : 0,
                                     "eligibility inventory");
  auto eventCount =
      checkedAdd(*registerEventOffset, declaration.registerFifoCount ? 3 : 0,
                 "event inventory");
  auto timingCount =
      checkedAdd(*registerTimingOffset, declaration.registerFifoCount ? 1 : 0,
                 "timing-contract inventory");
  if (!stateCount)
    return stateCount.takeError();
  if (!transitionCount)
    return transitionCount.takeError();
  if (!patternCount)
    return patternCount.takeError();
  if (!requesterCount)
    return requesterCount.takeError();
  if (!eligibilityCount)
    return eligibilityCount.takeError();
  if (!eventCount)
    return eventCount.takeError();
  if (!timingCount)
    return timingCount.takeError();

  for (TimingContractDeclaration &timing : combined.timingContracts)
    timing.eventRank.resize(*eventCount, 0);

  combined.states.reserve(*stateCount);
  combined.resourceTransitions.reserve(*transitionCount);
  combined.usePatterns.reserve(*patternCount);
  combined.requesters.reserve(*requesterCount);

  for (std::uint32_t unit = 0; unit != dispatchUnitCount; ++unit)
    combined.states.push_back(ResourceStateDeclaration{
        StateKey(dispatchStateOffset + unit),
        {{CapacityDimensionKey(0), CapacityUnits(1), CapacityUnits(0)}}});
  for (std::uint32_t candidate = 0; candidate != dispatchCandidateCount;
       ++candidate)
    combined.requesters.emplace_back(dispatchRequesterOffset + candidate);

  std::vector<std::uint32_t> dispatchEventRanks(*eventCount, 0);
  dispatchEventRanks[dispatchEventOffset] = 0;
  dispatchEventRanks[dispatchEventOffset + 1] = 1;
  combined.timingContracts.push_back(
      {TimingContractKey(dispatchTimingOffset), std::move(dispatchEventRanks)});
  for (std::uint32_t candidate = 0; candidate != dispatchCandidateCount;
       ++candidate) {
    const std::uint32_t unit = dispatchCandidates[candidate].allocationUnit;
    combined.usePatterns.push_back(UsePatternDeclaration{
        UsePatternKey(dispatchPatternOffset + candidate),
        RequesterKey(dispatchRequesterOffset + candidate),
        EligibilityKey(dispatchEligibilityOffset),
        EventKey(dispatchEventOffset),
        EventKey(dispatchEventOffset + 1),
        std::nullopt,
        TimingContractKey(dispatchTimingOffset),
        {{ClaimKey(0), StateKey(dispatchStateOffset + unit),
          CapacityDimensionKey(0), CapacityUnits(1)}},
        {}});
  }

  for (std::uint32_t fifo = 0; fifo != declaration.registerFifoCount; ++fifo) {
    ResourceStateDeclaration state{
        StateKey(*registerStateOffset + fifo),
        {{occupiedEntry, CapacityUnits(declaration.registerFifoDepth),
          CapacityUnits(0)},
         {firstPortService, CapacityUnits(1), CapacityUnits(0)}}};
    if (declaration.registerFifoPorts == 2)
      state.capacityDimensions.push_back(
          {secondPortService, CapacityUnits(1), CapacityUnits(0)});
    combined.states.push_back(std::move(state));
    combined.resourceTransitions.emplace_back(registerTransitionOffset + fifo);
    combined.resourceTransitions.emplace_back(
        registerTransitionOffset + declaration.registerFifoCount + fifo);
    combined.requesters.emplace_back(*registerRequesterOffset + fifo);
  }

  if (declaration.registerFifoCount != 0) {
    std::vector<std::uint32_t> eventRanks(*eventCount, 0);
    eventRanks[*registerEventOffset] = 0;
    eventRanks[*registerEventOffset + 1] = 0;
    eventRanks[*registerEventOffset + 2] = 1;
    combined.timingContracts.push_back(
        {TimingContractKey(*registerTimingOffset), std::move(eventRanks)});

    const auto appendPattern = [&](std::uint32_t fifo,
                                   bool write) -> UsePatternDeclaration {
      const StateKey state(*registerStateOffset + fifo);
      const CapacityDimensionKey service =
          write || declaration.registerFifoPorts == 1 ? firstPortService
                                                      : secondPortService;
      const std::uint32_t role = write ? 0 : 1;
      const ResourceTransitionKey transition(
          registerTransitionOffset + role * declaration.registerFifoCount +
          fifo);
      return UsePatternDeclaration{
          UsePatternKey(*registerPatternOffset +
                        role * declaration.registerFifoCount + fifo),
          RequesterKey(*registerRequesterOffset + fifo),
          EligibilityKey(*registerEligibilityOffset + role),
          EventKey(*registerEventOffset + role),
          EventKey(*registerEventOffset + 2),
          CommitDeclaration{EventKey(*registerEventOffset + role), transition},
          TimingContractKey(*registerTimingOffset),
          {{ClaimKey(0), state, service, CapacityUnits(1)}},
          {}};
    };
    for (std::uint32_t fifo = 0; fifo != declaration.registerFifoCount; ++fifo)
      combined.usePatterns.push_back(appendPattern(fifo, true));
    for (std::uint32_t fifo = 0; fifo != declaration.registerFifoCount; ++fifo)
      combined.usePatterns.push_back(appendPattern(fifo, false));
  }

  combined.eligibilityCount = *eligibilityCount;
  combined.eventCount = *eventCount;
  bool dispatchContended = false;
  for (const Span span : dispatchUnitSpans)
    dispatchContended = dispatchContended || span.count > 1;
  if (combined.grantPolicy) {
    appendRequesters(*combined.grantPolicy, dispatchRequesterOffset,
                     dispatchCandidateCount);
    appendRequesters(*combined.grantPolicy, *registerRequesterOffset,
                     declaration.registerFifoCount);
  } else if (dispatchContended) {
    std::vector<RequesterKey> cycle;
    cycle.reserve(*requesterCount);
    for (std::uint32_t requester = 0; requester != *requesterCount; ++requester)
      cycle.emplace_back(requester);
    combined.grantPolicy = GrantPolicyDeclaration(
        RoundRobinDeclaration{std::move(cycle), RequesterKey(0)});
  }

  auto contract = ResourceContract::create(combined);
  if (!contract)
    return contract.takeError();
  return TemporalPeResourceContract(
      std::move(*contract), std::move(dispatchCandidates),
      std::move(dispatchUnitCandidates), std::move(dispatchUnitSpans),
      dispatchStateOffset, dispatchRequesterOffset, dispatchPatternOffset,
      declaration.registerFifoCount, *registerStateOffset,
      *registerPatternOffset);
}

llvm::ArrayRef<std::uint32_t>
TemporalPeResourceContract::dispatchCandidatesOf(std::uint32_t unit) const {
  assert(unit < dispatchUnitSpans_.size() &&
         "context-dispatch unit ordinal out of range");
  const Span span = dispatchUnitSpans_[unit];
  return llvm::ArrayRef(dispatchUnitCandidates_).slice(span.first, span.count);
}

StateKey TemporalPeResourceContract::dispatchState(std::uint32_t unit) const {
  assert(unit < dispatchUnitSpans_.size() &&
         "context-dispatch unit ordinal out of range");
  return StateKey(dispatchStateOffset_ + unit);
}

RequesterKey
TemporalPeResourceContract::dispatchRequester(std::uint32_t candidate) const {
  assert(candidate < dispatchCandidates_.size() &&
         "context-dispatch candidate ordinal out of range");
  return RequesterKey(dispatchRequesterOffset_ + candidate);
}

UsePatternKey
TemporalPeResourceContract::dispatchPattern(std::uint32_t candidate) const {
  assert(candidate < dispatchCandidates_.size() &&
         "context-dispatch candidate ordinal out of range");
  return UsePatternKey(dispatchPatternOffset_ + candidate);
}

StateKey
TemporalPeResourceContract::registerFifoState(std::uint32_t fifo) const {
  assert(fifo < registerFifoCount_ && "register FIFO ordinal out of range");
  return StateKey(registerStateOffset_ + fifo);
}

UsePatternKey
TemporalPeResourceContract::registerFifoWritePattern(std::uint32_t fifo) const {
  assert(fifo < registerFifoCount_ && "register FIFO ordinal out of range");
  return UsePatternKey(registerPatternOffset_ + fifo);
}

UsePatternKey
TemporalPeResourceContract::registerFifoReadPattern(std::uint32_t fifo) const {
  assert(fifo < registerFifoCount_ && "register FIFO ordinal out of range");
  return UsePatternKey(registerPatternOffset_ + registerFifoCount_ + fifo);
}

llvm::Expected<loom::fabric::FabricUsePatternRef>
fabric::resolveTemporalPeOperandQueuePattern(
    const loom::fabric::FabricArtifactView &view,
    loom::fabric::InstructionContextRef context,
    loom::fabric::FabricFuOccurrenceRef fu, loom::fabric::FabricOrdinal fuInput,
    TemporalOperandQueueUse use) {
  using namespace loom::fabric;

  if (llvm::Error error = validateFabricRef(view, context))
    return std::move(error);
  if (llvm::Error error = validateFabricRef(view, fu))
    return std::move(error);
  if (view.peSchedule(context.pe) != Schedule::Temporal)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "logical operand queue owner is not a temporal PE");
  if (view.parentPeOf(fu) != context.pe)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "logical operand queue FU belongs to a different PE");

  std::uint64_t inputsPerContext = 0;
  std::uint64_t inputPrefix = 0;
  bool foundFu = false;
  for (FabricFuOccurrenceRef candidate : view.fuOccurrences()) {
    if (view.parentPeOf(candidate) != context.pe)
      continue;
    std::uint64_t candidateInputs = 0;
    const FabricTransportEndpointOwnerRef owner =
        FabricTransportEndpointOwnerRef::of(candidate);
    for (FabricOrdinal ordinal = 0;
         ordinal != view.transportEndpointCount(owner); ++ordinal) {
      const FabricTransportEndpointRef endpoint{owner, ordinal};
      if (view.transportEndpointDirection(endpoint) ==
          FabricPortDirection::Input)
        ++candidateInputs;
    }
    if (candidate == fu) {
      foundFu = true;
      if (fuInput >= candidateInputs)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "logical operand queue input is outside the FU domain");
      inputPrefix = inputsPerContext;
    }
    inputsPerContext += candidateInputs;
  }
  if (!foundFu)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "logical operand queue FU is absent from its PE inventory");

  const std::uint64_t contextCount = view.peResidentContextCount(context.pe);
  if (context.ordinal >= contextCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "logical operand queue context is outside the PE domain");
  if (inputsPerContext == 0 ||
      inputsPerContext > std::numeric_limits<std::uint32_t>::max() ||
      contextCount > std::numeric_limits<std::uint32_t>::max() ||
      contextCount * inputsPerContext >
          std::numeric_limits<std::uint32_t>::max())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "logical operand queue domain exceeds the owner key domain");

  const std::uint32_t queueCount =
      static_cast<std::uint32_t>(contextCount * inputsPerContext);
  const std::uint32_t queue = static_cast<std::uint32_t>(
      context.ordinal * inputsPerContext + inputPrefix + fuInput);
  const std::uint32_t patternOrdinal =
      queue + (use == TemporalOperandQueueUse::Dequeue ? queueCount : 0);
  const FabricInventoryOwnerRef peOwner =
      FabricInventoryOwnerRef::of(context.pe);
  const ResourceContract *contract = view.resourceContract(peOwner);
  if (!contract || patternOrdinal >= contract->usePatternCount())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "logical operand queue pattern is absent from the PE contract");
  const UsePattern pattern =
      contract->usePattern(UsePatternKey(patternOrdinal));
  if (pattern.requester.ordinal() != queue || pattern.claims.size() != 1)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "logical operand queue pattern disagrees with its canonical key");
  return FabricUsePatternRef{FabricUsePatternOwnerRef(peOwner), patternOrdinal};
}

llvm::Expected<loom::fabric::FabricUsePatternRef>
fabric::resolveTemporalPeDispatchPattern(
    const loom::fabric::FabricArtifactView &view,
    loom::fabric::InstructionContextRef context,
    loom::fabric::FabricFuOccurrenceRef fu) {
  using namespace loom::fabric;

  if (llvm::Error error = validateFabricRef(view, context))
    return std::move(error);
  if (llvm::Error error = validateFabricRef(view, fu))
    return std::move(error);
  if (view.peSchedule(context.pe) != Schedule::Temporal)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "context-dispatch owner is not a temporal PE");
  if (view.parentPeOf(fu) != context.pe)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "context-dispatch FU belongs to a different PE");

  std::uint64_t fuCount = 0;
  std::uint64_t fuOrdinal = 0;
  std::uint64_t inputsPerContext = 0;
  bool foundFu = false;
  for (FabricFuOccurrenceRef candidate : view.fuOccurrences()) {
    if (view.parentPeOf(candidate) != context.pe)
      continue;
    if (candidate == fu) {
      fuOrdinal = fuCount;
      foundFu = true;
    }
    ++fuCount;
    const FabricTransportEndpointOwnerRef owner =
        FabricTransportEndpointOwnerRef::of(candidate);
    for (FabricOrdinal endpoint = 0;
         endpoint != view.transportEndpointCount(owner); ++endpoint)
      if (view.transportEndpointDirection({owner, endpoint}) ==
          FabricPortDirection::Input)
        ++inputsPerContext;
  }
  if (!foundFu || fuCount == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "context-dispatch FU is absent from its PE inventory");

  const std::uint64_t contextCount = view.peResidentContextCount(context.pe);
  if (context.ordinal >= contextCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "context-dispatch context is outside the PE domain");
  if (contextCount > std::numeric_limits<std::uint32_t>::max() ||
      inputsPerContext > std::numeric_limits<std::uint32_t>::max() ||
      contextCount * inputsPerContext >
          std::numeric_limits<std::uint32_t>::max() ||
      contextCount * fuCount > std::numeric_limits<std::uint32_t>::max())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "context-dispatch domain exceeds the owner key domain");

  const std::uint64_t queueCount = contextCount * inputsPerContext;
  const std::uint64_t candidate = context.ordinal * fuCount + fuOrdinal;
  const std::uint64_t patternOrdinal64 = 2 * queueCount + candidate;
  if (patternOrdinal64 > std::numeric_limits<std::uint32_t>::max())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "context-dispatch pattern exceeds the owner key domain");
  const std::uint32_t patternOrdinal =
      static_cast<std::uint32_t>(patternOrdinal64);
  const FabricInventoryOwnerRef peOwner =
      FabricInventoryOwnerRef::of(context.pe);
  const ResourceContract *contract = view.resourceContract(peOwner);
  if (!contract || patternOrdinal >= contract->usePatternCount())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "context-dispatch pattern is absent from the PE contract");
  const UsePattern pattern =
      contract->usePattern(UsePatternKey(patternOrdinal));
  if (pattern.requester.ordinal() != queueCount + candidate ||
      pattern.claims.size() != 1 || pattern.commit)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "context-dispatch pattern disagrees with its canonical key");
  return FabricUsePatternRef{FabricUsePatternOwnerRef(peOwner), patternOrdinal};
}
