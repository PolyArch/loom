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

ResourceContractDeclaration cloneDeclaration(const ResourceContract &source) {
  ResourceContractDeclaration result;
  result.states.reserve(source.stateCount());
  for (std::uint32_t state = 0; state != source.stateCount(); ++state) {
    ResourceStateDeclaration declaration{StateKey(state), {}};
    for (auto [dimension, capacity] :
         llvm::enumerate(source.capacityDimensions(StateKey(state))))
      declaration.capacityDimensions.push_back(CapacityDimensionDeclaration{
          CapacityDimensionKey(static_cast<std::uint32_t>(dimension)),
          capacity.capacity, capacity.initialOccupancy});
    result.states.push_back(std::move(declaration));
  }

  result.resourceTransitions.reserve(source.resourceTransitionCount());
  for (std::uint32_t transition = 0;
       transition != source.resourceTransitionCount(); ++transition)
    result.resourceTransitions.emplace_back(transition);

  result.timingContracts.reserve(source.timingContractCount());
  for (std::uint32_t timing = 0; timing != source.timingContractCount();
       ++timing)
    result.timingContracts.push_back(TimingContractDeclaration{
        TimingContractKey(timing),
        std::vector<std::uint32_t>(
            source.eventOrder(TimingContractKey(timing)).begin(),
            source.eventOrder(TimingContractKey(timing)).end())});

  result.requesters.reserve(source.requesterCount());
  for (std::uint32_t requester = 0; requester != source.requesterCount();
       ++requester)
    result.requesters.emplace_back(requester);
  result.eligibilityCount = source.eligibilityCount();
  result.eventCount = source.eventCount();

  result.usePatterns.reserve(source.usePatternCount());
  for (std::uint32_t ordinal = 0; ordinal != source.usePatternCount();
       ++ordinal) {
    const UsePattern pattern = source.usePattern(UsePatternKey(ordinal));
    UsePatternDeclaration declaration{
        UsePatternKey(ordinal),
        pattern.requester,
        pattern.eligibility,
        pattern.acquire,
        pattern.release,
        pattern.commit ? std::optional<CommitDeclaration>(CommitDeclaration{
                             pattern.commit->event, pattern.commit->transition})
                       : std::nullopt,
        pattern.timingAndProgress,
        {},
        {}};
    declaration.claims.reserve(pattern.claims.size());
    for (auto [claimOrdinal, claim] : llvm::enumerate(pattern.claims))
      declaration.claims.push_back(
          ClaimDeclaration{ClaimKey(static_cast<std::uint32_t>(claimOrdinal)),
                           claim.state, claim.dimension, claim.amount});
    declaration.internalTransactions.reserve(pattern.internalTransactionCount);
    for (std::uint32_t transaction = 0;
         transaction != pattern.internalTransactionCount; ++transaction) {
      llvm::ArrayRef<ClaimKey> claims =
          source.internalTransaction(UsePatternKey(ordinal), transaction);
      declaration.internalTransactions.push_back(
          {std::vector<ClaimKey>(claims.begin(), claims.end())});
    }
    result.usePatterns.push_back(std::move(declaration));
  }

  if (std::optional<GrantPolicyView> policy = source.grantPolicy())
    result.grantPolicy = std::visit(
        [](const auto &view) -> GrantPolicyDeclaration {
          using View = std::decay_t<decltype(view)>;
          if constexpr (std::is_same_v<View, FixedPriorityView>) {
            return FixedPriorityDeclaration{std::vector<RequesterKey>(
                view.requesterOrder().begin(), view.requesterOrder().end())};
          } else {
            return RoundRobinDeclaration{
                std::vector<RequesterKey>(view.requesterCycle().begin(),
                                          view.requesterCycle().end()),
                view.resetCursor()};
          }
        },
        *policy);
  return result;
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

  auto operandBuffer =
      TemporalOperandBufferContract::create(TemporalOperandBufferDeclaration{
          declaration.pe, declaration.contextCount, declaration.fuInputCounts,
          declaration.operandBufferMode,
          declaration.operandEntriesPerAllocationUnit});
  if (!operandBuffer)
    return operandBuffer.takeError();

  ResourceContractDeclaration combined =
      cloneDeclaration(operandBuffer->resourceContract());
  const std::uint32_t stateOffset =
      operandBuffer->resourceContract().stateCount();
  const std::uint32_t transitionOffset =
      operandBuffer->resourceContract().resourceTransitionCount();
  const std::uint32_t timingOffset =
      operandBuffer->resourceContract().timingContractCount();
  const std::uint32_t patternOffset =
      operandBuffer->resourceContract().usePatternCount();
  const std::uint32_t requesterOffset =
      operandBuffer->resourceContract().requesterCount();
  const std::uint32_t eligibilityOffset =
      operandBuffer->resourceContract().eligibilityCount();
  const std::uint32_t eventOffset =
      operandBuffer->resourceContract().eventCount();

  auto stateCount = checkedAdd(stateOffset, declaration.registerFifoCount,
                               "resource-state inventory");
  auto registerActionCount =
      checkedAdd(declaration.registerFifoCount, declaration.registerFifoCount,
                 "register FIFO action inventory");
  if (!registerActionCount)
    return registerActionCount.takeError();
  auto transitionCount = checkedAdd(transitionOffset, *registerActionCount,
                                    "resource-transition inventory");
  auto patternCount =
      checkedAdd(patternOffset, *registerActionCount, "use-pattern inventory");
  auto requesterCount = checkedAdd(
      requesterOffset, declaration.registerFifoCount, "requester inventory");
  auto eligibilityCount =
      checkedAdd(eligibilityOffset, declaration.registerFifoCount ? 2 : 0,
                 "eligibility inventory");
  auto eventCount = checkedAdd(
      eventOffset, declaration.registerFifoCount ? 3 : 0, "event inventory");
  auto timingCount =
      checkedAdd(timingOffset, declaration.registerFifoCount ? 1 : 0,
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

  for (std::uint32_t fifo = 0; fifo != declaration.registerFifoCount; ++fifo) {
    ResourceStateDeclaration state{
        StateKey(stateOffset + fifo),
        {{occupiedEntry, CapacityUnits(declaration.registerFifoDepth),
          CapacityUnits(0)},
         {firstPortService, CapacityUnits(1), CapacityUnits(0)}}};
    if (declaration.registerFifoPorts == 2)
      state.capacityDimensions.push_back(
          {secondPortService, CapacityUnits(1), CapacityUnits(0)});
    combined.states.push_back(std::move(state));
    combined.resourceTransitions.emplace_back(transitionOffset + fifo);
    combined.resourceTransitions.emplace_back(
        transitionOffset + declaration.registerFifoCount + fifo);
    combined.requesters.emplace_back(requesterOffset + fifo);
  }

  if (declaration.registerFifoCount != 0) {
    std::vector<std::uint32_t> eventRanks(*eventCount, 0);
    eventRanks[eventOffset] = 0;
    eventRanks[eventOffset + 1] = 0;
    eventRanks[eventOffset + 2] = 1;
    combined.timingContracts.push_back(
        {TimingContractKey(timingOffset), std::move(eventRanks)});

    const auto appendPattern = [&](std::uint32_t fifo,
                                   bool write) -> UsePatternDeclaration {
      const StateKey state(stateOffset + fifo);
      const CapacityDimensionKey service =
          write || declaration.registerFifoPorts == 1 ? firstPortService
                                                      : secondPortService;
      const std::uint32_t role = write ? 0 : 1;
      const ResourceTransitionKey transition(
          transitionOffset + role * declaration.registerFifoCount + fifo);
      return UsePatternDeclaration{
          UsePatternKey(patternOffset + role * declaration.registerFifoCount +
                        fifo),
          RequesterKey(requesterOffset + fifo),
          EligibilityKey(eligibilityOffset + role),
          EventKey(eventOffset + role),
          EventKey(eventOffset + 2),
          CommitDeclaration{EventKey(eventOffset + role), transition},
          TimingContractKey(timingOffset),
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
  if (combined.grantPolicy)
    appendRequesters(*combined.grantPolicy, requesterOffset,
                     declaration.registerFifoCount);

  auto contract = ResourceContract::create(combined);
  if (!contract)
    return contract.takeError();
  return TemporalPeResourceContract(std::move(*contract),
                                    declaration.registerFifoCount, stateOffset,
                                    patternOffset);
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
