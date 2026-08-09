#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <system_error>
#include <variant>

namespace {

template <typename Case> fabric::UsePatternKey usePattern(Case transition) {
  return fabric::UsePatternKey(static_cast<std::uint32_t>(transition));
}

template <typename Case>
fabric::ResourceContract createLoopControlContract(llvm::ArrayRef<Case> cases,
                                                   bool registeredLatency) {
  using namespace fabric;
  ResourceContractDeclaration declaration;
  declaration.states = {ResourceStateDeclaration{StateKey(0), {}}};
  if (registeredLatency)
    declaration.states.push_back(ResourceStateDeclaration{
        StateKey(1),
        {CapacityDimensionDeclaration{CapacityDimensionKey(0), CapacityUnits(1),
                                      CapacityUnits(0)}}});

  declaration.resourceTransitions.reserve(cases.size());
  declaration.usePatterns.reserve(cases.size());
  declaration.requesters = {RequesterKey(0)};
  declaration.eligibilityCount = cases.size();
  declaration.eventCount = registeredLatency ? 2 : 1;
  declaration.timingContracts = {TimingContractDeclaration{
      TimingContractKey(0), registeredLatency ? std::vector<std::uint32_t>{0, 1}
                                              : std::vector<std::uint32_t>{0}}};

  for (Case transition : cases) {
    const std::uint32_t ordinal = static_cast<std::uint32_t>(transition);
    declaration.resourceTransitions.push_back(ResourceTransitionKey(ordinal));
    std::vector<ClaimDeclaration> claims;
    if (registeredLatency)
      claims.push_back(ClaimDeclaration{
          ClaimKey(0), StateKey(1), CapacityDimensionKey(0), CapacityUnits(1)});
    declaration.usePatterns.push_back(UsePatternDeclaration{
        UsePatternKey(ordinal),
        RequesterKey(0),
        EligibilityKey(ordinal),
        EventKey(0),
        EventKey(registeredLatency ? 1 : 0),
        CommitDeclaration{EventKey(registeredLatency ? 1 : 0),
                          ResourceTransitionKey(ordinal)},
        TimingContractKey(0),
        std::move(claims),
        {}});
  }
  return llvm::cantFail(ResourceContract::create(declaration));
}

fabric::ResourceContract createOneCycleElasticContract() {
  using namespace fabric;
  ResourceContractDeclaration declaration;
  declaration.states = {ResourceStateDeclaration{
      StateKey(0),
      {CapacityDimensionDeclaration{CapacityDimensionKey(0), CapacityUnits(1),
                                    CapacityUnits(0)}}}};
  declaration.resourceTransitions = {ResourceTransitionKey(0)};
  declaration.timingContracts = {
      TimingContractDeclaration{TimingContractKey(0), {0, 1, 1}}};
  declaration.usePatterns = {UsePatternDeclaration{
      UsePatternKey(0),
      RequesterKey(0),
      EligibilityKey(0),
      EventKey(0),
      EventKey(2),
      CommitDeclaration{EventKey(1), ResourceTransitionKey(0)},
      TimingContractKey(0),
      {ClaimDeclaration{ClaimKey(0), StateKey(0), CapacityDimensionKey(0),
                        CapacityUnits(1)}},
      {}}};
  declaration.requesters = {RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 3;
  return llvm::cantFail(ResourceContract::create(declaration));
}

llvm::Expected<fabric::ResourceContract>
createOrderedCardinalityContract(dataflow::OperationSchemaId schema,
                                 std::uint32_t maximumLaneCount) {
  using namespace dataflow::semantics;
  using namespace fabric;
  if (maximumLaneCount == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ordered-cardinality maximum lane count must be positive");

  std::uint32_t inputCount = 0;
  std::uint32_t resultCount = 0;
  switch (schema) {
  case dataflow::OperationSchemaId::DataflowParallelize:
    inputCount = 2;
    resultCount = 3;
    break;
  case dataflow::OperationSchemaId::DataflowSerialize:
    inputCount = 3;
    resultCount = 2;
    break;
  default:
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ordered-cardinality contract requires parallelize or serialize");
  }

  auto cases = projectActorHandshakeCases(schema, inputCount, resultCount);
  if (!cases)
    return cases.takeError();

  ResourceContractDeclaration declaration;
  declaration.states = {
      ResourceStateDeclaration{
          StateKey(0),
          {CapacityDimensionDeclaration{CapacityDimensionKey(0),
                                        CapacityUnits(1), CapacityUnits(0)}}},
      ResourceStateDeclaration{
          StateKey(1),
          {CapacityDimensionDeclaration{CapacityDimensionKey(0),
                                        CapacityUnits(1), CapacityUnits(0)}}},
  };
  declaration.requesters = {RequesterKey(0)};
  declaration.eligibilityCount = cases->size();
  declaration.eventCount = 3;
  declaration.timingContracts = {
      TimingContractDeclaration{TimingContractKey(0), {0, 1, 1}}};
  declaration.resourceTransitions.reserve(cases->size());
  declaration.usePatterns.reserve(cases->size());

  for (const ActorHandshakeCase &handshake : *cases) {
    std::uint64_t transactionCount = 0;
    for (const ActorResultProductionGroup &group : handshake.productionGroups) {
      const std::uint64_t increment =
          std::holds_alternative<ActorResultProductionOnce>(group.repeat)
              ? 1
              : maximumLaneCount;
      if (transactionCount >
          std::numeric_limits<std::uint32_t>::max() - increment)
        return llvm::createStringError(
            std::errc::value_too_large,
            "ordered-cardinality transaction inventory exceeds uint32");
      transactionCount += increment;
    }

    declaration.resourceTransitions.push_back(
        ResourceTransitionKey(handshake.ordinal));
    std::vector<ClaimDeclaration> claims = {ClaimDeclaration{
        ClaimKey(0), StateKey(0), CapacityDimensionKey(0), CapacityUnits(1)}};
    std::vector<InternalTransactionDeclaration> transactions;
    transactions.reserve(static_cast<std::size_t>(transactionCount));
    if (transactionCount != 0) {
      claims.push_back(ClaimDeclaration{
          ClaimKey(1), StateKey(1), CapacityDimensionKey(0), CapacityUnits(1)});
      for (std::uint64_t transaction = 0; transaction != transactionCount;
           ++transaction)
        transactions.push_back(InternalTransactionDeclaration{{ClaimKey(1)}});
    }
    declaration.usePatterns.push_back(UsePatternDeclaration{
        UsePatternKey(handshake.ordinal),
        RequesterKey(0),
        EligibilityKey(handshake.ordinal),
        EventKey(0),
        EventKey(2),
        CommitDeclaration{EventKey(1),
                          ResourceTransitionKey(handshake.ordinal)},
        TimingContractKey(0),
        std::move(claims),
        std::move(transactions),
        {},
        {}});
  }
  return ResourceContract::create(std::move(declaration));
}

} // namespace

const fabric::ResourceContract &
fabric::oneCycleElasticOperationResourceContract() {
  static const ResourceContract contract = createOneCycleElasticContract();
  return contract;
}

llvm::Expected<bool> fabric::isOneCycleElasticOperationResourceContract(
    const ResourceContract &contract) {
  auto actual = encodeResourceContractRecord(contract);
  if (!actual)
    return actual.takeError();
  auto expected =
      encodeResourceContractRecord(oneCycleElasticOperationResourceContract());
  if (!expected)
    return expected.takeError();
  return *actual == *expected;
}

llvm::Expected<fabric::ResourceContract>
fabric::createOrderedCardinalityOperationResourceContract(
    ::dataflow::OperationSchemaId schema, std::uint32_t maximumLaneCount) {
  return createOrderedCardinalityContract(schema, maximumLaneCount);
}

llvm::Expected<bool> fabric::isOrderedCardinalityOperationResourceContract(
    const ResourceContract &contract, ::dataflow::OperationSchemaId schema,
    std::uint32_t maximumLaneCount) {
  auto expected = createOrderedCardinalityContract(schema, maximumLaneCount);
  if (!expected)
    return expected.takeError();
  auto actualBytes = encodeResourceContractRecord(contract);
  if (!actualBytes)
    return actualBytes.takeError();
  auto expectedBytes = encodeResourceContractRecord(*expected);
  if (!expectedBytes)
    return expectedBytes.takeError();
  return *actualBytes == *expectedBytes;
}

const fabric::ResourceContract &fabric::loopStreamOperationResourceContract() {
  using Case = ::dataflow::semantics::StreamCase;
  static constexpr Case cases[] = {Case::StartTrue, Case::StartClose,
                                   Case::ContinueTrue, Case::ContinueClose};
  static const ResourceContract contract =
      createLoopControlContract<Case>(cases, true);
  return contract;
}

llvm::Expected<bool>
fabric::requiresActiveResultHandoff(const ResourceContract &contract) {
  auto oneCycle = isOneCycleElasticOperationResourceContract(contract);
  if (!oneCycle || *oneCycle)
    return oneCycle;

  auto actual = encodeResourceContractRecord(contract);
  if (!actual)
    return actual.takeError();
  auto loopStream =
      encodeResourceContractRecord(loopStreamOperationResourceContract());
  if (!loopStream)
    return loopStream.takeError();
  return *actual == *loopStream;
}

const fabric::ResourceContract &fabric::loopCarryOperationResourceContract() {
  using Case = ::dataflow::semantics::CarryCase;
  static constexpr Case cases[] = {Case::Init, Case::Next, Case::Close};
  static const ResourceContract contract =
      createLoopControlContract<Case>(cases, false);
  return contract;
}

const fabric::ResourceContract &
fabric::loopInvariantOperationResourceContract() {
  using Case = ::dataflow::semantics::InvariantCase;
  static constexpr Case cases[] = {Case::Init, Case::Replay, Case::Close};
  static const ResourceContract contract =
      createLoopControlContract<Case>(cases, false);
  return contract;
}

const fabric::ResourceContract &fabric::loopGateOperationResourceContract() {
  using Case = ::dataflow::semantics::GateCase;
  static constexpr Case cases[] = {Case::ClosedDrop, Case::FirstTrue,
                                   Case::ContinueTrue, Case::Close};
  static const ResourceContract contract =
      createLoopControlContract<Case>(cases, false);
  return contract;
}

llvm::Expected<fabric::UsePatternKey>
fabric::resolveOperationUsePattern(const ResourceContract &contract,
                                   std::uint32_t transitionCaseOrdinal) {
  if (contract.usePatternCount() == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "fabric.op resource contract has no use pattern");
  const std::uint32_t ordinal =
      contract.usePatternCount() == 1 ? 0 : transitionCaseOrdinal;
  if (ordinal >= contract.usePatternCount())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "fabric.op transition case has no resource use pattern");
  return UsePatternKey(ordinal);
}

fabric::UsePatternKey
fabric::loopControlUsePattern(::dataflow::semantics::StreamCase transition) {
  return usePattern(transition);
}

fabric::UsePatternKey
fabric::loopControlUsePattern(::dataflow::semantics::CarryCase transition) {
  return usePattern(transition);
}

fabric::UsePatternKey
fabric::loopControlUsePattern(::dataflow::semantics::InvariantCase transition) {
  return usePattern(transition);
}

fabric::UsePatternKey
fabric::loopControlUsePattern(::dataflow::semantics::GateCase transition) {
  return usePattern(transition);
}
