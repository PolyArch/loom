#include "Dataflow/IR/DataflowActorSemantics.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace dataflow::semantics {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_handshake_invalid: " + message);
}

llvm::SmallVector<std::uint32_t, 4> ordinalRange(std::uint32_t count) {
  llvm::SmallVector<std::uint32_t, 4> result;
  result.reserve(count);
  for (std::uint32_t ordinal = 0; ordinal < count; ++ordinal)
    result.push_back(ordinal);
  return result;
}

llvm::SmallVector<std::uint32_t, 4> ordinalsFromMask(SemanticInputMask mask,
                                                     std::uint32_t inputCount) {
  llvm::SmallVector<std::uint32_t, 4> result;
  for (std::uint32_t ordinal = 0; ordinal < inputCount; ++ordinal)
    if ((mask & (SemanticInputMask{1} << ordinal)) != 0)
      result.push_back(ordinal);
  return result;
}

ActorResultProductionGroup
onceGroup(llvm::SmallVector<std::uint32_t, 4> activeResults) {
  return ActorResultProductionGroup{std::move(activeResults),
                                    ActorResultProductionOnce{}};
}

ActorResultProductionGroup
repeatedLaneGroup(std::uint32_t maskInputOrdinal,
                  llvm::SmallVector<std::uint32_t, 4> activeResults) {
  return ActorResultProductionGroup{
      std::move(activeResults),
      ActorResultProductionForEachDefinedOneLane{maskInputOrdinal}};
}

ActorHandshakeCase
makeCase(std::uint32_t ordinal,
         llvm::SmallVector<std::uint32_t, 4> consumedInputs,
         llvm::SmallVector<ActorResultProductionGroup, 2> productionGroups) {
  ActorHandshakeCase result;
  result.ordinal = ordinal;
  result.consumedInputs = std::move(consumedInputs);
  result.productionGroups = std::move(productionGroups);
  for (const ActorResultProductionGroup &group : result.productionGroups)
    result.activeResults.append(group.activeResults.begin(),
                                group.activeResults.end());
  llvm::sort(result.activeResults);
  result.activeResults.erase(
      std::unique(result.activeResults.begin(), result.activeResults.end()),
      result.activeResults.end());
  return result;
}

ActorHandshakeCase
oneGroupCase(std::uint32_t ordinal,
             llvm::SmallVector<std::uint32_t, 4> consumedInputs,
             llvm::SmallVector<std::uint32_t, 4> activeResults) {
  llvm::SmallVector<ActorResultProductionGroup, 2> groups;
  if (!activeResults.empty())
    groups.push_back(onceGroup(std::move(activeResults)));
  return makeCase(ordinal, std::move(consumedInputs), std::move(groups));
}

llvm::Expected<llvm::SmallVector<ActorHandshakeCase, 4>>
requireArity(std::uint32_t inputCount, std::uint32_t resultCount,
             std::uint32_t expectedInputs, std::uint32_t expectedResults,
             llvm::SmallVector<ActorHandshakeCase, 4> cases) {
  if (inputCount != expectedInputs || resultCount != expectedResults)
    return invalid("registered actor schema has an invalid function arity");
  return cases;
}

} // namespace

llvm::Expected<llvm::SmallVector<ActorHandshakeCase, 4>>
projectActorHandshakeCases(::dataflow::OperationSchemaId schema,
                           std::uint32_t inputCount,
                           std::uint32_t resultCount) {
  using Schema = ::dataflow::OperationSchemaId;
  llvm::SmallVector<ActorHandshakeCase, 4> cases;

  switch (schema) {
  case Schema::DataflowStream: {
    static constexpr StreamCase transitions[] = {
        StreamCase::StartTrue, StreamCase::StartClose, StreamCase::ContinueTrue,
        StreamCase::ContinueClose};
    for (StreamCase transition : transitions) {
      const StreamCaseDescriptor descriptor = streamCaseDescriptor(transition);
      ActorHandshakeCase value;
      value.ordinal = static_cast<std::uint32_t>(transition);
      value.consumedInputs =
          ordinalsFromMask(descriptor.consumedInputs, inputCount);
      llvm::SmallVector<std::uint32_t, 4> activeResults;
      if (descriptor.ivSource != StreamOutputSource::None)
        activeResults.push_back(0);
      if (descriptor.emitPhase)
        activeResults.push_back(1);
      cases.push_back(oneGroupCase(value.ordinal,
                                   std::move(value.consumedInputs),
                                   std::move(activeResults)));
    }
    return requireArity(inputCount, resultCount, 3, 2, std::move(cases));
  }
  case Schema::DataflowCarry: {
    static constexpr CarryCase transitions[] = {
        CarryCase::Init, CarryCase::Next, CarryCase::Close};
    for (CarryCase transition : transitions) {
      const CarryCaseDescriptor descriptor = carryCaseDescriptor(transition);
      cases.push_back(oneGroupCase(
          static_cast<std::uint32_t>(transition),
          ordinalsFromMask(descriptor.consumedInputs, inputCount),
          descriptor.forwardedInput ? llvm::SmallVector<std::uint32_t, 4>{0}
                                    : llvm::SmallVector<std::uint32_t, 4>{}));
    }
    return requireArity(inputCount, resultCount, 3, 1, std::move(cases));
  }
  case Schema::DataflowInvariant: {
    static constexpr InvariantCase transitions[] = {
        InvariantCase::Init, InvariantCase::Replay, InvariantCase::Close};
    for (InvariantCase transition : transitions) {
      const InvariantCaseDescriptor descriptor =
          invariantCaseDescriptor(transition);
      cases.push_back(
          oneGroupCase(static_cast<std::uint32_t>(transition),
                       ordinalsFromMask(descriptor.consumedInputs, inputCount),
                       descriptor.output != InvariantOutputSource::None
                           ? llvm::SmallVector<std::uint32_t, 4>{0}
                           : llvm::SmallVector<std::uint32_t, 4>{}));
    }
    return requireArity(inputCount, resultCount, 2, 1, std::move(cases));
  }
  case Schema::DataflowGate: {
    static constexpr GateCase transitions[] = {
        GateCase::ClosedDrop, GateCase::FirstTrue, GateCase::ContinueTrue,
        GateCase::Close};
    for (GateCase transition : transitions) {
      const GateCaseDescriptor descriptor = gateCaseDescriptor(transition);
      ActorHandshakeCase value;
      value.ordinal = static_cast<std::uint32_t>(transition);
      value.consumedInputs =
          ordinalsFromMask(descriptor.consumedInputs, inputCount);
      llvm::SmallVector<std::uint32_t, 4> activeResults;
      if (descriptor.emitPhase)
        activeResults.push_back(0);
      if (descriptor.forwardedInput)
        activeResults.push_back(1);
      cases.push_back(oneGroupCase(value.ordinal,
                                   std::move(value.consumedInputs),
                                   std::move(activeResults)));
    }
    return requireArity(inputCount, resultCount, 2, 2, std::move(cases));
  }
  case Schema::DataflowParallelize: {
    const ParallelizeTransition accumulate = evaluateParallelizeTransition(
        ParallelizeSemanticState{0}, 2, true, true);
    const ParallelizeTransition full = evaluateParallelizeTransition(
        ParallelizeSemanticState{1}, 2, true, true);
    const ParallelizeTransition emptyClose = evaluateParallelizeTransition(
        ParallelizeSemanticState{0}, 2, false, false);
    const ParallelizeTransition tailFlush = evaluateParallelizeTransition(
        ParallelizeSemanticState{1}, 2, false, false);
    const ParallelizeTransition transitions[] = {accumulate, full, emptyClose,
                                                 tailFlush};
    for (auto [ordinal, transition] : llvm::enumerate(transitions)) {
      ActorHandshakeCase value;
      value.ordinal = static_cast<std::uint32_t>(ordinal);
      value.consumedInputs =
          ordinalsFromMask(transition.firing.consumedInputs, inputCount);
      llvm::SmallVector<ActorResultProductionGroup, 2> groups;
      if (transition.emitGroup) {
        llvm::SmallVector<std::uint32_t, 4> groupResults{0, 1};
        if (transition.emitTruePhase)
          groupResults.push_back(2);
        groups.push_back(onceGroup(std::move(groupResults)));
      }
      if (transition.emitFalsePhase)
        groups.push_back(onceGroup({2}));
      cases.push_back(makeCase(value.ordinal, std::move(value.consumedInputs),
                               std::move(groups)));
    }
    return requireArity(inputCount, resultCount, 2, 3, std::move(cases));
  }
  case Schema::DataflowSerialize: {
    const SerializeTransition active =
        evaluateSerializeTransition(true, true, true);
    const SerializeTransition close =
        evaluateSerializeTransition(false, false, false);
    const SerializeTransition transitions[] = {active, close};
    for (auto [ordinal, transition] : llvm::enumerate(transitions)) {
      ActorHandshakeCase value;
      value.ordinal = static_cast<std::uint32_t>(ordinal);
      value.consumedInputs =
          ordinalsFromMask(transition.firing.consumedInputs, inputCount);
      llvm::SmallVector<ActorResultProductionGroup, 2> groups;
      if (transition.emitActiveItems) {
        groups.push_back(repeatedLaneGroup(
            static_cast<std::uint32_t>(SerializeInput::Mask), {0, 1}));
      } else if (transition.emitFalsePhase) {
        groups.push_back(onceGroup({1}));
      }
      cases.push_back(makeCase(value.ordinal, std::move(value.consumedInputs),
                               std::move(groups)));
    }
    return requireArity(inputCount, resultCount, 3, 2, std::move(cases));
  }
  case Schema::DataflowConstant:
    cases.push_back(oneGroupCase(0, {0}, {0}));
    return requireArity(inputCount, resultCount, 1, 1, std::move(cases));
  case Schema::DataflowSync:
    if (inputCount == 0 || inputCount != resultCount)
      return invalid("dataflow.sync requires equal nonempty input and result "
                     "arity");
    return llvm::SmallVector<ActorHandshakeCase, 4>{
        oneGroupCase(0, ordinalRange(inputCount), ordinalRange(resultCount))};
  case Schema::DataflowMux:
    if (inputCount < 3 || resultCount != 1)
      return invalid("dataflow.mux requires a selector, at least two data "
                     "inputs, and one result");
    for (std::uint32_t lane = 0; lane + 1 < inputCount; ++lane)
      cases.push_back(oneGroupCase(lane, {0, lane + 1}, {0}));
    return cases;
  case Schema::DataflowDemux:
    if (inputCount != 2 || resultCount < 2)
      return invalid("dataflow.demux requires a selector, one data input, and "
                     "at least two results");
    for (std::uint32_t lane = 0; lane < resultCount; ++lane)
      cases.push_back(oneGroupCase(lane, {0, 1}, {lane}));
    return cases;
  default:
    return llvm::SmallVector<ActorHandshakeCase, 4>{
        oneGroupCase(0, ordinalRange(inputCount), ordinalRange(resultCount))};
  }
}

llvm::Expected<llvm::SmallVector<InitializedFeedbackInputDescriptor, 3>>
projectActorInitializedFeedbackInputs(::dataflow::OperationSchemaId schema,
                                      std::uint32_t inputCount,
                                      std::uint32_t resultCount) {
  auto cases = projectActorHandshakeCases(schema, inputCount, resultCount);
  if (!cases)
    return cases.takeError();

  using Schema = ::dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::DataflowCarry:
    return llvm::SmallVector<InitializedFeedbackInputDescriptor, 3>{
        {static_cast<std::uint32_t>(CarryInput::Phase), std::nullopt},
        {static_cast<std::uint32_t>(CarryInput::Next), std::uint64_t{1}}};
  case Schema::DataflowInvariant:
    return llvm::SmallVector<InitializedFeedbackInputDescriptor, 3>{
        {static_cast<std::uint32_t>(InvariantInput::Phase), std::nullopt}};
  default:
    return llvm::SmallVector<InitializedFeedbackInputDescriptor, 3>{};
  }
}

} // namespace dataflow::semantics
