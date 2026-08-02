#include "Dataflow/IR/DataflowActorSemantics.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>

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
      if (descriptor.ivSource != StreamOutputSource::None)
        value.activeResults.push_back(0);
      if (descriptor.emitPhase)
        value.activeResults.push_back(1);
      cases.push_back(std::move(value));
    }
    return requireArity(inputCount, resultCount, 3, 2, std::move(cases));
  }
  case Schema::DataflowCarry: {
    static constexpr CarryCase transitions[] = {
        CarryCase::Init, CarryCase::Next, CarryCase::Close};
    for (CarryCase transition : transitions) {
      const CarryCaseDescriptor descriptor = carryCaseDescriptor(transition);
      cases.push_back(ActorHandshakeCase{
          static_cast<std::uint32_t>(transition),
          ordinalsFromMask(descriptor.consumedInputs, inputCount),
          descriptor.forwardedInput ? llvm::SmallVector<std::uint32_t, 4>{0}
                                    : llvm::SmallVector<std::uint32_t, 4>{}});
    }
    return requireArity(inputCount, resultCount, 3, 1, std::move(cases));
  }
  case Schema::DataflowInvariant: {
    static constexpr InvariantCase transitions[] = {
        InvariantCase::Init, InvariantCase::Replay, InvariantCase::Close};
    for (InvariantCase transition : transitions) {
      const InvariantCaseDescriptor descriptor =
          invariantCaseDescriptor(transition);
      cases.push_back(ActorHandshakeCase{
          static_cast<std::uint32_t>(transition),
          ordinalsFromMask(descriptor.consumedInputs, inputCount),
          descriptor.output != InvariantOutputSource::None
              ? llvm::SmallVector<std::uint32_t, 4>{0}
              : llvm::SmallVector<std::uint32_t, 4>{}});
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
      if (descriptor.emitPhase)
        value.activeResults.push_back(0);
      if (descriptor.forwardedInput)
        value.activeResults.push_back(1);
      cases.push_back(std::move(value));
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
      if (transition.emitGroup) {
        value.activeResults.push_back(0);
        value.activeResults.push_back(1);
      }
      if (transition.emitTruePhase || transition.emitFalsePhase)
        value.activeResults.push_back(2);
      cases.push_back(std::move(value));
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
      if (transition.emitActiveItems) {
        value.activeResults.push_back(0);
        value.activeResults.push_back(1);
      } else if (transition.emitFalsePhase) {
        value.activeResults.push_back(1);
      }
      cases.push_back(std::move(value));
    }
    return requireArity(inputCount, resultCount, 3, 2, std::move(cases));
  }
  case Schema::DataflowConstant:
    return requireArity(inputCount, resultCount, 1, 1, {{0, {0}, {0}}});
  case Schema::DataflowSync:
    if (inputCount == 0 || inputCount != resultCount)
      return invalid("dataflow.sync requires equal nonempty input and result "
                     "arity");
    return llvm::SmallVector<ActorHandshakeCase, 4>{
        {0, ordinalRange(inputCount), ordinalRange(resultCount)}};
  case Schema::DataflowMux:
    if (inputCount < 3 || resultCount != 1)
      return invalid("dataflow.mux requires a selector, at least two data "
                     "inputs, and one result");
    for (std::uint32_t lane = 0; lane + 1 < inputCount; ++lane)
      cases.push_back({lane, {0, lane + 1}, {0}});
    return cases;
  case Schema::DataflowDemux:
    if (inputCount != 2 || resultCount < 2)
      return invalid("dataflow.demux requires a selector, one data input, and "
                     "at least two results");
    for (std::uint32_t lane = 0; lane < resultCount; ++lane)
      cases.push_back({lane, {0, 1}, {lane}});
    return cases;
  default:
    return llvm::SmallVector<ActorHandshakeCase, 4>{
        {0, ordinalRange(inputCount), ordinalRange(resultCount)}};
  }
}

} // namespace dataflow::semantics
