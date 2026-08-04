#include "DFGSimulatorInternal.h"

#include "llvm/ADT/STLExtras.h"

namespace loom::sim::detail {
namespace {

std::size_t outputCount(const SimulatorState &state, mlir::Value value) {
  auto found = state.observedOutputs.find(value);
  return found == state.observedOutputs.end() ? 0 : found->second.size();
}

mlir::Operation *unclosedStatefulActor(dataflow::GraphOp graph,
                                       SimulatorState &state) {
  mlir::Block &entry = graph.getBody().front();
  for (mlir::Operation &op : entry.without_terminator()) {
    if (auto stream = mlir::dyn_cast<dataflow::StreamOp>(op)) {
      auto found = state.streamStates.find(stream.getOperation());
      if (found != state.streamStates.end() &&
          found->second.mode != StreamMode::Idle)
        return &op;
      continue;
    }
    if (auto carry = mlir::dyn_cast<dataflow::CarryOp>(op)) {
      auto found = state.carryStates.find(carry.getOperation());
      if (found != state.carryStates.end() &&
          found->second.semanticState != PhaseSemanticState::Initial)
        return &op;
      continue;
    }
    if (auto invariant = mlir::dyn_cast<dataflow::InvariantOp>(op)) {
      auto found = state.invariantStates.find(invariant.getOperation());
      if (found != state.invariantStates.end() &&
          (found->second.semanticState != PhaseSemanticState::Initial ||
           found->second.latched.has_value()))
        return &op;
      continue;
    }
    if (auto gate = mlir::dyn_cast<dataflow::GateOp>(op))
      if (state.gateContinueStates.contains(gate.getOperation()))
        return &op;
  }
  return nullptr;
}

bool streamInputsCommitted(dataflow::GraphOp graph,
                           const PreparedGraphExecution &execution,
                           SimulatorState &state) {
  mlir::Block &entry = graph.getBody().front();
  for (unsigned index = 0; index < execution.applicationInputCount; ++index) {
    if (graph.getInputPortKind(index) != dataflow::GraphPortKind::Stream)
      continue;
    mlir::BlockArgument argument = entry.getArgument(index + 1);
    for (mlir::OpOperand &use : argument.getUses())
      if (hasToken(state, use))
        return false;
  }
  return true;
}

} // namespace

bool graphCompletionReady(const PreparedGraphExecution &execution,
                          const SimulatorState &state) {
  return !execution.returnObservation.complete.empty() &&
         llvm::all_of(execution.returnObservation.complete,
                      [&](mlir::Value witness) {
                        return outputCount(state, witness) != 0;
                      });
}

llvm::Error
validateGraphRetirementBoundary(dataflow::GraphOp graph,
                                const PreparedGraphExecution &execution,
                                SimulatorState &state) {
  if (!graphCompletionReady(execution, state))
    return llvm::createStringError(std::errc::state_not_recoverable,
                                   "graph retirement frontier is not visible");
  for (mlir::Value witness : execution.returnObservation.complete)
    if (outputCount(state, witness) != 1)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "completion witness produced multiple tokens before retirement");
  if (!streamInputsCommitted(graph, execution, state))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "graph retired before all stream input tokens were committed");
  if (mlir::Operation *actor = unclosedStatefulActor(graph, state))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "graph retired before stateful actor close/reset: %s",
        actor->getName().getStringRef().str().c_str());
  for (auto [index, value] :
       llvm::enumerate(execution.returnObservation.values)) {
    const std::size_t count = outputCount(state, value);
    if (count != 1)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "value output #%zu produced %zu tokens at retirement",
          static_cast<std::size_t>(index), count);
  }
  return llvm::Error::success();
}

} // namespace loom::sim::detail
