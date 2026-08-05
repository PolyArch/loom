//===- DFGSimulatorFunctionalObservations.cpp - Typed DFG results --------===//
//
// Projects one retired execution into the exact positional spatial
// observation algebra selected by its SimulationWorkload.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"
#include "SimulationWireInternal.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowOps.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {
namespace {

llvm::Expected<CanonicalValueSequence>
sequenceFromTokens(llvm::ArrayRef<Token> tokens, mlir::Type type,
                   const LaneShape &shape, mlir::Operation *scope) {
  CanonicalValueSequence sequence;
  sequence.tokenCount = tokens.size();
  if (shape.lanesPerToken == 0)
    return sequence;
  if (!tokens.empty() &&
      shape.lanesPerToken >
          std::numeric_limits<std::size_t>::max() / tokens.size())
    return llvm::createStringError(std::errc::value_too_large,
                                   "DFG observation lane count overflows");
  sequence.lanes.reserve(tokens.size() * shape.lanesPerToken);

  if (shape.laneBitWidth == 0 ||
      shape.lanesPerToken >
          std::numeric_limits<std::uint64_t>::max() / shape.laneBitWidth)
    return llvm::createStringError(
        std::errc::value_too_large,
        "DFG observation canonical lane width overflows");
  const std::uint64_t expectedWidth =
      shape.lanesPerToken * static_cast<std::uint64_t>(shape.laneBitWidth);
  if (expectedWidth > std::numeric_limits<unsigned>::max())
    return llvm::createStringError(
        std::errc::not_supported,
        "DFG observation token exceeds the APInt width domain");

  for (const Token &token : tokens) {
    if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
      auto lanes = vectorPrimitiveValues(token, vector, scope);
      if (!lanes)
        return lanes.takeError();
      if (lanes->size() != shape.lanesPerToken)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "DFG vector observation does not match its canonical lane shape");
      for (const PrimitiveValue &lane : *lanes) {
        if (lane.state == PrimitiveValueState::Poison)
          sequence.lanes.push_back(SemanticLane::poison());
        else if (lane.state == PrimitiveValueState::Undef)
          sequence.lanes.push_back(SemanticLane::undef());
        else
          sequence.lanes.push_back(SemanticLane::defined(*lane.bits));
      }
      continue;
    }
    if (token.valueState != PrimitiveValueState::Defined) {
      const SemanticLane lane = token.valueState == PrimitiveValueState::Poison
                                    ? SemanticLane::poison()
                                    : SemanticLane::undef();
      sequence.lanes.insert(sequence.lanes.end(), shape.lanesPerToken, lane);
      continue;
    }
    if (shape.pointerLayout) {
      if (shape.lanesPerToken != 1)
        return llvm::createStringError(
            std::errc::not_supported,
            "DFG observation does not support vector pointer tokens");
      const PointerValue *pointer = token.pointerValue();
      if (token.kind != TokenKind::Pointer || !pointer ||
          pointer->addressSpace != shape.pointerLayout->addressSpace ||
          pointer->representation.getBitWidth() !=
              shape.pointerLayout->representationBits ||
          pointer->byteOffset.getBitWidth() != shape.pointerLayout->addressBits)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "DFG pointer observation does not match its exact pointer type");
      sequence.lanes.push_back(SemanticLane::definedPointer(
          pointer->representation, pointer->objectOrdinal,
          pointer->byteOffset));
      continue;
    }
    llvm::Expected<llvm::APInt> bits =
        resolvedTokenBitPattern(token, type, scope);
    if (!bits)
      return bits.takeError();
    if (bits->getBitWidth() != expectedWidth)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "DFG observation bit width does not match the canonical lane shape");
    for (std::uint64_t lane = 0; lane < shape.lanesPerToken; ++lane)
      sequence.lanes.push_back(SemanticLane::defined(
          bits->extractBits(shape.laneBitWidth,
                            static_cast<unsigned>(lane * shape.laneBitWidth))));
  }
  return sequence;
}

const MemoryRootBindingEntry *
bindingFor(const SpatialSimulationRuntimeInput &input,
           dataflow::LogicalMemoryRootRef root) {
  for (const MemoryRootBindingEntry &entry : input.memoryRootBindings)
    if (entry.root == root)
      return &entry;
  return nullptr;
}

llvm::Expected<dataflow::LogicalMemoryRootOrViewRef>
resolveObservableRole(const SpatialMemoryObservableTarget &target,
                      const SpatialSimulationWorkload &workload,
                      const dataflow::CanonicalDataflowProgramView &program) {
  if (const auto *direct =
          std::get_if<dataflow::LogicalMemoryRootOrViewRef>(&target))
    return *direct;
  return program.resolveExposure(dataflow::MemoryExposureRef{
      workload.launchRef,
      std::get<MemoryExposureTarget>(target).memoryResultOrdinal});
}

dataflow::LogicalMemoryRootRef
rootOf(const dataflow::LogicalMemoryRootOrViewRef &role) {
  if (const auto *root = std::get_if<dataflow::LogicalMemoryRootRef>(&role))
    return *root;
  return std::get<dataflow::LogicalMemoryViewRef>(role).root;
}

llvm::Expected<MemoryView>
resolveRootMemory(dataflow::LogicalMemoryRootRef root, dataflow::GraphOp graph,
                  SimulatorState &state, const ResolvedLaunchContext &context,
                  const dataflow::CanonicalDataflowProgramView &program) {
  const std::uint64_t memoryBase =
      1 + context.numValueInputs + context.numStreamInputs;
  for (auto [ordinal, imported] : llvm::enumerate(context.memoryInputRoots)) {
    if (!imported || *imported != root)
      continue;
    mlir::BlockArgument argument =
        graph.getBody().front().getArgument(memoryBase + ordinal);
    std::optional<MemoryView> memory = resolveMemoryView(state, argument);
    if (!memory)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "retired DFG execution lost an imported memory root");
    return *memory;
  }

  llvm::Expected<dataflow::CanonicalLogicalMemoryRootView> resolved =
      program.resolve(root);
  if (!resolved)
    return resolved.takeError();
  if (resolved->formalArgIndex)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "observable imported memory root is not bound by the selected graph");
  if (!resolved->op || resolved->op->getNumResults() != 1)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "fresh logical memory root has no unique storage result");
  std::optional<MemoryView> memory =
      resolveMemoryView(state, resolved->op->getResult(0));
  if (!memory)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "retired DFG execution lost a fresh memory root");
  return *memory;
}

llvm::Expected<std::vector<SemanticMemoryByte>>
projectMemoryBytes(const MemoryView &view) {
  if (!view.memory || view.byteOffset < 0 ||
      static_cast<std::uint64_t>(view.byteOffset) > view.memory->bytes.size() ||
      view.memory->initialized.size() != view.memory->bytes.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "retired DFG memory view is outside its backing object");
  const std::size_t begin = static_cast<std::size_t>(view.byteOffset);
  std::vector<SemanticMemoryByte> bytes;
  bytes.reserve(view.memory->bytes.size() - begin);
  for (std::size_t offset = begin; offset < view.memory->bytes.size();
       ++offset) {
    if (!view.memory->initialized[offset]) {
      bytes.push_back({SemanticState::Undef, 0});
      continue;
    }
    SemanticMemoryByte byte = view.memory->bytes[offset];
    if (byte.state != SemanticState::Defined)
      byte.value = 0;
    bytes.push_back(byte);
  }
  return bytes;
}

bool semanticByteEqual(const SemanticMemoryByte &lhs,
                       const SemanticMemoryByte &rhs) {
  return lhs.state == rhs.state &&
         (lhs.state != SemanticState::Defined || lhs.value == rhs.value);
}

llvm::Expected<DiffMemoryObservation>
makeDiff(llvm::ArrayRef<SemanticMemoryByte> finalBytes,
         llvm::ArrayRef<SemanticMemoryByte> baseline) {
  if (finalBytes.size() != baseline.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "DFG memory observation and runtime baseline extents differ");
  DiffMemoryObservation diff;
  diff.byteCount = finalBytes.size();
  std::size_t offset = 0;
  while (offset < finalBytes.size()) {
    while (offset < finalBytes.size() &&
           semanticByteEqual(finalBytes[offset], baseline[offset]))
      ++offset;
    if (offset == finalBytes.size())
      break;
    MemoryDiffRun run;
    run.byteOffset = offset;
    do {
      run.changedBytes.push_back(finalBytes[offset]);
      ++offset;
    } while (offset < finalBytes.size() &&
             !semanticByteEqual(finalBytes[offset], baseline[offset]));
    diff.runs.push_back(std::move(run));
  }
  return diff;
}

llvm::Expected<MemoryObservationPayload> projectMemoryObservation(
    const SpatialMemoryObservable &observable,
    const SpatialSimulationWorkload &workload,
    const SpatialSimulationRuntimeInput &runtimeInput, dataflow::GraphOp graph,
    SimulatorState &state, const ResolvedLaunchContext &context,
    const dataflow::CanonicalDataflowProgramView &program) {
  llvm::Expected<dataflow::LogicalMemoryRootOrViewRef> role =
      resolveObservableRole(observable.target, workload, program);
  if (!role)
    return role.takeError();
  const dataflow::LogicalMemoryRootRef root = rootOf(*role);
  llvm::Expected<MemoryView> memory =
      resolveRootMemory(root, graph, state, context, program);
  if (!memory)
    return memory.takeError();
  llvm::Expected<std::vector<SemanticMemoryByte>> finalBytes =
      projectMemoryBytes(*memory);
  if (!finalBytes)
    return finalBytes.takeError();

  if (observable.form == MemoryObservationForm::FullState)
    return MemoryObservationPayload{
        FullMemoryObservation{std::move(*finalBytes)}};

  const MemoryRootBindingEntry *binding = bindingFor(runtimeInput, root);
  if (!binding ||
      binding->binding.objectOrdinal >= runtimeInput.memoryObjects.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "DFG diff observation has no admitted runtime baseline");
  const RuntimeMemoryObject &object =
      runtimeInput.memoryObjects[binding->binding.objectOrdinal];
  if (binding->binding.byteOffset > object.initialBytes.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "DFG diff observation baseline offset is out of range");
  llvm::ArrayRef<SemanticMemoryByte> baseline(object.initialBytes);
  baseline = baseline.drop_front(binding->binding.byteOffset);
  llvm::Expected<DiffMemoryObservation> diff = makeDiff(*finalBytes, baseline);
  if (!diff)
    return diff.takeError();
  return MemoryObservationPayload{std::move(*diff)};
}

} // namespace

llvm::Expected<CanonicalValueSequence>
canonicalValueSequenceFromTokens(llvm::ArrayRef<Token> tokens, mlir::Type type,
                                 mlir::Operation *scope) {
  auto shape = laneShapeOf(type, scope);
  if (!shape)
    return shape.takeError();
  return sequenceFromTokens(tokens, type, *shape, scope);
}

llvm::Expected<SpatialFunctionalObservations>
projectRetiredFunctionalObservations(
    dataflow::GraphOp graph, SimulatorState &state,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const ResolvedLaunchContext &context,
    const dataflow::CanonicalDataflowProgramView &program) {
  const SpatialSimulationWorkload &model = *workload.spatial();
  const SpatialSimulationRuntimeInput &input = *runtimeInput.spatial();
  auto graphReturn = mlir::cast<dataflow::GraphReturnOp>(
      graph.getBody().front().getTerminator());
  SpatialFunctionalObservations observations;

  observations.valueResults.reserve(
      model.observableContract.valueResults.size());
  for (std::uint64_t ordinal : model.observableContract.valueResults) {
    mlir::Value value = graphReturn.getValues()[ordinal];
    auto found = state.observedOutputs.find(value);
    if (found == state.observedOutputs.end() || found->second.size() != 1)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "retired DFG value result is not published exactly once");
    llvm::Expected<CanonicalValueSequence> sequence = sequenceFromTokens(
        found->second, value.getType(), context.valueResultShapes[ordinal],
        graph.getOperation());
    if (!sequence)
      return sequence.takeError();
    observations.valueResults.emplace_back(
        PublishedValueResult{std::move(*sequence)});
  }

  observations.streamOutputs.reserve(
      model.observableContract.streamOutputs.size());
  for (std::uint64_t ordinal : model.observableContract.streamOutputs) {
    mlir::Value value = graphReturn.getStreams()[ordinal];
    llvm::ArrayRef<Token> tokens;
    if (auto found = state.observedOutputs.find(value);
        found != state.observedOutputs.end())
      tokens = found->second;
    llvm::Expected<CanonicalValueSequence> sequence = sequenceFromTokens(
        tokens, value.getType(), context.streamOutputShapes[ordinal],
        graph.getOperation());
    if (!sequence)
      return sequence.takeError();
    observations.streamOutputs.push_back(CanonicalStreamSequence{
        std::move(*sequence), StreamTermination::ClosedAfterLast});
  }

  observations.memories.reserve(model.observableContract.memories.size());
  for (const SpatialMemoryObservable &observable :
       model.observableContract.memories) {
    llvm::Expected<MemoryObservationPayload> payload = projectMemoryObservation(
        observable, model, input, graph, state, context, program);
    if (!payload)
      return payload.takeError();
    observations.memories.push_back(std::move(*payload));
  }
  return observations;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
