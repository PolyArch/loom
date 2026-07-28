//===- DFGSimulatorTypedInput.cpp - typed spatial input seeding ----------===//

#include "DFGSimulatorInternal.h"
#include "SimulationWireInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>
#include <string>
#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {
namespace {

bool sequenceNeedsLaneStateSupport(const CanonicalValueSequence &sequence,
                                   const LaneShape &shape) {
  if (shape.lanesPerToken <= 1)
    return false;
  for (std::uint64_t token = 0; token < sequence.tokenCount; ++token) {
    const std::size_t begin = token * shape.lanesPerToken;
    const SemanticState state = sequence.lanes[begin].state;
    for (std::size_t lane = 1; lane < shape.lanesPerToken; ++lane)
      if (sequence.lanes[begin + lane].state != state)
        return true;
  }
  return false;
}

llvm::Expected<Token> tokenFromLanes(llvm::ArrayRef<SemanticLane> lanes,
                                     mlir::Type type,
                                     const LaneShape &shape) {
  if (shape.lanesPerToken == 0)
    return noneToken();
  const SemanticState state = lanes.front().state;
  if (state != SemanticState::Defined) {
    const PrimitiveValueState primitive =
        state == SemanticState::Poison ? PrimitiveValueState::Poison
                                       : PrimitiveValueState::Undef;
    return exceptionalValueToken(primitive, type);
  }
  if (shape.lanesPerToken == 1) {
    if (mlir::isa<mlir::IndexType>(type))
      return indexToken(lanes.front().bits);
    return tokenFromBitPattern(lanes.front().bits, type);
  }

  if (shape.lanesPerToken >
      std::numeric_limits<unsigned>::max() / shape.laneBitWidth)
    return llvm::createStringError(std::errc::value_too_large,
                                   "typed vector input is too wide");
  llvm::APInt bits(static_cast<unsigned>(shape.lanesPerToken) *
                       shape.laneBitWidth,
                   0);
  for (auto [ordinal, lane] : llvm::enumerate(lanes))
    bits.insertBits(lane.bits,
                    static_cast<unsigned>(ordinal) * shape.laneBitWidth);
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
    if (mlir::isa<mlir::IndexType>(vector.getElementType())) {
      Token token;
      token.kind = TokenKind::Vector;
      token.setExactBitPattern(std::move(bits));
      return token;
    }
  }
  return tokenFromBitPattern(bits, type);
}

llvm::Expected<llvm::SmallVector<Token>>
tokensFromSequence(const CanonicalValueSequence &sequence, mlir::Type type,
                   const LaneShape &shape) {
  llvm::SmallVector<Token> tokens;
  tokens.reserve(sequence.tokenCount);
  for (std::uint64_t token = 0; token < sequence.tokenCount; ++token) {
    llvm::ArrayRef<SemanticLane> lanes;
    if (shape.lanesPerToken != 0)
      lanes = llvm::ArrayRef(sequence.lanes)
                  .slice(token * shape.lanesPerToken, shape.lanesPerToken);
    auto converted = tokenFromLanes(lanes, type, shape);
    if (!converted)
      return converted.takeError();
    tokens.push_back(std::move(*converted));
  }
  return tokens;
}

const RuntimeValueEntry *runtimeValueAt(
    const SpatialSimulationRuntimeInput &input, std::uint64_t ordinal) {
  auto found = llvm::lower_bound(
      input.runtimeValues, ordinal,
      [](const RuntimeValueEntry &entry, std::uint64_t value) {
        return entry.valueInputOrdinal < value;
      });
  return found != input.runtimeValues.end() &&
                 found->valueInputOrdinal == ordinal
             ? &*found
             : nullptr;
}

const MemoryRootBindingEntry *bindingFor(
    const SpatialSimulationRuntimeInput &input,
    dataflow::LogicalMemoryRootRef root) {
  for (const MemoryRootBindingEntry &entry : input.memoryRootBindings)
    if (entry.root == root)
      return &entry;
  return nullptr;
}

} // namespace

std::optional<std::string> unsupportedTypedDfgInput(
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const ResolvedLaunchContext &context) {
  const SpatialSimulationWorkload &model = workload.model();
  const SpatialSimulationRuntimeInput &input = runtimeInput.model();
  for (std::uint64_t ordinal = 0; ordinal < model.valueInputPlan.size();
       ++ordinal) {
    const CanonicalValueSequence *sequence =
        std::get_if<CanonicalValueSequence>(&model.valueInputPlan[ordinal]);
    if (!sequence) {
      const RuntimeValueEntry *entry = runtimeValueAt(input, ordinal);
      if (entry)
        sequence = &entry->value;
    }
    if (sequence && sequenceNeedsLaneStateSupport(
                        *sequence, context.valueInputShapes[ordinal]))
      return "mixed per-lane exceptional value input is unsupported";
  }
  if (!input.runtimeStreams.empty())
    return "typed runtime stream termination is unsupported";
  for (const auto &root : context.memoryInputRoots)
    if (!root)
      return "a graph memory input without a runtime root is unsupported";
  if (!input.memoryObjects.empty()) {
    mlir::Attribute endianness =
        mlir::DataLayout::closest(context.graphOp).getEndianness();
    auto spelling = mlir::dyn_cast_or_null<mlir::StringAttr>(endianness);
    if (!spelling ||
        (spelling.getValue() != "little" && spelling.getValue() != "big"))
      return "typed runtime memory requires an explicit supported DataLayout "
             "endianness";
  }
  return std::nullopt;
}

llvm::Error seedTypedDfgInputs(
    SimulatorState &state, dataflow::GraphOp graph,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const ResolvedLaunchContext &context) {
  mlir::Block &entry = graph.getBody().front();
  const SpatialSimulationWorkload &model = workload.model();
  const SpatialSimulationRuntimeInput &input = runtimeInput.model();

  for (std::uint64_t ordinal = 0; ordinal < model.valueInputPlan.size();
       ++ordinal) {
    const CanonicalValueSequence *sequence =
        std::get_if<CanonicalValueSequence>(&model.valueInputPlan[ordinal]);
    if (!sequence) {
      const RuntimeValueEntry *runtime = runtimeValueAt(input, ordinal);
      if (!runtime)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "admitted typed value input has no runtime value");
      sequence = &runtime->value;
    }
    mlir::BlockArgument argument = entry.getArgument(ordinal + 1);
    auto tokens = tokensFromSequence(*sequence, argument.getType(),
                                     context.valueInputShapes[ordinal]);
    if (!tokens)
      return tokens.takeError();
    for (const Token &token : *tokens)
      seedBlockArgument(state, argument, token);
  }

  const std::uint64_t streamBase = 1 + context.numValueInputs;
  for (std::uint64_t ordinal = 0; ordinal < input.runtimeStreams.size();
       ++ordinal) {
    mlir::BlockArgument argument = entry.getArgument(streamBase + ordinal);
    auto tokens = tokensFromSequence(input.runtimeStreams[ordinal].values,
                                     argument.getType(),
                                     context.streamInputShapes[ordinal]);
    if (!tokens)
      return tokens.takeError();
    for (const Token &token : *tokens)
      seedBlockArgument(state, argument, token);
  }

  llvm::SmallVector<std::shared_ptr<MemoryValue>> objects;
  objects.reserve(input.memoryObjects.size());
  for (auto [ordinal, object] : llvm::enumerate(input.memoryObjects)) {
    if (object.initialBytes.size() > std::numeric_limits<unsigned>::max())
      return llvm::createStringError(
          std::errc::not_supported,
          "typed runtime memory exceeds the DFG provider capacity");
    auto memory = std::make_shared<MemoryValue>();
    memory->logicalRootId = ordinal;
    memory->bytes.append(object.initialBytes.begin(), object.initialBytes.end());
    memory->initialized = llvm::SmallBitVector(memory->bytes.size(), true);
    objects.push_back(std::move(memory));
  }
  state.nextMemoryRootId = objects.size();

  const std::uint64_t memoryBase = streamBase + context.numStreamInputs;
  for (std::uint64_t ordinal = 0; ordinal < context.memoryInputRoots.size();
       ++ordinal) {
    const auto &root = context.memoryInputRoots[ordinal];
    if (!root)
      return llvm::createStringError(
          std::errc::not_supported,
          "typed graph memory input has no runtime root");
    const MemoryRootBindingEntry *binding = bindingFor(input, *root);
    if (!binding || binding->binding.objectOrdinal >= objects.size() ||
        binding->binding.byteOffset >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "admitted typed memory binding is not executable");
    mlir::BlockArgument argument = entry.getArgument(memoryBase + ordinal);
    mlir::Type elementType;
    if (auto memoryType =
            mlir::dyn_cast<mlir::MemRefType>(argument.getType()))
      elementType = memoryType.getElementType();
    std::shared_ptr<MemoryValue> memory =
        objects[binding->binding.objectOrdinal];
    state.memoryRootIds[argument] = memory->logicalRootId;
    state.memories[argument] = memory;
    state.memoryViews[argument] = MemoryView{
        memory, argument,
        static_cast<std::int64_t>(binding->binding.byteOffset), elementType};
  }
  return llvm::Error::success();
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
