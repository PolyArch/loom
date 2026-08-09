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

llvm::Expected<Token>
tokenFromLanes(llvm::ArrayRef<SemanticLane> lanes, mlir::Type type,
               const LaneShape &shape,
               llvm::ArrayRef<std::shared_ptr<MemoryValue>> objects,
               mlir::Operation *scope) {
  if (shape.lanesPerToken == 0)
    return noneToken();
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
    llvm::SmallVector<PrimitiveValue, 8> values;
    values.reserve(lanes.size());
    for (const SemanticLane &lane : lanes) {
      if (lane.state == SemanticState::Poison)
        values.push_back(PrimitiveValue::poison());
      else if (lane.state == SemanticState::Undef)
        values.push_back(PrimitiveValue::undef());
      else
        values.push_back(PrimitiveValue::integer(lane.bits));
    }
    return tokenFromVectorPrimitiveValues(values, vector, scope);
  }
  const SemanticState state = lanes.front().state;
  if (state != SemanticState::Defined) {
    const PrimitiveValueState primitive = state == SemanticState::Poison
                                              ? PrimitiveValueState::Poison
                                              : PrimitiveValueState::Undef;
    return exceptionalValueToken(primitive, type);
  }
  if (shape.lanesPerToken == 1) {
    if (mlir::isa<mlir::IndexType>(type))
      return indexToken(lanes.front().bits);
    if (shape.pointerLayout) {
      const PointerTarget &target = *lanes.front().pointerTarget;
      if (target.objectOrdinal >= objects.size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "typed pointer input names an unavailable runtime object");
      Token token;
      token.kind = TokenKind::Pointer;
      token.setPointerValue(
          PointerValue{objects[target.objectOrdinal], target.objectOrdinal,
                       shape.pointerLayout->addressSpace, target.byteOffset,
                       lanes.front().bits});
      return token;
    }
    return tokenFromBitPattern(lanes.front().bits, type);
  }

  return llvm::createStringError(std::errc::invalid_argument,
                                 "typed multi-lane input is not a vector");
}

llvm::Expected<llvm::SmallVector<Token>>
tokensFromSequence(const CanonicalValueSequence &sequence, mlir::Type type,
                   const LaneShape &shape,
                   llvm::ArrayRef<std::shared_ptr<MemoryValue>> objects,
                   mlir::Operation *scope) {
  llvm::SmallVector<Token> tokens;
  tokens.reserve(sequence.tokenCount);
  for (std::uint64_t token = 0; token < sequence.tokenCount; ++token) {
    llvm::ArrayRef<SemanticLane> lanes;
    if (shape.lanesPerToken != 0)
      lanes = llvm::ArrayRef(sequence.lanes)
                  .slice(token * shape.lanesPerToken, shape.lanesPerToken);
    auto converted = tokenFromLanes(lanes, type, shape, objects, scope);
    if (!converted)
      return converted.takeError();
    tokens.push_back(std::move(*converted));
  }
  return tokens;
}

const RuntimeValueEntry *
runtimeValueAt(const SpatialSimulationRuntimeInput &input,
               std::uint64_t ordinal) {
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

const MemoryRootBindingEntry *
bindingFor(const SpatialSimulationRuntimeInput &input,
           dataflow::LogicalMemoryRootRef root) {
  for (const MemoryRootBindingEntry &entry : input.memoryRootBindings)
    if (entry.root == root)
      return &entry;
  return nullptr;
}

} // namespace

std::optional<std::string>
unsupportedTypedDfgInput(const CanonicalSimulationWorkload &workload,
                         const CanonicalSimulationRuntimeInput &runtimeInput,
                         const ResolvedLaunchContext &context) {
  (void)workload;
  const SpatialSimulationRuntimeInput &input = *runtimeInput.spatial();
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

llvm::Error
seedTypedDfgInputs(SimulatorState &state, dataflow::GraphOp graph,
                   const CanonicalSimulationWorkload &workload,
                   const CanonicalSimulationRuntimeInput &runtimeInput,
                   const ResolvedLaunchContext &context) {
  mlir::Block &entry = graph.getBody().front();
  const SpatialSimulationWorkload &model = *workload.spatial();
  const SpatialSimulationRuntimeInput &input = *runtimeInput.spatial();

  llvm::SmallVector<std::shared_ptr<MemoryValue>> objects;
  objects.reserve(input.memoryObjects.size());
  for (auto [ordinal, object] : llvm::enumerate(input.memoryObjects)) {
    if (object.initialBytes.size() > std::numeric_limits<unsigned>::max())
      return llvm::createStringError(
          std::errc::not_supported,
          "typed runtime memory exceeds the DFG provider capacity");
    auto memory = std::make_shared<MemoryValue>();
    memory->logicalRootId = ordinal;
    memory->bytes.append(object.initialBytes.begin(),
                         object.initialBytes.end());
    memory->initialized = llvm::SmallBitVector(memory->bytes.size(), true);
    objects.push_back(std::move(memory));
  }
  for (auto [storageOrdinal, object] : llvm::enumerate(input.memoryObjects)) {
    for (const RuntimeMemoryPointer &stored : object.pointerValues) {
      if (stored.target.objectOrdinal >= objects.size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "typed stored pointer names an unavailable runtime object");
      llvm::Expected<PointerLayout> layout =
          resolvePointerLayout(context.graphOp, stored.addressSpace);
      if (!layout)
        return layout.takeError();
      llvm::Expected<llvm::APInt> representation =
          decodeRuntimePointerRepresentation(object, stored, context.graphOp);
      if (!representation)
        return representation.takeError();
      objects[storageOrdinal]->pointerValues.emplace(
          stored.storageByteOffset,
          PointerValue{objects[stored.target.objectOrdinal],
                       stored.target.objectOrdinal, stored.addressSpace,
                       stored.target.byteOffset, std::move(*representation)});
    }
  }
  state.nextMemoryRootId = objects.size();

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
                                     context.valueInputShapes[ordinal], objects,
                                     graph.getOperation());
    if (!tokens)
      return tokens.takeError();
    for (const Token &token : *tokens)
      seedBlockArgument(state, argument, token);
  }

  const std::uint64_t streamBase = 1 + context.numValueInputs;
  for (std::uint64_t ordinal = 0; ordinal < input.runtimeStreams.size();
       ++ordinal) {
    mlir::BlockArgument argument = entry.getArgument(streamBase + ordinal);
    auto tokens = tokensFromSequence(
        input.runtimeStreams[ordinal].values, argument.getType(),
        context.streamInputShapes[ordinal], objects, graph.getOperation());
    if (!tokens)
      return tokens.takeError();
    for (const Token &token : *tokens)
      seedBlockArgument(state, argument, token);
  }

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
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "admitted typed memory binding is not executable");
    mlir::BlockArgument argument = entry.getArgument(memoryBase + ordinal);
    mlir::Type elementType;
    if (auto memoryType = mlir::dyn_cast<mlir::MemRefType>(argument.getType()))
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
