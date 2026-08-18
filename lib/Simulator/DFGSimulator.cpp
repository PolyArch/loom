#include "Simulator/DFGSimulator.h"
#include "DFGSimulatorInternal.h"
#include "SimulationWireInternal.h"

#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Simulator/SimulationAdmission.h"

#include "Common/ArtifactText.h"
#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>

using namespace loom::sim;
using namespace loom::sim::detail;

namespace loom::sim {

char NonRetiredDFGExecutionError::ID;

NonRetiredDFGExecutionError::NonRetiredDFGExecutionError(
    DFGSimulationReport report)
    : report_(std::move(report)) {}

void NonRetiredDFGExecutionError::log(llvm::raw_ostream &stream) const {
  stream << "DFG execution did not retire: " << report_.status;
  if (!report_.diagnostics.empty())
    stream << ": " << report_.diagnostics.front();
}

std::error_code NonRetiredDFGExecutionError::convertToErrorCode() const {
  return std::make_error_code(std::errc::state_not_recoverable);
}

struct PreparedDfgExecution::Impl {
  std::unique_ptr<dataflow::CanonicalDataflowArtifact> program;
  dataflow::RootedGraphLaunchRef launch;
  dataflow::CanonicalDataflowProgramView view;
  detail::ResolvedLaunchContext context;
  detail::PreparedGraphExecution execution;
};

PreparedDfgExecution::PreparedDfgExecution(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

PreparedDfgExecution::PreparedDfgExecution(PreparedDfgExecution &&) noexcept =
    default;

PreparedDfgExecution &
PreparedDfgExecution::operator=(PreparedDfgExecution &&) noexcept = default;

PreparedDfgExecution::~PreparedDfgExecution() = default;

namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

// A memory fixture is an operand of the graph that owns it, so its element
// tokens are encoded against that same scope as every other runtime token.
static llvm::Expected<llvm::SmallVector<Token>>
parseMemoryTokens(llvm::StringRef raw, mlir::Type type,
                  mlir::Operation *scope) {
  llvm::SmallVector<Token> tokens;
  llvm::SmallVector<llvm::StringRef> parts;
  raw.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (parts.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "memref fixture must contain values");
  for (llvm::StringRef part : parts) {
    auto tokenOrErr = parseRuntimeToken(part, type, scope);
    if (!tokenOrErr)
      return tokenOrErr.takeError();
    tokens.push_back(*tokenOrErr);
  }
  return tokens;
}

std::int64_t integerToken(const Token &token);

static std::string typeToString(mlir::Type type) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  type.print(os);
  return os.str();
}

static llvm::Expected<std::int64_t> byteSizeForBitWidth(std::uint64_t width) {
  if (width == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "token bit width must be nonzero");
  const std::uint64_t bytes = llvm::divideCeil(width, std::uint64_t{8});
  if (bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return llvm::createStringError(std::errc::value_too_large,
                                   "token byte size is unsupported");
  return static_cast<std::int64_t>(bytes);
}

llvm::Expected<std::int64_t> byteSizeOfType(mlir::Type type,
                                            mlir::Operation *scope) {
  auto width = resolvedTokenTypeBitWidth(type, scope);
  if (!width)
    return llvm::createStringError(
        std::errc::invalid_argument, "unsupported memory element type %s: %s",
        typeToString(type).c_str(), llvm::toString(width.takeError()).c_str());
  return byteSizeForBitWidth(*width);
}

llvm::Expected<std::shared_ptr<MemoryValue>>
materializeMemory(SimulatorState &state, mlir::Value root, llvm::StringRef raw,
                  mlir::Type elementType) {
  auto [rootIt, inserted] =
      state.memoryRootIds.try_emplace(root, state.nextMemoryRootId);
  if (inserted)
    ++state.nextMemoryRootId;
  auto existing = state.memories.find(root);
  if (existing != state.memories.end()) {
    if (existing->second->logicalRootId != rootIt->second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memory root identity mismatch");
    return existing->second;
  }
  auto tokensOrErr = parseMemoryTokens(raw, elementType, state.graphScope);
  if (!tokensOrErr)
    return tokensOrErr.takeError();
  llvm::SmallVector<Token> tokens = std::move(*tokensOrErr);
  llvm::SmallVector<SemanticMemoryByte> bytes;
  for (const Token &token : tokens) {
    auto encoded = encodeMemoryElement(token, elementType, state.graphScope);
    if (!encoded)
      return encoded.takeError();
    bytes.append(encoded->begin(), encoded->end());
  }
  llvm::SmallBitVector initialized(bytes.size(), /*t=*/true);
  auto memory = std::make_shared<MemoryValue>(MemoryValue{
      rootIt->second, std::move(bytes), std::move(initialized), {}});
  state.memories[root] = memory;
  return memory;
}

Token memoryCapabilityToken(mlir::Value root,
                            std::shared_ptr<MemoryValue> memory,
                            std::int64_t byteOffset, mlir::Type elementType) {
  Token token;
  token.kind = TokenKind::MemoryCapability;
  token.setMemoryView(
      MemoryView{std::move(memory), root, byteOffset, elementType});
  return token;
}

llvm::Expected<Token> zeroToken(mlir::Type type) {
  if (mlir::isa<mlir::IndexType>(type))
    return integerValueToken(0);
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return tokenFromBitPattern(llvm::APInt(intType.getWidth(), 0), intType);
  if (mlir::isa<mlir::FloatType>(type))
    return floatValueToken(0.0);
  if (mlir::isa<mlir::VectorType>(type)) {
    auto width = tokenTypeBitWidth(type);
    if (!width)
      return width.takeError();
    return tokenFromBitPattern(llvm::APInt(*width, 0), type);
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported zero-initialized memory type: %s",
                                 typeToString(type).c_str());
}

static ChannelSlot *findChannelSlot(SimulatorState &state,
                                    mlir::OpOperand &operand) {
  assert(state.execution && "channel lookup requires a prepared graph");
  auto found = state.execution->channelOrdinals.find(&operand);
  if (found != state.execution->channelOrdinals.end())
    return &state.channelSlots[found->second];
  return nullptr;
}

static const ChannelSlot *findChannelSlot(const SimulatorState &state,
                                          mlir::OpOperand &operand) {
  assert(state.execution && "channel lookup requires a prepared graph");
  auto found = state.execution->channelOrdinals.find(&operand);
  if (found == state.execution->channelOrdinals.end())
    return nullptr;
  return &state.channelSlots[found->second];
}

TokenQueue &channelQueue(SimulatorState &state, mlir::OpOperand &operand) {
  ChannelSlot *slot = findChannelSlot(state, operand);
  assert(slot && "finalized graph operand has no prepared channel");
  return slot->ready;
}

bool hasToken(const SimulatorState &state, mlir::OpOperand &operand) {
  const ChannelSlot *slot = findChannelSlot(state, operand);
  return slot && !slot->ready.empty();
}

static const ChannelSlot &inputChannelSlot(const SimulatorState &state,
                                           unsigned operandOrdinal) {
  assert(state.currentActorPlan &&
         operandOrdinal < state.currentActorPlan->inputChannelCount &&
         "input operand is outside the active actor plan");
  return state
      .channelSlots[state.currentActorPlan->firstInputChannel + operandOrdinal];
}

static ChannelSlot &inputChannelSlot(SimulatorState &state,
                                     unsigned operandOrdinal) {
  return const_cast<ChannelSlot &>(inputChannelSlot(
      static_cast<const SimulatorState &>(state), operandOrdinal));
}

bool hasInputToken(const SimulatorState &state, unsigned operandOrdinal) {
  return !inputChannelSlot(state, operandOrdinal).ready.empty();
}

static void scheduleActor(SimulatorState &state, unsigned ordinal) {
  if (ordinal == InvalidActorOrdinal)
    return;
  state.nextActorCandidates.set(ordinal);
  if (state.execution->actorPlans[ordinal].isPlainMemory())
    state.plainMemoryCandidates.set(ordinal);
}

Token popInputToken(SimulatorState &state, unsigned operandOrdinal) {
  TokenQueue &queue = inputChannelSlot(state, operandOrdinal).ready;
  assert(!queue.empty() && "pop requires a nonempty actor input channel");
  Token token = std::move(queue.front());
  queue.pop_front();
  if (!token.memoryOrder.empty())
    state.firingMemoryOrderFrontier.absorb(token.memoryOrder);
  ++state.actorMutationEpoch;
  return token;
}

const Token &peekInputToken(const SimulatorState &state,
                            unsigned operandOrdinal) {
  const TokenQueue &queue = inputChannelSlot(state, operandOrdinal).ready;
  assert(!queue.empty() && "peek requires a nonempty actor input channel");
  return queue.front();
}

static void publishToken(SimulatorState &state, unsigned resultOrdinal,
                         Token token) {
  assert(state.currentActorPlan &&
         "token publication requires an active execution plan");
  assert(resultOrdinal < state.currentActorPlan->outputs.size() &&
         "token publication does not match the active actor result");
  if (state.actorEmissionCapture) {
    state.actorEmissionCapture->push_back(
        ActorResultEmission{resultOrdinal, std::move(token)});
    ++state.actorMutationEpoch;
    return;
  }
  const ActorExecutionPlan::Output &output =
      state.currentActorPlan->outputs[resultOrdinal];
  if (output.observed) {
    auto &pending = state.pendingObservedOutputs[output.value];
    if (pending.empty())
      state.pendingObservedValues.push_back(output.value);
    if (output.channels.empty())
      pending.push_back(std::move(token));
    else
      pending.push_back(token);
  }
  for (auto [index, ordinal] : llvm::enumerate(output.channels)) {
    ChannelSlot &slot = state.channelSlots[ordinal];
    TokenQueue &pending = slot.pending;
    if (pending.empty())
      state.pendingChannelOrdinals.push_back(ordinal);
    if (index + 1 == output.channels.size())
      pending.push_back(std::move(token));
    else
      pending.push_back(token);
  }
  ++state.actorMutationEpoch;
}

void emitResultToken(SimulatorState &state, unsigned resultOrdinal,
                     Token token) {
  if (!state.firingMemoryOrderFrontier.empty() || !token.memoryOrder.empty())
    token.memoryOrder = publishFiredMemoryOrder(state, token.memoryOrder);
  publishToken(state, resultOrdinal, std::move(token));
}

void emitResultTokenWithMemoryOrder(SimulatorState &state,
                                    unsigned resultOrdinal, Token token,
                                    MemoryOrderFrontierId memoryOrder) {
  token.memoryOrder = memoryOrder;
  publishToken(state, resultOrdinal, std::move(token));
}

bool recordEvent(SimulatorState &state, dataflow::OperationSchemaId schema) {
  ++state.eventCount;
  const auto ordinal = static_cast<std::size_t>(schema);
  assert(ordinal < state.operationFireCounts.size() &&
         "registered operation schema is outside its dense domain");
  ++state.operationFireCounts[ordinal];
  return true;
}

void flushPendingTokens(SimulatorState &state) {
  for (ChannelOrdinal ordinal : state.pendingChannelOrdinals) {
    ChannelSlot &slot = state.channelSlots[ordinal];
    assert(slot.operand && !slot.pending.empty() &&
           "a scheduled pending channel must exist and contain a token");
    TokenQueue &pending = slot.pending;
    scheduleActor(state, slot.ownerActorOrdinal);
    TokenQueue &target = slot.ready;
    target.appendFrom(pending);
  }
  state.pendingChannelOrdinals.clear();
  for (mlir::Value value : state.pendingObservedValues) {
    auto &pending = state.pendingObservedOutputs[value];
    auto &target = state.observedOutputs[value];
    target.append(std::make_move_iterator(pending.begin()),
                  std::make_move_iterator(pending.end()));
    pending.clear();
  }
  state.pendingObservedValues.clear();
}

std::int64_t integerToken(const Token &token) {
  if (token.hasExactBitPattern())
    return token.exactBitPattern().sextOrTrunc(64).getSExtValue();
  if (token.kind == TokenKind::Bool)
    return token.scalarValue != 0 ? 1 : 0;
  return static_cast<std::int64_t>(token.scalarValue);
}

bool boolToken(const Token &token) {
  if (token.hasExactBitPattern())
    return !token.exactBitPattern().isZero();
  if (token.kind == TokenKind::Bool)
    return token.scalarValue != 0;
  return static_cast<std::int64_t>(token.scalarValue) != 0;
}

llvm::Expected<PrimitiveValue> primitiveValueFromToken(const Token &token,
                                                       mlir::Type type,
                                                       unsigned indexBitWidth) {
  if (token.valueState == PrimitiveValueState::Poison)
    return PrimitiveValue::poison();
  if (token.valueState == PrimitiveValueState::Undef)
    return PrimitiveValue::undef();
  if (mlir::isa<mlir::IndexType>(type)) {
    if (indexBitWidth == 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "index primitive operand has no resolved bit width");
    auto bits = indexTokenBitPattern(token, indexBitWidth);
    if (!bits)
      return bits.takeError();
    return PrimitiveValue::integer(*bits);
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    auto bits = tokenBitPattern(token, intType);
    if (!bits)
      return bits.takeError();
    return PrimitiveValue::integer(*bits);
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    auto bits = tokenBitPattern(token, floatType);
    if (!bits)
      return bits.takeError();
    return PrimitiveValue::floating(
        llvm::APFloat(floatType.getFloatSemantics(), *bits));
  }
  return llvm::createStringError(
      std::errc::invalid_argument,
      "primitive operand type has no scalar simulator representation");
}

llvm::Expected<Token> tokenFromPrimitiveValue(const PrimitiveValue &value,
                                              mlir::Type type) {
  if (value.state != PrimitiveValueState::Defined)
    return exceptionalValueToken(value.state, type);
  if (!value.bits)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "defined primitive result has no bits");
  if (mlir::isa<mlir::IndexType>(type)) {
    return indexToken(*value.bits);
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    auto token = tokenFromBitPattern(*value.bits, intType);
    if (!token)
      return token.takeError();
    if (intType.getWidth() >= 2 && intType.getWidth() <= 64)
      token->scalarValue =
          static_cast<std::uint64_t>(value.bits->getSExtValue());
    return *token;
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    return tokenFromBitPattern(*value.bits, floatType);
  }
  return llvm::createStringError(
      std::errc::invalid_argument,
      "primitive result type has no scalar simulator representation");
}

static llvm::Expected<unsigned> primitiveBitWidth(mlir::Type type,
                                                  mlir::Operation *scope) {
  if (!type)
    return 0u;
  return resolvedTokenTypeBitWidth(type, scope);
}

llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(const dataflow::CanonicalActorSchemaProjection &projection,
                    mlir::Operation *op, mlir::Value result) {
  mlir::Type operandType =
      op->getNumOperands() == 0 ? mlir::Type{} : op->getOperand(0).getType();
  return primitiveDescriptor(projection, op, result.getType(), operandType);
}

llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(const dataflow::CanonicalActorSchemaProjection &projection,
                    mlir::Operation *op, mlir::Type resultType,
                    mlir::Type operandType) {
  auto resultBitWidth = primitiveBitWidth(resultType, op);
  if (!resultBitWidth)
    return resultBitWidth.takeError();
  auto operandBitWidth = primitiveBitWidth(operandType, op);
  if (!operandBitWidth)
    return operandBitWidth.takeError();
  return PrimitiveOperationDescriptor{projection, *resultBitWidth,
                                      *operandBitWidth};
}

static bool isSupportedNonEvent(mlir::Operation *op) {
  return mlir::isa<dataflow::GraphReturnOp, mlir::memref::AllocOp,
                   mlir::memref::CastOp>(op);
}

std::string unsupportedOperationLabel(mlir::Operation *op) {
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto callee = call.getCallee();
    if (callee.has_value() && !callee->empty())
      return llvm::formatv("{0} @{1}", op->getName().getStringRef(), *callee)
          .str();
  }
  return op->getName().getStringRef().str();
}

static dataflow::GraphOp findGraph(mlir::ModuleOp module,
                                   llvm::StringRef name) {
  if (name.starts_with("@"))
    name = name.drop_front();
  dataflow::GraphOp match;
  module.walk([&](dataflow::GraphOp graph) {
    if (!match && graph.getSymName() == name)
      match = graph;
  });
  return match;
}

static llvm::Expected<llvm::StringMap<llvm::SmallVector<std::string>>>
indexRuntimeArgs(llvm::ArrayRef<DFGRuntimeArg> args, unsigned argCount) {
  llvm::StringMap<llvm::SmallVector<std::string>> byIndex;
  for (const DFGRuntimeArg &arg : args) {
    if (arg.index >= argCount)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "argument index %u is out of range",
                                     arg.index);
    std::string key = std::to_string(arg.index);
    byIndex[key].push_back(arg.value);
  }
  return byIndex;
}

static llvm::Expected<llvm::StringMap<MemoryFixture>>
indexMemoryArgs(llvm::ArrayRef<DFGMemoryArg> args, unsigned argCount) {
  llvm::StringMap<MemoryFixture> byIndex;
  for (const DFGMemoryArg &arg : args) {
    if (arg.index >= argCount)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memref index %u is out of range",
                                     arg.index);
    std::string key = std::to_string(arg.index);
    if (byIndex.contains(key))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memref index %u is repeated", arg.index);
    byIndex.try_emplace(key, MemoryFixture{arg.values, arg.byteOffset});
  }
  return byIndex;
}

static GraphReturnObservation observeReturnOperands(dataflow::GraphOp graph) {
  GraphReturnObservation observation;
  auto ret = mlir::dyn_cast_or_null<dataflow::GraphReturnOp>(
      graph.getBody().front().getTerminator());
  if (!ret)
    return observation;
  observation.complete.append(ret.getComplete().begin(),
                              ret.getComplete().end());
  observation.values.append(ret.getValues().begin(), ret.getValues().end());
  observation.streams.append(ret.getStreams().begin(), ret.getStreams().end());
  observation.memories.append(ret.getMemories().begin(),
                              ret.getMemories().end());
  return observation;
}

llvm::Expected<GraphPreparationResult>
prepareGraphExecution(mlir::ModuleOp module, dataflow::GraphOp graph) {
  if (llvm::Error error = dataflow::validateFinalizedProgram(module))
    return std::move(error);

  PreparedGraphExecution execution;
  execution.graph = graph;
  execution.applicationInputCount = graph.getFunctionType().getNumInputs();
  execution.returnObservation = observeReturnOperands(graph);
  execution.observedValues.insert(execution.returnObservation.complete.begin(),
                                  execution.returnObservation.complete.end());
  execution.observedValues.insert(execution.returnObservation.values.begin(),
                                  execution.returnObservation.values.end());
  execution.observedValues.insert(execution.returnObservation.streams.begin(),
                                  execution.returnObservation.streams.end());

  mlir::Block &entry = graph.getBody().front();
  std::size_t channelCount = 0;
  for (mlir::Operation &op : entry.getOperations()) {
    if (op.getNumOperands() >
        std::numeric_limits<std::size_t>::max() - channelCount)
      return llvm::createStringError(std::errc::value_too_large,
                                     "DFG channel count overflow");
    channelCount += op.getNumOperands();
  }
  if (channelCount >= std::numeric_limits<ChannelOrdinal>::max())
    return llvm::createStringError(std::errc::value_too_large,
                                   "DFG channel ordinal overflow");
  execution.channelOrdinals.reserve(channelCount);
  execution.channels.reserve(channelCount);
  for (mlir::Operation &op : entry.getOperations()) {
    for (mlir::OpOperand &operand : op.getOpOperands()) {
      const ChannelOrdinal ordinal =
          static_cast<ChannelOrdinal>(execution.channels.size());
      execution.channelOrdinals.try_emplace(&operand, ordinal);
      execution.channels.push_back({&operand, InvalidActorOrdinal});
    }
  }

  llvm::DenseMap<mlir::Operation *, unsigned> actorOrdinals;
  std::set<std::pair<std::string, std::string>> unsupported;
  for (mlir::Operation &op : entry.getOperations()) {
    if (isSupportedNonEvent(&op))
      continue;
    if (!dataflow::operationSchemaOf(&op)) {
      unsupported.emplace(unsupportedOperationLabel(&op), "");
      continue;
    }
    auto projection = dataflow::projectRegisteredActorSchemaProjection(&op);
    if (!projection) {
      unsupported.emplace(unsupportedOperationLabel(&op),
                          llvm::toString(projection.takeError()));
      continue;
    }
    if (auto diagnostic = unsupportedActorProvider(&op, *projection)) {
      unsupported.emplace(diagnostic->label, diagnostic->reason);
      continue;
    }

    std::optional<MemoryActorExecutionPlan> memoryActor;
    if (mlir::isa<dataflow::LoadOp, dataflow::StoreOp>(op)) {
      auto plan = memoryActorExecutionPlan(&op, graph);
      if (!plan)
        return GraphPreparationResult{
            GraphPreparationFailure{"invalid",
                                    {"invalid memory actor execution plan: " +
                                     llvm::toString(plan.takeError())}}};
      memoryActor = std::move(*plan);
    }
    std::optional<GepExecutionPlan> gepActor;
    if (auto gep = mlir::dyn_cast<mlir::LLVM::GEPOp>(op)) {
      auto plan = gepExecutionPlan(gep, graph);
      if (!plan) {
        unsupported.emplace(unsupportedOperationLabel(&op),
                            llvm::toString(plan.takeError()));
        continue;
      }
      gepActor = std::move(*plan);
    }
    std::optional<PrimitiveOperationDescriptor> primitive;
    if (isSupportedPrimitiveOperation(projection->schema)) {
      auto descriptor = primitiveDescriptorForActor(*projection, &op);
      if (!descriptor) {
        unsupported.emplace(unsupportedOperationLabel(&op),
                            llvm::toString(descriptor.takeError()));
        continue;
      }
      primitive = std::move(*descriptor);
    }

    ChannelOrdinal firstInput = 0;
    if (op.getNumOperands() != 0) {
      auto channel = execution.channelOrdinals.find(&op.getOpOperand(0));
      assert(channel != execution.channelOrdinals.end() &&
             "admitted actor input channel was not initialized");
      firstInput = channel->second;
      for (auto [inputOrdinal, operand] : llvm::enumerate(op.getOpOperands()))
        assert(execution.channelOrdinals.find(&operand)->second ==
                   firstInput + inputOrdinal &&
               "actor input channels are not contiguous");
    }
    const auto runtimeProvider = actorRuntimeProvider(projection->schema);
    assert(runtimeProvider && runtimeProvider->commit &&
           "admitted actor has no simulator provider");
    auto handshakeCases = dataflow::semantics::projectActorHandshakeCases(
        projection->schema, op.getNumOperands(), op.getNumResults());
    if (!handshakeCases)
      return GraphPreparationResult{GraphPreparationFailure{
          "invalid",
          {"invalid actor handshake projection: " +
           llvm::toString(handshakeCases.takeError())}}};
    llvm::SmallVector<ActorExecutionPlan::Output, 2> outputs;
    outputs.reserve(op.getNumResults());
    for (mlir::Value result : op.getResults()) {
      ActorExecutionPlan::Output output;
      output.value = result;
      output.observed = execution.observedValues.contains(result);
      for (mlir::OpOperand &use : result.getUses()) {
        auto channel = execution.channelOrdinals.find(&use);
        assert(channel != execution.channelOrdinals.end() &&
               "admitted actor output channel was not initialized");
        output.channels.push_back(channel->second);
      }
      outputs.push_back(std::move(output));
    }
    const unsigned actorOrdinal = execution.actorPlans.size();
    actorOrdinals.try_emplace(&op, actorOrdinal);
    execution.actorPlans.push_back(ActorExecutionPlan{
        &op, std::move(*projection), runtimeProvider->commit, firstInput,
        static_cast<std::uint32_t>(op.getNumOperands()), std::move(outputs),
        std::move(primitive), std::move(memoryActor), std::move(gepActor),
        std::move(*handshakeCases), runtimeProvider->probe});
  }
  if (!unsupported.empty()) {
    GraphPreparationFailure failure{"unsupported", {}};
    for (const auto &[label, reason] : unsupported) {
      std::string diagnostic = "unsupported op: " + label;
      if (!reason.empty())
        diagnostic += ": " + reason;
      failure.diagnostics.push_back(std::move(diagnostic));
    }
    return GraphPreparationResult{std::move(failure)};
  }

  execution.initialPlainMemoryCandidates.resize(execution.actorPlans.size(),
                                                false);
  for (auto [ordinal, plan] : llvm::enumerate(execution.actorPlans))
    if (plan.isPlainMemory())
      execution.initialPlainMemoryCandidates.set(ordinal);
  for (PreparedGraphExecution::Channel &channel : execution.channels) {
    auto owner = actorOrdinals.find(channel.operand->getOwner());
    if (owner != actorOrdinals.end())
      channel.ownerActorOrdinal = owner->second;
  }
  return GraphPreparationResult{std::move(execution)};
}

void seedBlockArgument(SimulatorState &state, mlir::BlockArgument arg,
                       const Token &token) {
  const std::uint64_t occurrence = state.seededTokenCounts[arg]++;
  state.observedOutputs[arg].push_back(token);
  if (state.graphIngressCapture && !arg.use_empty()) {
    state.graphIngressCapture->push_back(
        GraphIngressEmission{arg.getArgNumber(), occurrence, token});
    return;
  }
  for (mlir::OpOperand &use : arg.getUses())
    channelQueue(state, use).push_back(token);
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

namespace {

struct DfgRun {
  dataflow::GraphOp graph;
  const PreparedGraphExecution &execution;
  SimulatorState &state;
  DFGSimulationReport &report;
  bool retirementObserved = false;
  bool finalized = false;
};

void observeRetirement(DfgRun &run) {
  if (run.retirementObserved || !graphCompletionReady(run.execution, run.state))
    return;
  run.retirementObserved = true;
  if (llvm::Error error = validateGraphRetirementBoundary(
          run.graph, run.execution, run.state)) {
    run.report.status = "invalid";
    run.report.diagnostics.push_back(llvm::toString(std::move(error)));
  }
}

enum class DfgAdvanceStop { Yielded, Retired, Stopped, ExecutionLimit };

DfgAdvanceStop advanceDfgRun(
    DfgRun &run, std::uint64_t maxWavefrontSteps,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline) {
  if (run.report.status != "pass" || run.state.failure != RunFailure::None)
    return DfgAdvanceStop::Stopped;

  const std::uint64_t initialWavefront = run.report.wavefrontSteps;
  auto reachedExecutionDeadline = [&]() {
    return executionDeadline &&
           std::chrono::steady_clock::now() >= *executionDeadline;
  };
  auto stopAtExecutionDeadline = [&]() {
    run.report.status = "execution_limit";
    run.report.diagnostics.push_back("DFG execution wall-time limit reached");
  };
  llvm::SmallBitVector candidates(run.execution.actorPlans.size(), false);

  while ((run.report.wavefrontSteps - initialWavefront < maxWavefrontSteps ||
          run.retirementObserved) &&
         run.report.status != "invalid") {
    if (reachedExecutionDeadline()) {
      stopAtExecutionDeadline();
      return DfgAdvanceStop::ExecutionLimit;
    }
    if (!admitReadyPlainMemoryActions(run.state))
      return DfgAdvanceStop::Stopped;

    candidates.swap(run.state.nextActorCandidates);
    run.state.nextActorCandidates.reset();
    bool fired = false;
    unsigned actorsVisited = 0;
    for (int ordinal = candidates.find_first(); ordinal >= 0;
         ordinal = candidates.find_next(ordinal)) {
      if (++actorsVisited % 256 == 0 && reachedExecutionDeadline()) {
        stopAtExecutionDeadline();
        return DfgAdvanceStop::ExecutionLimit;
      }
      const ActorExecutionPlan &plan = run.execution.actorPlans[ordinal];
      mlir::Operation *operation = plan.operation;
      ActorTransitionCommitOutcome outcome =
          commitActorTransition(plan, run.state);
      if (outcome == ActorTransitionCommitOutcome::Committed)
        scheduleActor(run.state, static_cast<unsigned>(ordinal));
      if (run.state.failure != RunFailure::None)
        return DfgAdvanceStop::Stopped;
      if (run.retirementObserved &&
          outcome != ActorTransitionCommitOutcome::NotReady) {
        run.report.status = "invalid";
        run.report.diagnostics.push_back(
            ("actor '" + operation->getName().getStringRef() +
             (outcome == ActorTransitionCommitOutcome::Committed
                  ? "' fired after graph retirement"
                  : "' failed after graph retirement"))
                .str());
        return DfgAdvanceStop::Stopped;
      }
      fired |= outcome == ActorTransitionCommitOutcome::Committed;
    }
    if (run.report.status != "pass" || run.state.failure != RunFailure::None)
      return DfgAdvanceStop::Stopped;
    if (!fired)
      return run.retirementObserved ? DfgAdvanceStop::Retired
                                    : DfgAdvanceStop::Stopped;

    flushPendingTokens(run.state);
    ++run.report.wavefrontSteps;
    observeRetirement(run);
  }

  return run.report.status == "pass" ? DfgAdvanceStop::Yielded
                                     : DfgAdvanceStop::Stopped;
}

llvm::Expected<DFGSimulationReport>
finalizeDfgRun(DfgRun &run, const CanonicalSimulationWorkload *typedWorkload,
               const CanonicalSimulationRuntimeInput *typedRuntimeInput,
               const ResolvedLaunchContext *typedContext,
               const dataflow::CanonicalDataflowProgramView *typedProgramView,
               SpatialFunctionalObservations *retiredObservations) {
  if (run.finalized)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "DFG execution was already finalized");
  run.finalized = true;

  bool missingReturn = false;
  bool pendingVectorGroups = false;
  if (!applyRunFailureTerminal(run.state, run.report)) {
    if (!run.retirementObserved) {
      run.report.finalOutputs.push_back("missing");
      missingReturn = true;
    } else {
      mlir::Value witness = run.execution.returnObservation.complete.front();
      auto serialized =
          tokenToString(run.state.observedOutputs.find(witness)->second.front(),
                        witness.getType(), run.graph);
      if (!serialized)
        return serialized.takeError();
      run.report.finalOutputs.push_back(std::move(*serialized));
    }
    for (mlir::Value value : run.execution.returnObservation.values) {
      auto it = run.state.observedOutputs.find(value);
      if (it == run.state.observedOutputs.end() || it->second.empty()) {
        run.report.finalOutputs.push_back("missing");
        missingReturn = true;
        continue;
      }
      auto serialized =
          tokenToString(it->second.front(), value.getType(), run.graph);
      if (!serialized)
        return serialized.takeError();
      run.report.finalOutputs.push_back(std::move(*serialized));
    }
    for (mlir::Value stream : run.execution.returnObservation.streams) {
      llvm::SmallVector<std::string> tokens;
      auto it = run.state.observedOutputs.find(stream);
      if (it != run.state.observedOutputs.end())
        for (const Token &token : it->second) {
          auto serialized = tokenToString(token, stream.getType(), run.graph);
          if (!serialized)
            return serialized.takeError();
          tokens.push_back(std::move(*serialized));
        }
      run.report.finalStreamOutputs.push_back(std::move(tokens));
    }
    if (!typedWorkload)
      if (llvm::Error error =
              captureFinalMemoryState(run.graph, run.state, run.report))
        return std::move(error);
    pendingVectorGroups = hasPendingVectorGroups(run.state);
  }
  if (run.report.status == "pass" && !run.retirementObserved) {
    run.report.status = "blocked";
    run.report.diagnostics.push_back(
        "graph did not fire its retirement frontier");
  }
  if (run.report.status == "pass" && !run.state.diagnostics.empty()) {
    run.report.status = "blocked";
    run.report.diagnostics.push_back(
        "DFG-sim stopped with runtime diagnostics");
  }
  if (run.report.status == "pass" && (missingReturn || pendingVectorGroups)) {
    run.report.status = run.retirementObserved ? "invalid" : "blocked";
    run.report.diagnostics.push_back(
        run.retirementObserved
            ? "graph retired with incomplete internal state"
            : "graph stopped before retirement outputs were complete");
  }
  projectRunObservations(run.state, run.report);
  if (retiredObservations && run.report.status == "pass") {
    if (!typedWorkload || !typedRuntimeInput || !typedContext ||
        !typedProgramView)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "typed DFG observation projection has no admitted owner context");
    auto observations = projectRetiredFunctionalObservations(
        run.graph, run.state, *typedWorkload, *typedRuntimeInput, *typedContext,
        *typedProgramView);
    if (!observations)
      return observations.takeError();
    *retiredObservations = std::move(*observations);
  }
  return std::move(run.report);
}

} // namespace

struct loom::sim::DfgExecutionSession::Impl {
  const CanonicalSimulationWorkload *workload = nullptr;
  const CanonicalSimulationRuntimeInput *runtimeInput = nullptr;
  const ResolvedLaunchContext *context = nullptr;
  const dataflow::CanonicalDataflowProgramView *programView = nullptr;
  SimulatorState dynamicState;
  DFGSimulationReport report;
  DfgRun run;
  DfgExecutionSessionState lifecycle = DfgExecutionSessionState::Runnable;
  bool resultTaken = false;

  Impl(dataflow::GraphOp graph, const PreparedGraphExecution &execution,
       DFGSimulationReport initialReport,
       const CanonicalSimulationWorkload &workload,
       const CanonicalSimulationRuntimeInput &runtimeInput,
       const ResolvedLaunchContext &context,
       const dataflow::CanonicalDataflowProgramView &programView)
      : workload(&workload), runtimeInput(&runtimeInput), context(&context),
        programView(&programView), report(std::move(initialReport)),
        run{graph, execution, dynamicState, report} {}
};

loom::sim::DfgExecutionSession::DfgExecutionSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

loom::sim::DfgExecutionSession::DfgExecutionSession(
    DfgExecutionSession &&) noexcept = default;

loom::sim::DfgExecutionSession &loom::sim::DfgExecutionSession::operator=(
    DfgExecutionSession &&) noexcept = default;

loom::sim::DfgExecutionSession::~DfgExecutionSession() = default;

loom::sim::DfgExecutionSessionState
loom::sim::DfgExecutionSession::state() const {
  return impl_ ? impl_->lifecycle : DfgExecutionSessionState::Failed;
}

std::uint64_t loom::sim::DfgExecutionSession::wavefrontSteps() const {
  return impl_ ? impl_->report.wavefrontSteps : 0;
}

llvm::Expected<loom::sim::DfgExecutionSessionState>
loom::sim::DfgExecutionSession::advance(
    std::uint64_t maxWavefrontSteps,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline) {
  if (!impl_)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "DFG execution session is empty");
  if (impl_->resultTaken)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "DFG execution result was already taken");
  if (impl_->lifecycle != DfgExecutionSessionState::Runnable)
    return impl_->lifecycle;
  if (maxWavefrontSteps == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "DFG execution advance requires a positive wavefront budget");

  switch (advanceDfgRun(impl_->run, maxWavefrontSteps, executionDeadline)) {
  case DfgAdvanceStop::Yielded:
    return impl_->lifecycle;
  case DfgAdvanceStop::Retired:
    impl_->lifecycle = DfgExecutionSessionState::Retired;
    return impl_->lifecycle;
  case DfgAdvanceStop::Stopped:
    impl_->lifecycle = DfgExecutionSessionState::Failed;
    return impl_->lifecycle;
  case DfgAdvanceStop::ExecutionLimit:
    impl_->lifecycle = DfgExecutionSessionState::StoppedByLimit;
    return llvm::createStringError(std::errc::timed_out,
                                   "DFG execution wall-time limit reached");
  }
  llvm_unreachable("closed DFG advance stop");
}

llvm::Expected<loom::sim::DfgExecutionSession>
loom::sim::startDfgExecutionSession(
    const PreparedDfgExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  if (!prepared.impl_)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "prepared DFG execution is empty");
  const SpatialSimulationWorkload *spatial = workload.spatial();
  if (!spatial)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "DFG execution session requires a Spatial workload");
  if (spatial->launchRef != prepared.impl_->launch)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "runtime workload does not name the prepared rooted graph launch");

  auto graphRef =
      admitDfgSpatialSimulation(workload, runtimeInput, prepared.impl_->view);
  if (!graphRef)
    return graphRef.takeError();
  if (*graphRef != prepared.impl_->context.graph)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "runtime workload resolves to a different prepared graph");
  auto graphView = prepared.impl_->view.resolve(*graphRef);
  if (!graphView)
    return graphView.takeError();
  dataflow::GraphOp graph = mlir::cast<dataflow::GraphOp>(graphView->op);
  if (std::optional<std::string> reason = unsupportedTypedDfgInput(
          workload, runtimeInput, prepared.impl_->context))
    return llvm::createStringError(std::errc::not_supported, "%s",
                                   reason->c_str());

  DFGSimulationReport report;
  report.graph = graph.getSymName().str();
  report.workload = formatArtifactIdentityHex(workload.identity());
  report.status = "pass";
  auto impl = std::make_unique<DfgExecutionSession::Impl>(
      graph, prepared.impl_->execution, std::move(report), workload,
      runtimeInput, prepared.impl_->context, prepared.impl_->view);
  if (llvm::Error error = initializeTypedGraphExecutionState(
          impl->dynamicState, prepared.impl_->execution, graph, workload,
          runtimeInput, prepared.impl_->context))
    return std::move(error);
  observeRetirement(impl->run);
  if (impl->report.status != "pass")
    impl->lifecycle = DfgExecutionSessionState::Failed;
  return DfgExecutionSession(std::move(impl));
}

llvm::Expected<loom::sim::RetiredDFGSimulation>
loom::sim::DfgExecutionSession::takeRetiredSimulation() {
  if (!impl_)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "DFG execution session is empty");
  if (impl_->resultTaken)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "DFG execution result was already taken");
  if (impl_->lifecycle != DfgExecutionSessionState::Retired)
    return llvm::createStringError(
        std::errc::state_not_recoverable,
        "DFG execution session has not retired successfully");

  SpatialFunctionalObservations observations;
  auto report =
      finalizeDfgRun(impl_->run, impl_->workload, impl_->runtimeInput,
                     impl_->context, impl_->programView, &observations);
  impl_->resultTaken = true;
  if (!report)
    return report.takeError();
  if (report->status != "pass") {
    std::string message = "DFG execution did not retire: " + report->status;
    if (!report->diagnostics.empty())
      message += ": " + report->diagnostics.front();
    return llvm::createStringError(std::errc::state_not_recoverable, "%s",
                                   message.c_str());
  }
  return RetiredDFGSimulation{std::move(*report), std::move(observations)};
}

static llvm::Expected<DFGSimulationReport> simulateDataflowGraphImpl(
    mlir::ModuleOp module, const DFGSimulationOptions &options,
    dataflow::GraphOp admittedGraph,
    const CanonicalSimulationWorkload *typedWorkload,
    const CanonicalSimulationRuntimeInput *typedRuntimeInput,
    const ResolvedLaunchContext *typedContext,
    const dataflow::CanonicalDataflowProgramView *typedProgramView,
    SpatialFunctionalObservations *retiredObservations,
    const PreparedGraphExecution *preparedExecution) {
  DFGSimulationReport report;
  report.graph = options.graphName;
  report.workload =
      options.workloadName.empty() ? options.graphName : options.workloadName;
  report.status = "pass";

  dataflow::GraphOp graph =
      admittedGraph ? admittedGraph : findGraph(module, options.graphName);
  if (!graph) {
    report.status = "unsupported";
    report.diagnostics.push_back(
        llvm::formatv("dataflow.graph '{0}' was not found", options.graphName)
            .str());
    return report;
  }
  if (graph.isExternal()) {
    report.status = "unsupported";
    report.diagnostics.push_back(
        llvm::formatv("dataflow.graph '{0}' is external", options.graphName)
            .str());
    return report;
  }

  std::optional<PreparedGraphExecution> ownedExecution;
  if (preparedExecution && preparedExecution->graph != graph)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "prepared DFG execution does not belong to the admitted graph");
  if (!preparedExecution) {
    auto prepared = prepareGraphExecution(module, graph);
    if (!prepared)
      return prepared.takeError();
    if (auto *failure = std::get_if<GraphPreparationFailure>(&*prepared)) {
      report.status = failure->status;
      report.diagnostics = std::move(failure->diagnostics);
      return report;
    }
    ownedExecution.emplace(
        std::move(std::get<PreparedGraphExecution>(*prepared)));
    preparedExecution = &*ownedExecution;
  }

  llvm::ArrayRef<int32_t> resultSegments = graph.getResultSegmentSizes();
  if (options.invocations == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "invocations must be nonzero");
  if (options.invocations > 1) {
    if (resultSegments[0] != 0 || resultSegments[1] != 0)
      return llvm::createStringError(
          std::errc::not_supported,
          "multiple invocations with value or stream results are unsupported");

    unsigned applicationInputCount = graph.getFunctionType().getNumInputs();
    auto groupedArgsOrErr =
        indexRuntimeArgs(options.args, applicationInputCount);
    if (!groupedArgsOrErr)
      return groupedArgsOrErr.takeError();
    llvm::StringMap<llvm::SmallVector<std::string>> groupedArgs =
        std::move(*groupedArgsOrErr);
    for (unsigned index = 0; index < applicationInputCount; ++index) {
      dataflow::GraphPortKind kind = graph.getInputPortKind(index);
      if (kind == dataflow::GraphPortKind::Memory)
        continue;
      if (kind == dataflow::GraphPortKind::Stream)
        return llvm::createStringError(
            std::errc::not_supported,
            "multiple invocations with stream inputs are unsupported");
      std::string key = std::to_string(index);
      auto it = groupedArgs.find(key);
      if (it == groupedArgs.end() || it->second.size() != options.invocations)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "value argument %u requires exactly one token per invocation",
            index);
    }

    DFGSimulationReport aggregate = report;
    llvm::SmallVector<DFGMemoryArg> currentMemories = options.memories;
    for (const DFGMemoryArg &memory : currentMemories)
      if (memory.byteOffset != 0)
        return llvm::createStringError(
            std::errc::not_supported,
            "multiple invocations with nonzero memory fixture offsets are "
            "unsupported");

    for (std::uint64_t invocation = 0; invocation < options.invocations;
         ++invocation) {
      DFGSimulationOptions single = options;
      single.invocations = 1;
      single.args.clear();
      single.memories = currentMemories;
      for (unsigned index = 0; index < applicationInputCount; ++index) {
        if (graph.getInputPortKind(index) != dataflow::GraphPortKind::Value)
          continue;
        std::string key = std::to_string(index);
        single.args.push_back({index, groupedArgs.lookup(key)[invocation]});
      }

      auto singleReportOrErr = simulateDataflowGraph(module, single);
      if (!singleReportOrErr)
        return singleReportOrErr.takeError();
      DFGSimulationReport singleReport = std::move(*singleReportOrErr);
      aggregate.wavefrontSteps += singleReport.wavefrontSteps;
      aggregate.eventCount += singleReport.eventCount;
      aggregate.dynamicWorkItems += singleReport.dynamicWorkItems;
      for (const auto &[name, count] : singleReport.operationFireCounts)
        aggregate.operationFireCounts[name] += count;
      for (const auto &[name, count] : singleReport.modeledLibraryCalls)
        aggregate.modeledLibraryCalls[name] += count;
      aggregate.finalOutputs = std::move(singleReport.finalOutputs);
      aggregate.finalMemoryState = std::move(singleReport.finalMemoryState);
      aggregate.finalMemoryRoots = std::move(singleReport.finalMemoryRoots);

      for (const std::string &diagnostic : singleReport.diagnostics)
        aggregate.diagnostics.push_back(
            llvm::formatv("invocation {0}: {1}", invocation, diagnostic).str());
      if (singleReport.status != "pass") {
        aggregate.status = singleReport.status;
        break;
      }

      for (DFGMemoryArg &memory : currentMemories) {
        std::string key = llvm::formatv("arg{0}", memory.index).str();
        auto stateIt = aggregate.finalMemoryState.find(key);
        if (stateIt == aggregate.finalMemoryState.end())
          return llvm::createStringError(
              std::errc::invalid_argument,
              "memory argument %u was not materialized by invocation %llu",
              memory.index, static_cast<unsigned long long>(invocation));
        auto fixtureOrErr = memoryFixtureFromSerializedValues(stateIt->second);
        if (!fixtureOrErr)
          return fixtureOrErr.takeError();
        memory.values = std::move(*fixtureOrErr);
      }
    }

    return aggregate;
  }

  mlir::Block &entry = graph.getBody().front();
  const unsigned applicationInputCount =
      preparedExecution->applicationInputCount;
  llvm::StringMap<llvm::SmallVector<std::string>> args;
  llvm::StringMap<MemoryFixture> memories;
  if (!typedWorkload) {
    auto argsOrErr = indexRuntimeArgs(options.args, applicationInputCount);
    if (!argsOrErr)
      return argsOrErr.takeError();
    args = std::move(*argsOrErr);
    auto memoriesOrErr =
        indexMemoryArgs(options.memories, applicationInputCount);
    if (!memoriesOrErr)
      return memoriesOrErr.takeError();
    memories = std::move(*memoriesOrErr);
  }

  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *preparedExecution);
  seedBlockArgument(state, graph.getStart(), noneToken());

  if (typedWorkload) {
    if (!typedRuntimeInput || !typedContext)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "typed DFG execution is missing admitted runtime context");
    if (llvm::Error error = seedTypedDfgInputs(
            state, graph, *typedWorkload, *typedRuntimeInput, *typedContext))
      return std::move(error);
  } else {
    for (unsigned index = 0; index < applicationInputCount; ++index) {
      mlir::BlockArgument arg = entry.getArgument(index + 1);
      std::string key = std::to_string(index);
      dataflow::GraphPortKind kind = graph.getInputPortKind(index);
      if (kind == dataflow::GraphPortKind::Memory) {
        if (!memories.contains(key))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "missing memory fixture for argument %u", unsigned(index));
        if (args.contains(key))
          return llvm::createStringError(std::errc::invalid_argument,
                                         "memory argument %u must use --memref",
                                         unsigned(index));
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(arg.getType())) {
          if (memories.lookup(key).byteOffset != 0)
            return llvm::createStringError(
                std::errc::invalid_argument,
                "memref argument %u cannot use a nonzero memory fixture byte "
                "offset",
                unsigned(index));
          auto tokensOrErr = parseMemoryTokens(
              memories.lookup(key).values, memrefType.getElementType(), graph);
          if (!tokensOrErr)
            return llvm::joinErrors(
                llvm::createStringError(std::errc::invalid_argument,
                                        "invalid memref argument %u",
                                        unsigned(index)),
                tokensOrErr.takeError());
          llvm::SmallVector<SemanticMemoryByte> bytes;
          for (const Token &token : *tokensOrErr) {
            auto encoded = encodeMemoryElement(
                token, memrefType.getElementType(), state.graphScope);
            if (!encoded)
              return encoded.takeError();
            bytes.append(encoded->begin(), encoded->end());
          }
          llvm::SmallBitVector initialized(bytes.size(), /*t=*/true);
          auto [rootIt, inserted] =
              state.memoryRootIds.try_emplace(arg, state.nextMemoryRootId);
          if (inserted)
            ++state.nextMemoryRootId;
          auto memory = std::make_shared<MemoryValue>(MemoryValue{
              rootIt->second, std::move(bytes), std::move(initialized), {}});
          state.memories[arg] = memory;
          state.memoryViews[arg] =
              MemoryView{memory, arg, 0, memrefType.getElementType()};
        } else {
          if (!state.memoryRootIds.contains(arg))
            state.memoryRootIds[arg] = state.nextMemoryRootId++;
          state.rawMemoryFixtures[arg] = memories.lookup(key);
        }
        continue;
      }

      if (memories.contains(key))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "value argument %u must not use --memref", unsigned(index));
      auto argIt = args.find(key);
      if (argIt == args.end())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "missing runtime argument %u",
                                       unsigned(index));
      if (kind == dataflow::GraphPortKind::Value && argIt->second.size() != 1)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "value argument %u requires exactly one token", unsigned(index));
      for (llvm::StringRef rawToken : argIt->second) {
        auto tokenOrErr = parseRuntimeToken(rawToken, arg.getType(), graph);
        if (!tokenOrErr)
          return llvm::joinErrors(
              llvm::createStringError(std::errc::invalid_argument,
                                      "invalid argument %u", unsigned(index)),
              tokenOrErr.takeError());
        seedBlockArgument(state, arg, *tokenOrErr);
      }
    }
  }

  if (llvm::Error err = initializeFreshMemoryRoots(entry, state))
    return std::move(err);
  if (llvm::Error err = propagateMemoryAliases(entry, state))
    return std::move(err);

  DfgRun run{graph, *preparedExecution, state, report};
  observeRetirement(run);
  const DfgAdvanceStop stop =
      advanceDfgRun(run, options.maxEventSteps, options.executionDeadline);
  if (stop == DfgAdvanceStop::Yielded && !run.retirementObserved &&
      state.failure == RunFailure::None) {
    report.status = "blocked";
    report.diagnostics.push_back("maximum event steps reached");
  }
  return finalizeDfgRun(run, typedWorkload, typedRuntimeInput, typedContext,
                        typedProgramView, retiredObservations);
}

llvm::Expected<DFGSimulationReport>
loom::sim::simulateDataflowGraph(mlir::ModuleOp module,
                                 const DFGSimulationOptions &options) {
  return simulateDataflowGraphImpl(module, options, {}, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr);
}

static llvm::Expected<DFGSimulationReport> simulateTypedDfgWorkload(
    const dataflow::CanonicalDataflowArtifact &program,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventSteps,
    SpatialFunctionalObservations *retiredObservations,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline =
        std::nullopt,
    const dataflow::CanonicalDataflowProgramView *preparedView = nullptr,
    const ResolvedLaunchContext *preparedContext = nullptr,
    const PreparedGraphExecution *preparedExecution = nullptr,
    const dataflow::RootedGraphLaunchRef *preparedLaunch = nullptr) {
  std::optional<dataflow::CanonicalDataflowProgramView> ownedView;
  if (!preparedView) {
    auto view = program.view();
    if (!view)
      return view.takeError();
    ownedView.emplace(std::move(*view));
    preparedView = &*ownedView;
  }
  if (preparedLaunch && workload.spatial()->launchRef != *preparedLaunch)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "runtime workload does not name the prepared rooted graph launch");
  llvm::Expected<dataflow::GraphRef> graphRef =
      admitDfgSpatialSimulation(workload, runtimeInput, *preparedView);
  if (!graphRef)
    return graphRef.takeError();
  std::optional<ResolvedLaunchContext> ownedContext;
  if (!preparedContext) {
    auto context =
        resolveLaunchContext(*preparedView, workload.spatial()->launchRef);
    if (!context)
      return context.takeError();
    ownedContext.emplace(std::move(*context));
    preparedContext = &*ownedContext;
  }
  if (preparedContext->graph != *graphRef)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "runtime workload resolves to a different prepared graph");
  llvm::Expected<dataflow::CanonicalGraphView> graph =
      preparedView->resolve(*graphRef);
  if (!graph)
    return graph.takeError();

  DFGSimulationOptions options;
  options.graphName =
      mlir::cast<dataflow::GraphOp>(graph->op).getSymName().str();
  options.workloadName = formatArtifactIdentityHex(workload.identity());
  options.maxEventSteps = maxEventSteps;
  options.executionDeadline = executionDeadline;
  if (std::optional<std::string> reason =
          unsupportedTypedDfgInput(workload, runtimeInput, *preparedContext)) {
    DFGSimulationReport report;
    report.graph = options.graphName;
    report.workload = options.workloadName;
    report.status = "unsupported";
    report.diagnostics.push_back(std::move(*reason));
    return report;
  }
  return simulateDataflowGraphImpl(
      program.module(), options, mlir::cast<dataflow::GraphOp>(graph->op),
      &workload, &runtimeInput, preparedContext, preparedView,
      retiredObservations, preparedExecution);
}

llvm::Expected<PreparedDfgExecution> loom::sim::prepareDfgExecution(
    const dataflow::CanonicalDataflowArtifact &program,
    const dataflow::RootedGraphLaunchRef &launch) {
  auto imported = dataflow::importCanonicalDataflow(program.identity(),
                                                    program.canonicalBytes());
  if (!imported)
    return imported.takeError();
  auto owned = std::make_unique<dataflow::CanonicalDataflowArtifact>(
      std::move(*imported));
  auto view = owned->view();
  if (!view)
    return view.takeError();
  auto context = resolveLaunchContext(*view, launch);
  if (!context)
    return context.takeError();
  auto graphView = view->resolve(context->graph);
  if (!graphView)
    return graphView.takeError();
  auto prepared = prepareGraphExecution(
      owned->module(), mlir::cast<dataflow::GraphOp>(graphView->op));
  if (!prepared)
    return prepared.takeError();
  if (auto *failure = std::get_if<GraphPreparationFailure>(&*prepared)) {
    std::string message;
    for (const std::string &diagnostic : failure->diagnostics) {
      if (!message.empty())
        message += "; ";
      message += diagnostic;
    }
    return llvm::createStringError(failure->status == "unsupported"
                                       ? std::errc::not_supported
                                       : std::errc::invalid_argument,
                                   "%s", message.c_str());
  }
  return PreparedDfgExecution(
      std::make_unique<PreparedDfgExecution::Impl>(PreparedDfgExecution::Impl{
          std::move(owned), launch, std::move(*view), std::move(*context),
          std::move(std::get<PreparedGraphExecution>(*prepared))}));
}

llvm::Expected<DFGSimulationReport> loom::sim::simulateDfgWorkload(
    const dataflow::CanonicalDataflowArtifact &program,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventSteps) {
  return simulateTypedDfgWorkload(program, workload, runtimeInput,
                                  maxEventSteps, nullptr);
}

static llvm::Expected<RetiredDFGSimulation>
requireRetiredDfgExecution(llvm::Expected<DFGSimulationReport> report,
                           SpatialFunctionalObservations observations,
                           std::uint64_t maxEventSteps) {
  if (!report)
    return report.takeError();
  if (report->status == "pass")
    return RetiredDFGSimulation{std::move(*report), std::move(observations)};

  std::string message = "DFG execution did not retire: " + report->status;
  if (!report->diagnostics.empty())
    message += ": " + report->diagnostics.front();
  if (report->status == "unsupported")
    return llvm::createStringError(std::errc::not_supported, "%s",
                                   message.c_str());
  if (report->status == "execution_limit")
    return llvm::createStringError(std::errc::timed_out, "%s", message.c_str());
  if (report->status == "blocked" && report->wavefrontSteps == maxEventSteps &&
      llvm::is_contained(report->diagnostics, "maximum event steps reached"))
    return llvm::createStringError(std::errc::timed_out, "%s", message.c_str());
  if (report->status == "blocked" || report->status == "invalid")
    return llvm::make_error<NonRetiredDFGExecutionError>(std::move(*report));
  return llvm::createStringError(std::errc::state_not_recoverable, "%s",
                                 message.c_str());
}

llvm::Expected<RetiredDFGSimulation> loom::sim::simulateRetiredDfgWorkload(
    const dataflow::CanonicalDataflowArtifact &program,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventSteps,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline) {
  const SpatialSimulationWorkload *spatial = workload.spatial();
  if (!spatial)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "retired DFG execution requires a Spatial workload");
  auto prepared = prepareDfgExecution(program, spatial->launchRef);
  if (!prepared)
    return prepared.takeError();
  return simulateRetiredDfgWorkload(*prepared, workload, runtimeInput,
                                    maxEventSteps, executionDeadline);
}

llvm::Expected<RetiredDFGSimulation> loom::sim::simulateRetiredDfgWorkload(
    const PreparedDfgExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventSteps,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline) {
  if (maxEventSteps == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "retired DFG execution requires a positive wavefront budget");
  auto session = startDfgExecutionSession(prepared, workload, runtimeInput);
  if (!session)
    return session.takeError();
  auto state = session->advance(maxEventSteps, executionDeadline);
  if (!state)
    return state.takeError();
  if (*state == DfgExecutionSessionState::Retired)
    return session->takeRetiredSimulation();

  SpatialFunctionalObservations observations;
  if (*state == DfgExecutionSessionState::Runnable) {
    session->impl_->report.status = "blocked";
    session->impl_->report.diagnostics.push_back("maximum event steps reached");
    session->impl_->lifecycle = DfgExecutionSessionState::StoppedByLimit;
  }
  auto report =
      finalizeDfgRun(session->impl_->run, session->impl_->workload,
                     session->impl_->runtimeInput, session->impl_->context,
                     session->impl_->programView, &observations);
  session->impl_->resultTaken = true;
  return requireRetiredDfgExecution(std::move(report), std::move(observations),
                                    maxEventSteps);
}
