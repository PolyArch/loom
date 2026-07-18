#include "DFGSimulatorInternal.h"

#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <utility>

using namespace loom::sim::detail;

namespace {

constexpr llvm::StringLiteral fixture = R"mlir(
module {
  func.func @parallelize(%data: i8, %phase: i1) {
    %vector, %mask, %group_phase =
      dataflow.parallelize %data, %phase
        : (i8, i1) -> (vector<2xi8>, vector<2xi1>, i1)
    %packed = dataflow.pack %vector : vector<2xi8> -> i16
    return
  }

  func.func @serialize(%packed: i32, %packed_mask: i4, %group_phase: i1) {
    %vector = dataflow.unpack %packed : i32 -> vector<4xi8>
    %mask = dataflow.unpack %packed_mask : i4 -> vector<4xi1>
    %data, %scalar_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<4xi8>, vector<4xi1>, i1) -> (i8, i1)
    return
  }

  func.func @structured_vector_memory(
      %take: i1, %start: none, %idx: index, %value: vector<4xi8>,
      %mem: memref<?xi8>) -> (vector<4xi8>, none) {
    %result:2 = scf.if %take -> (vector<4xi8>, none) {
      %stored = dataflow.store %mem[%idx] %value %start
          : memref<?xi8>, vector<4xi8>
      %loaded, %done = dataflow.load %mem[%idx] %stored
          : memref<?xi8>, vector<4xi8>
      scf.yield %loaded, %done : vector<4xi8>, none
    } else {
      scf.yield %value, %start : vector<4xi8>, none
    }
    return %result#0, %result#1 : vector<4xi8>, none
  }
}
)mlir";

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DFGVectorBoundaryTest: " << message << "\n";
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T takeExpected(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

Token tokenWithBits(mlir::Type type, uint64_t value) {
  unsigned width = takeExpected(tokenTypeBitWidth(type));
  return takeExpected(tokenFromBitPattern(llvm::APInt(width, value), type));
}

Token malformedToken(TokenKind kind, unsigned width) {
  Token token;
  token.kind = kind;
  token.bitPattern = llvm::APInt(width, 0);
  return token;
}

uint64_t bitsOf(const Token &token, mlir::Type type) {
  return takeExpected(tokenBitPattern(token, type)).getZExtValue();
}

void expectBits(llvm::ArrayRef<Token> tokens, mlir::Type type,
                std::initializer_list<uint64_t> expected,
                llvm::StringRef message) {
  require(tokens.size() == expected.size(), message);
  for (auto [token, value] : llvm::zip_equal(tokens, expected))
    require(bitsOf(token, type) == value, message);
}

void expectPhases(llvm::ArrayRef<Token> tokens,
                  std::initializer_list<bool> expected,
                  llvm::StringRef message) {
  require(tokens.size() == expected.size(), message);
  for (auto [token, value] : llvm::zip_equal(tokens, expected))
    require(boolToken(token) == value, message);
}

void flushPending(SimulatorState &state) {
  for (auto &entry : state.pendingChannels) {
    auto &target = state.channels[entry.first];
    while (!entry.second.empty()) {
      target.push_back(entry.second.front());
      entry.second.pop_front();
    }
  }
  state.pendingChannels.clear();
  for (auto &entry : state.pendingObservedOutputs) {
    auto &target = state.observedOutputs[entry.first];
    target.append(entry.second.begin(), entry.second.end());
  }
  state.pendingObservedOutputs.clear();
}

void parallelizePreservesQueuedActivation(dataflow::ParallelizeOp op) {
  SimulatorState state;
  auto &data = state.channels[&op.getDataMutable()];
  data.push_back(tokenWithBits(op.getData().getType(), 17));
  data.push_back(tokenWithBits(op.getData().getType(), 18));
  auto &phase = state.channels[&op.getScalarPhaseMutable()];
  phase.push_back(boolValueToken(true));
  phase.push_back(boolValueToken(false));
  phase.push_back(boolValueToken(true));
  phase.push_back(boolValueToken(false));

  require(fireActorOperation(op, state), "first scalar true did not fire");
  require(data.size() == 1, "first scalar true consumed the wrong payload");
  require(fireActorOperation(op, state), "first scalar false did not fire");
  require(data.size() == 1,
          "scalar false consumed the next activation payload");
  require(fireActorOperation(op, state), "second scalar true did not fire");
  require(fireActorOperation(op, state), "second scalar false did not fire");

  expectBits(state.pendingObservedOutputs[op.getVector()],
             op.getVector().getType(), {17, 18},
             "parallelize did not reset zero-filled lanes");
  expectBits(state.pendingObservedOutputs[op.getMask()], op.getMask().getType(),
             {1, 1}, "parallelize did not reset active masks");
  expectPhases(state.pendingObservedOutputs[op.getGroupPhase()],
               {true, false, true, false},
               "parallelize emitted the wrong activation phases");
}

void serializePreservesQueuedActivation(dataflow::SerializeOp op,
                                        dataflow::UnpackOp vectorUnpack,
                                        dataflow::UnpackOp maskUnpack) {
  SimulatorState state;
  auto &packed = state.channels[&vectorUnpack.getPackedMutable()];
  packed.push_back(
      tokenWithBits(vectorUnpack.getPacked().getType(), 0x44332211U));
  packed.push_back(
      tokenWithBits(vectorUnpack.getPacked().getType(), 0x44332211U));
  auto &packedMask = state.channels[&maskUnpack.getPackedMutable()];
  packedMask.push_back(tokenWithBits(maskUnpack.getPacked().getType(), 0));
  packedMask.push_back(tokenWithBits(maskUnpack.getPacked().getType(), 5));

  require(fireActorOperation(vectorUnpack, state),
          "first vector unpack did not fire");
  require(fireActorOperation(vectorUnpack, state),
          "second vector unpack did not fire");
  require(fireActorOperation(maskUnpack, state),
          "first mask unpack did not fire");
  require(fireActorOperation(maskUnpack, state),
          "second mask unpack did not fire");
  flushPending(state);

  auto &vectors = state.channels[&op.getVectorMutable()];
  auto &masks = state.channels[&op.getMaskMutable()];
  require(vectors.size() == 2 && masks.size() == 2,
          "unpack did not queue both activation payloads");
  auto &phase = state.channels[&op.getGroupPhaseMutable()];
  phase.push_back(boolValueToken(true));
  phase.push_back(boolValueToken(false));
  phase.push_back(boolValueToken(true));
  phase.push_back(boolValueToken(false));

  require(fireActorOperation(op, state), "all-zero true group did not fire");
  require(vectors.size() == 1 && masks.size() == 1,
          "all-zero true group did not consume its payload");
  require(state.pendingObservedOutputs[op.getData()].empty(),
          "all-zero group emitted scalar data");
  require(fireActorOperation(op, state), "first group false did not fire");
  require(vectors.size() == 1 && masks.size() == 1,
          "group false consumed the next activation payload");
  require(fireActorOperation(op, state), "sparse true group did not fire");
  require(fireActorOperation(op, state), "second group false did not fire");

  expectBits(state.pendingObservedOutputs[op.getData()], op.getData().getType(),
             {0x11, 0x33}, "serialize did not preserve low-slice lane order");
  expectPhases(state.pendingObservedOutputs[op.getScalarPhase()],
               {false, true, true, false},
               "serialize emitted the wrong activation phases");
}

void parallelizeFailureIsAtomic(dataflow::ParallelizeOp op) {
  {
    SimulatorState state;
    state.channels[&op.getDataMutable()].push_back(
        malformedToken(TokenKind::Integer, 16));
    state.channels[&op.getScalarPhaseMutable()].push_back(boolValueToken(true));

    require(!fireActorOperation(op, state),
            "parallelize accepted a malformed scalar token");
    require(state.channels[&op.getDataMutable()].size() == 1 &&
                state.channels[&op.getScalarPhaseMutable()].size() == 1,
            "parallelize consumed input on conversion failure");
    require(!state.parallelizeStates.contains(op.getOperation()),
            "parallelize changed actor state on conversion failure");
    require(state.pendingChannels.empty() &&
                state.pendingObservedOutputs.empty() &&
                state.actorMutationEpoch == 0,
            "parallelize published output on conversion failure");
  }

  {
    SimulatorState state;
    ParallelizeState pending;
    pending.semanticState.pendingItems = 1;
    pending.slots.resize(2);
    pending.slots[0] = malformedToken(TokenKind::Integer, 16);
    state.parallelizeStates[op.getOperation()] = pending;
    state.channels[&op.getScalarPhaseMutable()].push_back(
        boolValueToken(false));

    require(!fireActorOperation(op, state),
            "parallelize assembled a malformed pending group");
    const ParallelizeState &preserved =
        state.parallelizeStates.find(op.getOperation())->second;
    require(preserved.semanticState.pendingItems == 1 && preserved.slots[0] &&
                preserved.slots[0]->bitPattern->getBitWidth() == 16,
            "parallelize changed pending state on group construction failure");
    require(state.channels[&op.getScalarPhaseMutable()].size() == 1,
            "parallelize consumed phase on group construction failure");
    require(state.pendingChannels.empty() &&
                state.pendingObservedOutputs.empty() &&
                state.actorMutationEpoch == 0,
            "parallelize published a malformed pending group");
  }
}

void packFailureIsAtomic(dataflow::PackOp op) {
  SimulatorState state;
  state.channels[&op.getVectorMutable()].push_back(
      malformedToken(TokenKind::Vector, 8));

  require(!fireActorOperation(op, state), "pack accepted a malformed vector");
  require(state.channels[&op.getVectorMutable()].size() == 1,
          "pack consumed input on conversion failure");
  require(state.pendingChannels.empty() &&
              state.pendingObservedOutputs.empty() &&
              state.actorMutationEpoch == 0,
          "pack published output on conversion failure");
}

void unpackFailureIsAtomic(dataflow::UnpackOp op) {
  SimulatorState state;
  state.channels[&op.getPackedMutable()].push_back(
      malformedToken(TokenKind::Integer, 8));

  require(!fireActorOperation(op, state),
          "unpack accepted a malformed packed token");
  require(state.channels[&op.getPackedMutable()].size() == 1,
          "unpack consumed input on conversion failure");
  require(state.pendingChannels.empty() &&
              state.pendingObservedOutputs.empty() &&
              state.actorMutationEpoch == 0,
          "unpack published output on conversion failure");
}

void serializeFailureIsAtomic(dataflow::SerializeOp op) {
  SimulatorState state;
  state.channels[&op.getVectorMutable()].push_back(
      malformedToken(TokenKind::Vector, 8));
  state.channels[&op.getMaskMutable()].push_back(
      tokenWithBits(op.getMask().getType(), 1));
  state.channels[&op.getGroupPhaseMutable()].push_back(boolValueToken(true));

  require(!fireActorOperation(op, state),
          "serialize accepted a malformed vector");
  require(state.channels[&op.getVectorMutable()].size() == 1 &&
              state.channels[&op.getMaskMutable()].size() == 1 &&
              state.channels[&op.getGroupPhaseMutable()].size() == 1,
          "serialize consumed input on conversion failure");
  require(state.pendingChannels.empty() &&
              state.pendingObservedOutputs.empty() &&
              state.actorMutationEpoch == 0,
          "serialize published output on conversion failure");
}

void structuredVectorMemoryUsesSharedLaneSemantics(mlir::scf::IfOp op) {
  mlir::Block &entry =
      op->getParentOfType<mlir::func::FuncOp>().getBody().front();
  mlir::Value start = entry.getArgument(1);
  mlir::Value index = entry.getArgument(2);
  mlir::Value value = entry.getArgument(3);
  mlir::Value memory = entry.getArgument(4);
  mlir::Type elementType =
      mlir::cast<mlir::MemRefType>(memory.getType()).getElementType();

  SimulatorState state;
  state.channels[&op->getOpOperand(0)].push_back(boolValueToken(true));
  state.observedOutputs[start].push_back(noneToken());
  state.observedOutputs[index].push_back(integerValueToken(1));
  state.observedOutputs[value].push_back(
      tokenWithBits(value.getType(), 0x44332211U));

  llvm::SmallVector<Token> elements;
  for (uint64_t byte : {9U, 8U, 7U, 6U, 5U, 4U})
    elements.push_back(tokenWithBits(elementType, byte));
  state.memories[memory] = std::make_shared<MemoryValue>(
      MemoryValue{0, elementType, std::move(elements),
                  llvm::SmallBitVector(6, /*t=*/true)});

  require(fireStructuredOperation(op, state),
          "structured vector memory did not fire");
  require(state.channels[&op->getOpOperand(0)].empty(),
          "structured vector memory did not consume its condition");
  expectBits(state.pendingObservedOutputs[op->getResult(0)],
             op->getResult(0).getType(), {0x44332211U},
             "structured vector load returned the wrong lane order");
  llvm::ArrayRef<Token> completions =
      state.pendingObservedOutputs[op->getResult(1)];
  require(completions.size() == 1 &&
              completions.front().kind == TokenKind::None,
          "structured vector memory did not publish one completion");

  expectBits(state.memories[memory]->elements, elementType,
             {9, 17, 34, 51, 68, 4},
             "structured vector store wrote the wrong lanes");
  require(state.operationFireCounts["dataflow.store"] == 1 &&
              state.operationFireCounts["dataflow.load"] == 1 &&
              state.operationFireCounts["scf.if"] == 1,
          "structured vector memory fired an operation more than once");
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  require(static_cast<bool>(module), "unable to parse fixture");

  dataflow::ParallelizeOp parallelize;
  dataflow::PackOp pack;
  dataflow::SerializeOp serialize;
  mlir::scf::IfOp structuredVectorMemory;
  llvm::SmallVector<dataflow::UnpackOp, 2> unpacks;
  module->walk([&](dataflow::ParallelizeOp op) { parallelize = op; });
  module->walk([&](dataflow::PackOp op) { pack = op; });
  module->walk([&](dataflow::SerializeOp op) { serialize = op; });
  module->walk([&](mlir::scf::IfOp op) { structuredVectorMemory = op; });
  module->walk([&](dataflow::UnpackOp op) { unpacks.push_back(op); });
  require(parallelize && pack && serialize && structuredVectorMemory &&
              unpacks.size() == 2,
          "fixture actors are missing");

  dataflow::UnpackOp vectorUnpack = unpacks[0];
  dataflow::UnpackOp maskUnpack = unpacks[1];
  if (mlir::cast<mlir::VectorType>(vectorUnpack.getVector().getType())
          .getElementType()
          .isInteger(1))
    std::swap(vectorUnpack, maskUnpack);

  parallelizePreservesQueuedActivation(parallelize);
  serializePreservesQueuedActivation(serialize, vectorUnpack, maskUnpack);
  parallelizeFailureIsAtomic(parallelize);
  packFailureIsAtomic(pack);
  unpackFailureIsAtomic(vectorUnpack);
  serializeFailureIsAtomic(serialize);
  structuredVectorMemoryUsesSharedLaneSemantics(structuredVectorMemory);
  return 0;
}
