#include "DFGSimulatorInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
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

  func.func @rank_two(%packed: i48, %vector: vector<2x3xi8>) {
    %lanes = dataflow.unpack %packed : i48 -> vector<2x3xi8>
    %bits = dataflow.pack %vector : vector<2x3xi8> -> i48
    return
  }

  func.func @memory(%mem: memref<?xi8>, %addr: index,
                    %addresses: vector<2xindex>, %data: vector<2xi8>,
                    %ctrl: none) {
    %loaded, %read = dataflow.load %mem[%addr] %ctrl : memref<?xi8>
    %written = dataflow.store %mem[%addresses] %data %ctrl
        : memref<?xi8>, vector<2xindex>, vector<2xi8>
    return
  }
}
)mlir";

// Row-major flattened lane order for the fixture's vector<2x3xi8>: element
// [row][column] owns flattened lane row * 3 + column, lane zero owns the least
// significant byte, and the whole value packs to 0x060504030201.
constexpr unsigned kRows = 2;
constexpr unsigned kColumns = 3;
constexpr unsigned kElementWidth = 8;
constexpr unsigned kRankTwoWidth = kRows * kColumns * kElementWidth;
constexpr uint64_t kRankTwoPacked = 0x060504030201ULL;
constexpr uint8_t kLanes[kRows][kColumns] = {{0x01, 0x02, 0x03},
                                             {0x04, 0x05, 0x06}};

unsigned rowMajorLane(unsigned row, unsigned column) {
  return row * kColumns + column;
}

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DFGVectorBoundaryTest: " << message << "\n";
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

// A token keeps its bit pattern in one APInt, whose width is an unsigned. An
// exact vector width past that is reported at this boundary instead of being
// narrowed into a legal one. The exact width is named in the diagnostic and no
// value of that width is ever built, so the check stays arithmetic.
void tokenWidthNarrowsAtTokenBoundary() {
  mlir::MLIRContext context;
  mlir::Type wide =
      mlir::VectorType::get({257}, mlir::IntegerType::get(&context, 16777215));
  llvm::Expected<unsigned> width = tokenTypeBitWidth(wide);
  require(!width, "a width past the token representation was accepted");
  const std::string message = llvm::toString(width.takeError());
  require(message == "bit width 4311744255 exceeds the token representation",
          "the token boundary did not report its own narrowing: " + message);
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

// Independent oracle for the rank-two bit representation: the expected packed
// value is assembled here from the two-dimensional lane table, so a permuted or
// padded lane placement cannot cancel out against its own inverse.
llvm::APInt rowMajorLaneBits() {
  llvm::APInt bits(kRankTwoWidth, 0);
  for (unsigned row = 0; row < kRows; ++row)
    for (unsigned column = 0; column < kColumns; ++column)
      bits.insertBits(llvm::APInt(kElementWidth, kLanes[row][column]),
                      rowMajorLane(row, column) * kElementWidth);
  return bits;
}

void unpackPlacesRowMajorLanes(dataflow::UnpackOp op) {
  SimulatorState state;
  state.channels[&op.getPackedMutable()].push_back(
      tokenWithBits(op.getPacked().getType(), kRankTwoPacked));

  require(fireActorOperation(op, state), "rank-two unpack did not fire");
  auto &published = state.pendingObservedOutputs[op.getVector()];
  require(published.size() == 1,
          "rank-two unpack did not publish exactly one vector token");
  llvm::APInt bits = takeExpected(
      tokenBitPattern(published.front(), op.getVector().getType()));
  require(bits.getBitWidth() == kRankTwoWidth,
          "rank-two unpack published the wrong token bit width");
  for (unsigned row = 0; row < kRows; ++row)
    for (unsigned column = 0; column < kColumns; ++column) {
      llvm::APInt lane = bits.extractBits(
          kElementWidth, rowMajorLane(row, column) * kElementWidth);
      require(lane.getZExtValue() == kLanes[row][column],
              "rank-two unpack placed a lane outside its row-major bit slice");
    }
}

void packFlattensRowMajorLanes(dataflow::PackOp op) {
  SimulatorState state;
  state.channels[&op.getVectorMutable()].push_back(takeExpected(
      tokenFromBitPattern(rowMajorLaneBits(), op.getVector().getType())));

  require(fireActorOperation(op, state), "rank-two pack did not fire");
  auto &published = state.pendingObservedOutputs[op.getPacked()];
  require(published.size() == 1,
          "rank-two pack did not publish exactly one packed token");
  llvm::APInt bits = takeExpected(
      tokenBitPattern(published.front(), op.getPacked().getType()));
  require(bits.getBitWidth() == kRankTwoWidth,
          "rank-two pack published the wrong packed bit width");
  require(bits.getZExtValue() == kRankTwoPacked,
          "rank-two pack did not flatten lanes row-major with lane zero low");
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

// A memory actor rejects an access entirely on peeked inputs. Only its reason
// and the unsupported-capability outcome may change; inputs, outputs, actor
// mutation state, events, fire counts, and memory may not.
std::shared_ptr<MemoryValue>
makeMemory(mlir::Type elementType, std::initializer_list<uint64_t> values) {
  auto memory = std::make_shared<MemoryValue>();
  for (uint64_t value : values)
    memory->elements.push_back(tokenWithBits(elementType, value));
  memory->elementType = elementType;
  memory->initialized = llvm::SmallBitVector(memory->elements.size(), true);
  return memory;
}

unsigned resolvedIndexBits(mlir::Operation *scope) {
  return takeExpected(loom::getIndexBitWidth(scope));
}

Token indexVectorToken(unsigned indexBits,
                       std::initializer_list<uint64_t> lanes) {
  llvm::APInt bits(indexBits * lanes.size(), 0);
  unsigned lane = 0;
  for (uint64_t value : lanes)
    bits.insertBits(llvm::APInt(indexBits, value), indexBits * lane++);
  Token token;
  token.kind = TokenKind::Vector;
  token.bitPattern = bits;
  return token;
}

void expectUntouchedRun(SimulatorState &state, const MemoryValue &memory,
                        llvm::ArrayRef<uint64_t> elements,
                        llvm::StringRef message) {
  require(state.pendingChannels.empty() && state.pendingObservedOutputs.empty(),
          message);
  require(state.actorMutationEpoch == 0 && state.eventCount == 0 &&
              state.operationFireCounts.empty(),
          message);
  require(state.terminalPrimitiveOps.empty(), message);
  require(memory.elements.size() == elements.size(), message);
  for (auto [token, value] : llvm::zip_equal(memory.elements, elements))
    require(bitsOf(token, memory.elementType) == value, message);
}

void loadRejectionIsAtomic(dataflow::LoadOp op) {
  SimulatorState state;
  auto memoryType = mlir::cast<mlir::MemRefType>(op.getMem().getType());
  auto memory = makeMemory(memoryType.getElementType(), {0x11, 0x22});
  state.channels[&op.getMemMutable()].push_back(
      pointerToken(op.getMem(), memory, 0));
  state.channels[&op.getAddrMutable()].push_back(
      indexToken(llvm::APInt(resolvedIndexBits(op.getOperation()), 99)));
  state.channels[&op.getCtrlMutable()].push_back(noneToken());

  require(!fireActorOperation(op, state),
          "load accepted an out-of-range address");
  require(state.diagnostics.size() == 1, "load recorded no rejection reason");
  require(!fireActorOperation(op, state),
          "load accepted an out-of-range address when re-polled");
  require(state.diagnostics.size() == 2,
          "a re-polled load rejection is not detectable as a failed attempt");
  require(state.channels[&op.getMemMutable()].size() == 1 &&
              state.channels[&op.getAddrMutable()].size() == 1 &&
              state.channels[&op.getCtrlMutable()].size() == 1,
          "load consumed input on a rejected access");
  expectUntouchedRun(state, *memory, {0x11, 0x22},
                     "load changed run or memory state on a rejected access");
}

void storeDuplicateScatterIsAtomic(dataflow::StoreOp op) {
  SimulatorState state;
  auto memoryType = mlir::cast<mlir::MemRefType>(op.getMem().getType());
  auto memory = makeMemory(memoryType.getElementType(), {0x11, 0x22});
  state.channels[&op.getMemMutable()].push_back(
      pointerToken(op.getMem(), memory, 0));
  state.channels[&op.getAddrMutable()].push_back(
      indexVectorToken(resolvedIndexBits(op.getOperation()), {1, 1}));
  state.channels[&op.getDataMutable()].push_back(
      tokenWithBits(op.getData().getType(), 0xAB43));
  state.channels[&op.getCtrlMutable()].push_back(noneToken());

  require(!fireActorOperation(op, state),
          "store accepted duplicate active destinations");
  require(state.diagnostics.size() == 1, "store recorded no rejection reason");
  require(!fireActorOperation(op, state),
          "store accepted duplicate active destinations when re-polled");
  require(state.diagnostics.size() == 2,
          "a re-polled store rejection is not detectable as a failed attempt");
  require(state.channels[&op.getMemMutable()].size() == 1 &&
              state.channels[&op.getAddrMutable()].size() == 1 &&
              state.channels[&op.getDataMutable()].size() == 1 &&
              state.channels[&op.getCtrlMutable()].size() == 1,
          "store consumed input on a rejected access");
  require(state.runtimeUnsupportedCapability,
          "duplicate active scatter did not report an unsupported capability");
  expectUntouchedRun(state, *memory, {0x11, 0x22},
                     "store changed run or memory state on a rejected access");
}

} // namespace

int main() {
  tokenWidthNarrowsAtTokenBoundary();

  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  require(static_cast<bool>(module), "unable to parse fixture");

  auto parallelizeFunc =
      module->lookupSymbol<mlir::func::FuncOp>("parallelize");
  auto serializeFunc = module->lookupSymbol<mlir::func::FuncOp>("serialize");
  auto rankTwoFunc = module->lookupSymbol<mlir::func::FuncOp>("rank_two");
  auto memoryFunc = module->lookupSymbol<mlir::func::FuncOp>("memory");
  require(parallelizeFunc && serializeFunc && rankTwoFunc && memoryFunc,
          "fixture functions are missing");

  dataflow::ParallelizeOp parallelize;
  dataflow::PackOp pack;
  dataflow::SerializeOp serialize;
  dataflow::PackOp rankTwoPack;
  dataflow::UnpackOp rankTwoUnpack;
  llvm::SmallVector<dataflow::UnpackOp, 2> unpacks;
  parallelizeFunc.walk([&](dataflow::ParallelizeOp op) { parallelize = op; });
  parallelizeFunc.walk([&](dataflow::PackOp op) { pack = op; });
  serializeFunc.walk([&](dataflow::SerializeOp op) { serialize = op; });
  serializeFunc.walk([&](dataflow::UnpackOp op) { unpacks.push_back(op); });
  rankTwoFunc.walk([&](dataflow::PackOp op) { rankTwoPack = op; });
  rankTwoFunc.walk([&](dataflow::UnpackOp op) { rankTwoUnpack = op; });
  dataflow::LoadOp load;
  dataflow::StoreOp store;
  memoryFunc.walk([&](dataflow::LoadOp op) { load = op; });
  memoryFunc.walk([&](dataflow::StoreOp op) { store = op; });
  require(parallelize && pack && serialize && unpacks.size() == 2 &&
              rankTwoPack && rankTwoUnpack && load && store,
          "fixture actors are missing");

  dataflow::UnpackOp vectorUnpack = unpacks[0];
  dataflow::UnpackOp maskUnpack = unpacks[1];
  if (mlir::cast<mlir::VectorType>(vectorUnpack.getVector().getType())
          .getElementType()
          .isInteger(1))
    std::swap(vectorUnpack, maskUnpack);

  parallelizePreservesQueuedActivation(parallelize);
  serializePreservesQueuedActivation(serialize, vectorUnpack, maskUnpack);
  unpackPlacesRowMajorLanes(rankTwoUnpack);
  packFlattensRowMajorLanes(rankTwoPack);
  parallelizeFailureIsAtomic(parallelize);
  packFailureIsAtomic(pack);
  unpackFailureIsAtomic(vectorUnpack);
  serializeFailureIsAtomic(serialize);
  loadRejectionIsAtomic(load);
  storeDuplicateScatterIsAtomic(store);
  return 0;
}
