#ifndef LOOM_LIB_SIMULATOR_DFGSIMULATORINTERNAL_H
#define LOOM_LIB_SIMULATOR_DFGSIMULATORINTERNAL_H

#include "Simulator/DFGSimulator.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

inline constexpr std::uint64_t kLoadAddressScore = 1;
inline constexpr std::uint64_t kStoreAddressScore = 2;

struct MemoryValue;

struct MemoryView {
  std::shared_ptr<MemoryValue> memory;
  mlir::Value root;
  std::int64_t byteOffset = 0;
};

enum class TokenKind { None, Integer, Float, Bool, Vector, Pointer };

struct Token {
  TokenKind kind = TokenKind::None;
  // Index storage and the schema 2.2 projection for integers up to 64 bits.
  std::int64_t intValue = 0;
  double floatValue = 0.0;
  bool boolValue = false;
  std::optional<llvm::APInt> bitPattern;
  MemoryView pointer;
};

struct DataflowMemoryRead {
  Token data;
  bool accessedMemory = false;
};

using ChannelMap = llvm::DenseMap<const mlir::OpOperand *, std::deque<Token>>;
using OutputMap = llvm::DenseMap<mlir::Value, llvm::SmallVector<Token>>;

struct LoopState {
  PhaseSemanticState semanticState = PhaseSemanticState::Initial;
  std::optional<Token> latched;
};

struct ParallelizeState {
  ParallelizeSemanticState semanticState;
  llvm::SmallVector<std::optional<Token>, 8> slots;
};

struct MemoryValue {
  std::uint64_t logicalRootId = 0;
  mlir::Type elementType;
  llvm::SmallVector<Token> elements;
  llvm::SmallBitVector initialized;
};

struct MemoryFixture {
  std::string values;
  std::int64_t byteOffset = 0;
};

struct SimulatorState {
  ChannelMap channels;
  ChannelMap pendingChannels;
  OutputMap observedOutputs;
  OutputMap pendingObservedOutputs;
  llvm::DenseMap<mlir::Value, std::shared_ptr<MemoryValue>> memories;
  llvm::DenseMap<mlir::Value, std::uint64_t> memoryRootIds;
  llvm::DenseMap<mlir::Value, MemoryFixture> rawMemoryFixtures;
  llvm::DenseMap<mlir::Operation *, StreamSemanticState> streamStates;
  llvm::DenseSet<mlir::Operation *> failedStreamOps;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> streamTrueEmissionCounts;
  llvm::DenseMap<mlir::Operation *, LoopState> carryStates;
  llvm::DenseMap<mlir::Operation *, LoopState> invariantStates;
  llvm::DenseMap<mlir::Operation *, ParallelizeState> parallelizeStates;
  llvm::DenseSet<mlir::Operation *> gateContinueStates;
  llvm::DenseSet<mlir::Operation *> oneShotOps;
  llvm::DenseSet<mlir::Operation *> terminalPrimitiveOps;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> structuredEffectFireCounts;
  llvm::DenseMap<mlir::Value, std::uint64_t> seededTokenCounts;
  llvm::SmallVector<std::string> diagnostics;
  std::map<std::string, std::uint64_t> operationFireCounts;
  std::map<std::string, std::uint64_t> modeledLibraryCalls;
  std::uint64_t nextMemoryRootId = 0;
  std::uint64_t modeledLibraryScore = 0;
  std::uint64_t eventCount = 0;
  std::uint64_t memoryAddressScore = 0;
  std::uint64_t structuredLoopIterations = 0;
  std::uint64_t maxStructuredLoopIterations = 0;
  std::uint64_t actorMutationEpoch = 0;
};

struct UnsupportedOperation {
  std::string label;
  std::string reason;
};

Token noneToken();
Token integerValueToken(std::int64_t value);
Token floatValueToken(double value);
Token boolValueToken(bool value);
llvm::Expected<unsigned> tokenTypeBitWidth(mlir::Type type);
llvm::Expected<llvm::APInt> tokenBitPattern(const Token &token,
                                            mlir::Type type);
llvm::Expected<Token> tokenFromBitPattern(const llvm::APInt &bits,
                                          mlir::Type type);
llvm::Expected<Token> parseRuntimeToken(llvm::StringRef raw, mlir::Type type);
std::string tokenToString(const Token &token, mlir::Type type);
Token pointerToken(mlir::Value root, std::shared_ptr<MemoryValue> memory = {},
                   std::int64_t byteOffset = 0);
llvm::Expected<Token> tokenFromTypedAttr(mlir::TypedAttr attr);
llvm::Expected<Token> zeroToken(mlir::Type type);
llvm::Expected<Token> ensurePointerMemory(SimulatorState &state, Token token,
                                          mlir::Type elementType);
llvm::Expected<std::int64_t> gepByteOffset(mlir::LLVM::GEPOp op,
                                           llvm::ArrayRef<Token> dynamicTokens);

bool hasToken(ChannelMap &channels, mlir::OpOperand &operand);
Token popToken(SimulatorState &state, mlir::OpOperand &operand);
Token peekToken(ChannelMap &channels, mlir::OpOperand &operand);
void emitToken(SimulatorState &state, mlir::Value value, Token token);
bool recordEvent(SimulatorState &state, llvm::StringRef opName);
bool hasComputedAddress(mlir::Value value);
std::int64_t integerToken(const Token &token);
bool boolToken(const Token &token);
llvm::Expected<std::int64_t> byteSizeOfType(mlir::Type type,
                                            mlir::Operation *scope);

std::optional<std::size_t>
resolveElementIndex(const MemoryView &view, const Token &addr,
                    SimulatorState &state, mlir::Operation *scope,
                    llvm::StringRef opName, std::uint64_t laneOffset = 0);
std::optional<Token> readMemoryElement(const MemoryView &view,
                                       std::size_t index, SimulatorState &state,
                                       llvm::StringRef opName);
void writeMemoryElement(const MemoryView &view, std::size_t index, Token value);
std::optional<DataflowMemoryRead>
readDataflowMemory(const MemoryView &view, const Token &addr,
                   mlir::MemRefType memoryType, mlir::Type dataType,
                   const Token *mask, mlir::Type maskType,
                   SimulatorState &state, mlir::Operation *scope);
std::optional<bool>
writeDataflowMemory(const MemoryView &view, const Token &addr,
                    const Token &data, mlir::MemRefType memoryType,
                    mlir::Type dataType, const Token *mask, mlir::Type maskType,
                    SimulatorState &state, mlir::Operation *scope);
bool isSupportedLLVMCall(mlir::LLVM::CallOp op);
bool executeCmsisNNVecMatMultTS8(mlir::LLVM::CallOp op, SimulatorState &state,
                                 llvm::ArrayRef<Token> operands, Token &result);
bool isSupportedPointerICmp(mlir::LLVM::ICmpOp op);
llvm::Expected<Token> evaluatePointerICmp(mlir::LLVM::ICmpOp op,
                                          const Token &lhs, const Token &rhs);
llvm::Expected<PrimitiveValue> primitiveValueFromToken(const Token &token,
                                                       mlir::Type type);
llvm::Expected<Token> tokenFromPrimitiveValue(const PrimitiveValue &value,
                                              mlir::Type type);
std::string primitivePredicate(mlir::Operation *op);
std::string primitiveOperationName(mlir::Operation *op);
llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(mlir::Operation *op, llvm::StringRef predicate,
                    mlir::Value result);
llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(mlir::Operation *op, llvm::StringRef predicate,
                    mlir::Type resultType, mlir::Type operandType);
llvm::Error validatePrimitiveTokenTypes(mlir::Operation *op,
                                        mlir::Value result);
llvm::Expected<Token> evaluatePrimitiveToken(mlir::Operation *op,
                                             mlir::Value result,
                                             llvm::ArrayRef<Token> inputTokens);

bool executeLLVMMemcpy(mlir::LLVM::MemcpyOp op, SimulatorState &state,
                       const Token &dst, const Token &src, const Token &len);
bool isPointerSelect(mlir::LLVM::SelectOp op);
std::optional<Token> evaluatePointerSelect(mlir::LLVM::SelectOp op,
                                           const Token &condition,
                                           const Token &trueValue,
                                           const Token &falseValue,
                                           SimulatorState &state);
bool fireActorOperation(mlir::Operation *op, SimulatorState &state);
std::optional<UnsupportedOperation>
unsupportedActorOperation(mlir::Operation *op);

bool isStructuredOperation(mlir::Operation *op);
bool hasPendingOrderedStructuredFire(mlir::Operation *op,
                                     const SimulatorState &state);
bool fireStructuredOperation(mlir::Operation *op, SimulatorState &state);
std::optional<UnsupportedOperation>
unsupportedStructuredOperation(mlir::Operation *op);

std::string unsupportedOperationLabel(mlir::Operation *op);

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

#endif // LOOM_LIB_SIMULATOR_DFGSIMULATORINTERNAL_H
