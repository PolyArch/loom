//===-- SCFToHandshakeImpl.h - SCF to Handshake internal API ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This header declares internal implementation details for the SCF-to-Handshake
// conversion, including the HandshakeConversion class that manages value mapping,
// memory access tracking, and control flow conversion from structured control
// flow to dataflow operations.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_SCFTOHANDSHAKEIMPL_H
#define LOOM_CONVERSION_SCFTOHANDSHAKEIMPL_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace loom {
namespace detail {

struct PathEntry {
  mlir::Operation *op = nullptr;
  unsigned region = 0;
};

using ScfPath = llvm::SmallVector<PathEntry, 4>;

enum class AccessKind { Load, Store };

struct MemAccess {
  mlir::Operation *origOp = nullptr;
  mlir::Value origMemref;
  mlir::Value memref;
  AccessKind kind = AccessKind::Load;
  unsigned order = 0;
  ScfPath path;
  circt::handshake::LoadOp loadOp;
  circt::handshake::StoreOp storeOp;
  mlir::Value doneToken;
  mlir::Value controlToken;
};

class HandshakeConversion {
public:
  HandshakeConversion(mlir::func::FuncOp func, mlir::AliasAnalysis &aa);
  mlir::LogicalResult run();

private:
  struct RegionState;

  mlir::LogicalResult convertOp(mlir::Operation *op, RegionState &state);
  mlir::LogicalResult convertFor(mlir::scf::ForOp op, RegionState &state);
  mlir::LogicalResult convertWhile(mlir::scf::WhileOp op, RegionState &state);
  mlir::LogicalResult convertIf(mlir::scf::IfOp op, RegionState &state);
  mlir::LogicalResult convertIndexSwitch(mlir::scf::IndexSwitchOp op,
                                         RegionState &state);
  mlir::LogicalResult convertReturn(mlir::func::ReturnOp op, RegionState &state);
  mlir::LogicalResult convertLoad(mlir::memref::LoadOp op, RegionState &state);
  mlir::LogicalResult convertStore(mlir::memref::StoreOp op, RegionState &state);
  mlir::Value mapValue(mlir::Value value, RegionState &state,
                       mlir::Location loc);
  mlir::Value getEntryToken(mlir::Location loc);
  mlir::Value makeConstant(mlir::Location loc, mlir::Attribute value,
                           mlir::Type type, mlir::Value ctrlToken);
  mlir::Value makeBool(mlir::Location loc, bool value);
  mlir::Value makeDummyData(mlir::Location loc, mlir::Type type);
  void updateInvariantCond(RegionState &state, mlir::Value cond);
  void finalizeMemory();
  mlir::LogicalResult buildMemoryControl();
  mlir::LogicalResult verifyMemoryControl();
  void assignHandshakeNames();

  mlir::func::FuncOp func;
  circt::handshake::FuncOp handshakeFunc;
  mlir::AliasAnalysis &aliasAnalysis;
  mlir::OpBuilder builder;
  llvm::SmallVector<MemAccess, 16> memAccesses;
  mlir::Value entrySignal;
  mlir::Value entryToken;
  mlir::Value memoryDoneToken;
  mlir::Value doneSignal;
  llvm::SmallVector<mlir::Value, 4> pendingReturnValues;
  mlir::Location returnLoc;
  bool sawReturn = false;
  unsigned orderCounter = 0;
  int memoryId = 0;
  llvm::DenseMap<mlir::Operation *, mlir::Value> forConds;
  llvm::DenseMap<mlir::Operation *, mlir::Value> whileConds;
  llvm::DenseMap<mlir::Operation *, mlir::Value> ifConds;
  llvm::DenseMap<mlir::Operation *, mlir::Value> switchConds;
};

// --- Analysis helpers (used by both Analysis and Convert files) ---

void copyLoomAnnotations(mlir::Operation *src, mlir::Operation *dst);
std::optional<std::string> resolveSourcePath(mlir::Location loc);
std::string demangleBaseName(llvm::StringRef name);
bool readFile(llvm::StringRef path, std::string &out);
bool extractFunctionSource(llvm::StringRef content, llvm::StringRef funcName,
                           std::string &params, std::string &body);
llvm::SmallVector<std::string, 8> extractParamNames(llvm::StringRef params);
std::optional<std::string> extractReturnName(llvm::StringRef body);
mlir::Value castToIndex(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value);
mlir::Value castIndexToType(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value, mlir::Type targetType);

struct StreamStepInfo {
  int64_t constant = 0;
  mlir::Value value;
  bool isConst = false;
  llvm::StringRef stepOp;
};

struct StreamWhileAttr {
  int64_t ivIndex = -1;
  llvm::StringRef stepOp;
  llvm::StringRef stopCond;
  bool cmpOnUpdate = false;
};

struct StreamWhileOperands {
  mlir::Value init;
  mlir::Value step;
  mlir::Value bound;
  int64_t stepConst = 0;
  bool stepIsConst = false;
  bool bodyInBefore = false;
};

std::optional<StreamWhileAttr> getStreamWhileAttr(mlir::scf::WhileOp op);
mlir::LogicalResult analyzeStreamableWhile(mlir::scf::WhileOp op,
                                           const StreamWhileAttr &attr,
                                           StreamWhileOperands &result);
bool isLocalToRegion(mlir::Value value, mlir::Region *region);
ScfPath computeScfPath(mlir::Operation *op);

// --- Other utilities ---

mlir::LogicalResult inlineCallsInAccel(mlir::func::FuncOp func,
                                       mlir::SymbolTable &symbols);
bool isAccelFunc(mlir::func::FuncOp func);
mlir::Value getMemrefRoot(mlir::Value memref);

enum class MemTargetHint { None, Rom, Extmemory };

MemTargetHint getMemTargetHint(mlir::Operation *op);
std::optional<int64_t> getStaticMemrefByteSize(mlir::MemRefType type);

} // namespace detail
} // namespace loom

#endif // LOOM_CONVERSION_SCFTOHANDSHAKEIMPL_H
