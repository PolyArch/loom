//===- SCFToHandshakeImpl.h - SCF->Handshake internal API -------*- C++ -*-===//
//
// Part of the Loom project.
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

class HandshakeLowering {
public:
  HandshakeLowering(mlir::func::FuncOp func, mlir::AliasAnalysis &aa);
  mlir::LogicalResult run();

private:
  struct RegionState;

  mlir::LogicalResult lowerOp(mlir::Operation *op, RegionState &state);
  mlir::LogicalResult lowerFor(mlir::scf::ForOp op, RegionState &state);
  mlir::LogicalResult lowerWhile(mlir::scf::WhileOp op, RegionState &state);
  mlir::LogicalResult lowerIf(mlir::scf::IfOp op, RegionState &state);
  mlir::LogicalResult lowerIndexSwitch(mlir::scf::IndexSwitchOp op,
                                       RegionState &state);
  mlir::LogicalResult lowerReturn(mlir::func::ReturnOp op, RegionState &state);
  mlir::LogicalResult lowerLoad(mlir::memref::LoadOp op, RegionState &state);
  mlir::LogicalResult lowerStore(mlir::memref::StoreOp op, RegionState &state);
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
  void insertForks();
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

mlir::LogicalResult inlineCallsInAccel(mlir::func::FuncOp func,
                                       mlir::SymbolTable &symbols);
bool isAccelFunc(mlir::func::FuncOp func);
mlir::Value getMemrefRoot(mlir::Value memref);

} // namespace detail
} // namespace loom

#endif // LOOM_CONVERSION_SCFTOHANDSHAKEIMPL_H
