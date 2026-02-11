//===-- HandshakeOptimize.cpp - Handshake cleanup helpers -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements cleanup utilities for Handshake IR. It provides sink
// insertion for unused values and dead code elimination for side-effect-free
// operations feeding only sinks.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/HandshakeOptimize.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace loom {
namespace {

static bool hasNoSideEffects(Operation *op) {
  if (!op)
    return false;
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op))
    return iface.hasNoEffect();
  if (isPure(op))
    return true;
  if (auto *dialect = op->getDialect()) {
    if (dialect->getNamespace() == "dataflow")
      return true;
  }
  return false;
}

static LogicalResult insertHandshakeSinks(circt::handshake::FuncOp func,
                                          OpBuilder &builder) {
  llvm::SmallVector<Value, 16> toSink;
  func.walk([&](Operation *op) {
    if (isa<circt::handshake::FuncOp, circt::handshake::ReturnOp,
            circt::handshake::SinkOp>(op))
      return;
    if (op->getNumResults() == 0)
      return;
    for (Value result : op->getResults()) {
      if (result.use_empty())
        toSink.push_back(result);
    }
  });

  for (Value result : toSink) {
    Operation *def = result.getDefiningOp();
    if (!def)
      continue;
    builder.setInsertionPointAfter(def);
    circt::handshake::SinkOp::create(builder, def->getLoc(), result);
  }
  return success();
}

static LogicalResult eliminateHandshakeDeadCode(circt::handshake::FuncOp func,
                                                OpBuilder &builder) {
  bool madeProgress = true;
  while (madeProgress) {
    madeProgress = false;
    llvm::SmallVector<Operation *, 16> deadOps;
    llvm::SmallPtrSet<Operation *, 16> sinksToErase;

    func.walk([&](Operation *op) {
      if (isa<circt::handshake::FuncOp, circt::handshake::ReturnOp>(op))
        return;
      if (op->getNumResults() == 0)
        return;
      if (!hasNoSideEffects(op))
        return;

      bool allDead = true;
      for (Value result : op->getResults()) {
        for (OpOperand &use : result.getUses()) {
          Operation *user = use.getOwner();
          if (!isa<circt::handshake::SinkOp>(user)) {
            allDead = false;
            break;
          }
          sinksToErase.insert(user);
        }
        if (!allDead)
          break;
      }
      if (allDead)
        deadOps.push_back(op);
    });

    for (Operation *sink : sinksToErase)
      sink->erase();
    for (Operation *op : deadOps)
      op->erase();
    if (!deadOps.empty())
      madeProgress = true;
  }
  return success();
}

} // namespace

LogicalResult runHandshakeCleanup(circt::handshake::FuncOp func,
                                  OpBuilder &builder) {
  if (failed(insertHandshakeSinks(func, builder)))
    return failure();
  if (failed(eliminateHandshakeDeadCode(func, builder)))
    return failure();
  return success();
}

} // namespace loom
