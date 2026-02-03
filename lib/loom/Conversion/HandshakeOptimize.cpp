//===- HandshakeOptimize.cpp - Handshake cleanup helpers -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/HandshakeOptimize.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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
    builder.create<circt::handshake::SinkOp>(def->getLoc(), result);
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
      if (isa<memref::GetGlobalOp>(op))
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

static LogicalResult optimizeHandshakeForks(circt::handshake::FuncOp func,
                                            OpBuilder &builder) {
  bool madeProgress = true;
  while (madeProgress) {
    madeProgress = false;
    llvm::SmallVector<circt::handshake::ForkOp, 16> forks;
    func.walk([&](circt::handshake::ForkOp forkOp) { forks.push_back(forkOp); });

    for (circt::handshake::ForkOp forkOp : forks) {
      llvm::SmallVector<Value, 8> liveOutputs;
      llvm::SmallVector<unsigned, 8> sinkOnlyOutputs;

      for (unsigned i = 0, e = forkOp.getNumResults(); i < e; ++i) {
        Value result = forkOp.getResults()[i];
        bool hasUses = !result.use_empty();
        bool onlySinks = hasUses;
        for (Operation *user : result.getUsers()) {
          if (!isa<circt::handshake::SinkOp>(user)) {
            onlySinks = false;
            break;
          }
        }
        if (hasUses && onlySinks)
          sinkOnlyOutputs.push_back(i);
        else if (hasUses)
          liveOutputs.push_back(result);
      }

      if (liveOutputs.empty())
        continue;

      if (!sinkOnlyOutputs.empty() ||
          liveOutputs.size() < forkOp.getNumResults()) {
        for (unsigned idx : sinkOnlyOutputs) {
          Value result = forkOp.getResults()[idx];
          for (auto user :
               llvm::make_early_inc_range(result.getUsers())) {
            if (isa<circt::handshake::SinkOp>(user))
              user->erase();
          }
        }

        Value input = forkOp.getOperand();
        if (liveOutputs.size() == 1) {
          liveOutputs[0].replaceAllUsesWith(input);
        } else {
          builder.setInsertionPoint(forkOp);
          auto newFork = builder.create<circt::handshake::ForkOp>(
              forkOp.getLoc(), input,
              static_cast<unsigned>(liveOutputs.size()));
          for (unsigned i = 0, e = liveOutputs.size(); i < e; ++i) {
            liveOutputs[i].replaceAllUsesWith(newFork.getResults()[i]);
          }
        }

        forkOp->erase();
        madeProgress = true;
        continue;
      }

      if (forkOp.getNumResults() == 1) {
        Value input = forkOp.getOperand();
        forkOp.getResults()[0].replaceAllUsesWith(input);
        forkOp->erase();
        madeProgress = true;
      }
    }
  }
  return success();
}

} // namespace

LogicalResult runHandshakeCleanup(circt::handshake::FuncOp func,
                                  OpBuilder &builder) {
  if (failed(insertHandshakeSinks(func, builder)))
    return failure();
  if (failed(optimizeHandshakeForks(func, builder)))
    return failure();
  if (failed(eliminateHandshakeDeadCode(func, builder)))
    return failure();
  return success();
}

} // namespace loom
