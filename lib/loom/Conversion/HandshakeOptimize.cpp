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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
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

// --- Pass wrapper ---

namespace {

struct HandshakeCleanupPass
    : public mlir::PassWrapper<HandshakeCleanupPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HandshakeCleanupPass)

  llvm::StringRef getArgument() const override {
    return "loom-handshake-cleanup";
  }
  llvm::StringRef getDescription() const override {
    return "Insert handshake.sink for dead values, eliminate dead ops "
           "(fixed-point, runs after canonicalize/CSE)";
  }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::OpBuilder builder(module.getContext());

    module.walk([&](circt::handshake::FuncOp func) {
      // Fixed-point: repeat sink insertion + DCE until stable.
      // Bounded to prevent infinite loops.
      for (unsigned iter = 0; iter < 20; ++iter) {
        bool changed = false;

        // Insert sinks for unused values.
        llvm::SmallVector<mlir::Value, 16> toSink;
        func.walk([&](mlir::Operation *op) {
          if (mlir::isa<circt::handshake::FuncOp,
                        circt::handshake::ReturnOp,
                        circt::handshake::SinkOp>(op))
            return;
          for (mlir::Value result : op->getResults()) {
            if (result.use_empty())
              toSink.push_back(result);
          }
        });
        for (mlir::Value result : toSink) {
          if (auto *def = result.getDefiningOp()) {
            builder.setInsertionPointAfter(def);
            circt::handshake::SinkOp::create(builder, def->getLoc(), result);
            changed = true;
          }
        }

        // Also check block arguments (func args that are unused).
        for (auto arg : func.getBody().getArguments()) {
          if (arg.use_empty()) {
            builder.setInsertionPointToStart(&func.getBody().front());
            circt::handshake::SinkOp::create(builder, func.getLoc(), arg);
            changed = true;
          }
        }

        if (!changed)
          break;

        // Run DCE: eliminate ops whose results all feed only sinks.
        if (failed(loom::runHandshakeCleanup(func, builder)))
          return signalPassFailure();
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::createHandshakeCleanupPass() {
  return std::make_unique<HandshakeCleanupPass>();
}
