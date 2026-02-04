//===-- SCFToHandshake.cpp - SCF to Handshake pass entry --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SCF-to-Handshake dataflow conversion pass. It
// identifies accelerator-marked functions, inlines callees, invokes the
// HandshakeLowering class to convert them to Handshake dialect, and rewrites
// host-side calls to use ESI channel-based accelerator instances.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/SCFToHandshake.h"
#include "loom/Conversion/SCFToHandshakeImpl.h"
#include "loom/Dialect/Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace {

using namespace mlir;

using loom::detail::HandshakeLowering;
using loom::detail::inlineCallsInAccel;
using loom::detail::isAccelFunc;

static void rewriteHostCalls(ModuleOp module,
                             llvm::DenseSet<StringRef> accelNames) {
  llvm::DenseMap<func::FuncOp, Value> clkCache;
  llvm::DenseMap<func::FuncOp, Value> rstCache;
  llvm::DenseMap<func::FuncOp, Value> trueCache;
  llvm::DenseMap<func::FuncOp, Value> falseCache;
  llvm::DenseMap<func::FuncOp, unsigned> instCount;

  auto getBoolConst = [&](func::FuncOp func, bool value) -> Value {
    auto &cache = value ? trueCache : falseCache;
    auto it = cache.find(func);
    if (it != cache.end())
      return it->second;
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.front());
    Value cst = builder.create<arith::ConstantOp>(
        func.getLoc(), builder.getI1Type(), builder.getBoolAttr(value));
    cache[func] = cst;
    return cst;
  };

  auto getClockConst = [&](func::FuncOp func) -> Value {
    auto it = clkCache.find(func);
    if (it != clkCache.end())
      return it->second;
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.front());
    auto attr = circt::seq::ClockConstAttr::get(
        func.getContext(), circt::seq::ClockConst::Low);
    Value clk = builder.create<circt::seq::ConstClockOp>(func.getLoc(), attr)
                    .getResult();
    clkCache[func] = clk;
    return clk;
  };

  auto getResetConst = [&](func::FuncOp func) -> Value {
    auto it = rstCache.find(func);
    if (it != rstCache.end())
      return it->second;
    Value rst = getBoolConst(func, false);
    rstCache[func] = rst;
    return rst;
  };

  module.walk([&](func::FuncOp func) {
    llvm::SmallVector<func::CallOp, 8> calls;
    func.walk([&](func::CallOp call) {
      if (!accelNames.contains(call.getCallee()))
        return;
      calls.push_back(call);
    });

    for (func::CallOp call : calls) {
      OpBuilder builder(call);
      Location loc = call.getLoc();
      Value clk = getClockConst(func);
      Value rst = getResetConst(func);
      Value valid = getBoolConst(func, true);
      Value ready = getBoolConst(func, true);

      llvm::SmallVector<Value, 8> chanOperands;
      chanOperands.reserve(call.getNumOperands() + 1);
      for (Value operand : call.getOperands()) {
        auto wrap = builder.create<circt::esi::WrapValidReadyOp>(
            loc, operand, valid);
        chanOperands.push_back(wrap.getResult(0));
      }
      auto entryWrap =
          builder.create<circt::esi::WrapValidReadyOp>(loc, valid, valid);
      chanOperands.push_back(entryWrap.getResult(0));

      llvm::SmallVector<Type, 4> resultTypes;
      resultTypes.reserve(call.getNumResults() + 1);
      for (Type type : call.getResultTypes())
        resultTypes.push_back(
            circt::esi::ChannelType::get(call.getContext(), type));
      resultTypes.push_back(
          circt::esi::ChannelType::get(call.getContext(), builder.getI1Type()));

      unsigned count = instCount[func]++;
      std::string instName = call.getCallee().str() + "_inst" +
                             std::to_string(count);

      auto esiInst = builder.create<circt::handshake::ESIInstanceOp>(
          loc, resultTypes, call.getCallee(), instName, clk, rst,
          chanOperands);

      for (unsigned i = 0, e = call.getNumResults(); i < e; ++i) {
        auto unwrap = builder.create<circt::esi::UnwrapValidReadyOp>(
            loc, esiInst.getResult(i), ready);
        call.getResult(i).replaceAllUsesWith(unwrap.getResult(0));
      }
      if (call.getNumResults() < esiInst.getNumResults()) {
        auto doneUnwrap = builder.create<circt::esi::UnwrapValidReadyOp>(
            loc, esiInst.getResult(call.getNumResults()), ready);
        (void)doneUnwrap;
      }

      call.erase();
    }
  });
}

struct SCFToHandshakeDataflowPass
    : public PassWrapper<SCFToHandshakeDataflowPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<loom::dataflow::DataflowDialect,
                    circt::handshake::HandshakeDialect, circt::esi::ESIDialect,
                    circt::seq::SeqDialect, func::FuncDialect,
                    scf::SCFDialect, memref::MemRefDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::SmallVector<func::FuncOp, 8> accelFuncs;
    llvm::DenseSet<StringRef> accelNames;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!isAccelFunc(func))
        continue;
      accelFuncs.push_back(func);
      accelNames.insert(func.getName());
    }

    for (func::FuncOp func : accelFuncs) {
      SymbolTable symbols(module);
      if (failed(inlineCallsInAccel(func, symbols))) {
        signalPassFailure();
        return;
      }
    }

    for (func::FuncOp func : accelFuncs) {
      AliasAnalysis aliasAnalysis(func);
      HandshakeLowering lowering(func, aliasAnalysis);
      if (failed(lowering.run())) {
        signalPassFailure();
        return;
      }
    }

    rewriteHostCalls(module, accelNames);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::createSCFToHandshakeDataflowPass() {
  return std::make_unique<SCFToHandshakeDataflowPass>();
}
