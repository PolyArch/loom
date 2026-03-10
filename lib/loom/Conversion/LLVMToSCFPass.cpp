//===-- LLVMToSCFPass.cpp - LLVM to SCF conversion pass ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file wires the LLVM-to-SCF conversion helpers into an MLIR pass and
// drives the conversion for all eligible LLVM functions and globals.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/LLVMToSCF.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace loom::llvm_to_scf {

class ConvertLLVMToSCFPass
    : public PassWrapper<ConvertLLVMToSCFPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "loom-convert-llvm-to-scf"; }
  StringRef getDescription() const final {
    return "Convert LLVM dialect to scf-stage dialects";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    OpBuilder builder(context);

    llvm::StringSet<> varargFunctions;
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      if (func.getFunctionType().isVarArg())
        varargFunctions.insert(func.getName());
    }

    llvm::StringMap<ConvertedGlobal> convertedGlobals;
    if (failed(convertGlobals(module, builder, convertedGlobals))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<LLVM::LLVMFuncOp, 8> llvmFunctions;
    for (auto func : module.getOps<LLVM::LLVMFuncOp>())
      llvmFunctions.push_back(func);

    for (LLVM::LLVMFuncOp func : llvmFunctions) {
      if (varargFunctions.contains(func.getName()))
        continue;
      if (IsRawPointerCallee(func.getName()))
        continue;

      // Snapshot existing func.func ops so we can clean up on failure.
      llvm::DenseSet<Operation *> existingFuncs;
      for (auto f : module.getOps<mlir::func::FuncOp>())
        existingFuncs.insert(f.getOperation());

      LogicalResult result = convertFunction(module, func, builder,
                                             convertedGlobals,
                                             varargFunctions);
      if (failed(result)) {
        llvm::errs() << "warning: conversion of function '" << func.getName()
                      << "' failed, skipping\n";
        // Erase any partially-created func.func ops from this conversion.
        llvm::SmallVector<mlir::func::FuncOp, 2> toErase;
        for (auto f : module.getOps<mlir::func::FuncOp>()) {
          if (!existingFuncs.contains(f.getOperation()))
            toErase.push_back(f);
        }
        for (auto f : toErase)
          f.erase();
        continue;
      }
    }

    llvm::SmallVector<LLVM::GlobalOp, 8> globalsToErase;
    for (auto &entry : convertedGlobals)
      globalsToErase.push_back(entry.second.oldGlobal);
    for (LLVM::GlobalOp global : globalsToErase)
      global.erase();
  }
};

} // namespace loom::llvm_to_scf

namespace loom {

std::unique_ptr<mlir::Pass> createConvertLLVMToSCFPass() {
  return std::make_unique<llvm_to_scf::ConvertLLVMToSCFPass>();
}

} // namespace loom
