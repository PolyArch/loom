//===-- LLVMToSCFInternal.h - Internal converter class ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Internal header for the LLVM-to-SCF function converter class.
// This is an implementation detail, not part of the public API.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_LLVMTOSCFINTERNAL_H
#define LOOM_CONVERSION_LLVMTOSCFINTERNAL_H

#include "loom/Conversion/LLVMToSCF.h"

#include "llvm/ADT/DenseSet.h"

namespace loom::llvm_to_scf {

/// Encapsulates shared conversion state for a single LLVM function.
class FunctionConverter {
public:
  FunctionConverter(mlir::ModuleOp module, mlir::LLVM::LLVMFuncOp func,
                    mlir::OpBuilder &builder,
                    llvm::StringMap<ConvertedGlobal> &globals,
                    const llvm::StringSet<> &varargFunctions)
      : module(module), func(func), builder(builder), globals(globals),
        varargFunctions(varargFunctions) {}

  /// Main entry: setup, dispatch loop, epilogue.
  mlir::LogicalResult convert();

  /// Handler methods return FailureOr<bool>:
  ///   true  = op was handled
  ///   false = not this handler's op (try next)
  ///   failure() = conversion error
  mlir::FailureOr<bool> handleConstantOps(mlir::Operation &op,
                                           mlir::Location loc);
  mlir::FailureOr<bool> handleVectorOps(mlir::Operation &op,
                                         mlir::Location loc);
  mlir::FailureOr<bool> handleMemoryOps(mlir::Operation &op,
                                         mlir::Location loc);
  mlir::FailureOr<bool> handleLoadStore(mlir::Operation &op,
                                         mlir::Location loc);
  mlir::FailureOr<bool> handleTerminatorOps(mlir::Operation &op,
                                             mlir::Location loc);
  mlir::FailureOr<bool> handleCallOp(mlir::Operation &op,
                                      mlir::Location loc);
  mlir::FailureOr<bool> handleIntrinsicCall(mlir::Operation &op,
                                             mlir::Location loc);
  mlir::FailureOr<bool> handleArithOps(mlir::Operation &op,
                                        mlir::Location loc);
  mlir::FailureOr<bool> handleSelectOp(mlir::Operation &op,
                                        mlir::Location loc);
  mlir::FailureOr<bool> handleCastOps(mlir::Operation &op,
                                       mlir::Location loc);

  // --- Shared conversion state ---
  mlir::ModuleOp module;
  mlir::LLVM::LLVMFuncOp func;
  mlir::func::FuncOp newFunc;
  mlir::OpBuilder &builder;
  llvm::StringMap<ConvertedGlobal> &globals;
  const llvm::StringSet<> &varargFunctions;

  llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
  VectorMapT vectorMap;
  llvm::DenseMap<mlir::Value, PointerInfo> pointerMap;
  llvm::DenseMap<mlir::Value, PointerInfo> pointerSlotValues;
  llvm::DenseSet<mlir::Value> pointerSlots;
  llvm::DenseMap<mlir::Value, mlir::ArrayAttr> pointerSlotAnnotations;
  llvm::DenseMap<unsigned, llvm::SmallVector<mlir::StringAttr, 4>>
      argAnnotations;
  llvm::DenseMap<mlir::Block *, mlir::Block *> blockMap;

  /// LLVM load results promoted to function arguments (Part 5).
  llvm::DenseSet<mlir::Value> promotedPtrLoads;
};

/// Validate that a PointerInfo has a usable base memref and index.
inline bool isValidPointerInfo(const PointerInfo &info) {
  if (!info.base)
    return false;
  if (!llvm::isa<mlir::MemRefType>(info.base.getType()))
    return false;
  if (info.index && !llvm::isa<mlir::IndexType>(info.index.getType()))
    return false;
  return true;
}

} // namespace loom::llvm_to_scf

#endif // LOOM_CONVERSION_LLVMTOSCFINTERNAL_H
