//===-- LLVMToSCFConvert.h - LLVM to SCF conversion helpers -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Internal entry points for LLVM-to-SCF lowering. These functions implement
// global and function conversion and are used by the pass driver.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_LLVMTOSCF_CONVERT_H
#define LOOM_CONVERSION_LLVMTOSCF_CONVERT_H

#include "loom/Conversion/LLVMToSCFSupport.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace loom::llvm_to_scf {

struct ConvertedGlobal {
  mlir::MemRefType type;
  mlir::LLVM::GlobalOp oldGlobal;
};

mlir::LogicalResult
convertGlobals(mlir::ModuleOp module, mlir::OpBuilder &builder,
               llvm::StringMap<ConvertedGlobal> &out);

mlir::LogicalResult
convertFunction(mlir::ModuleOp module, mlir::LLVM::LLVMFuncOp func,
                mlir::OpBuilder &builder,
                llvm::StringMap<ConvertedGlobal> &globals,
                const llvm::StringSet<> &varargFunctions);

} // namespace loom::llvm_to_scf

#endif // LOOM_CONVERSION_LLVMTOSCF_CONVERT_H
