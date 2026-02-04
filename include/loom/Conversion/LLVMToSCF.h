//===-- LLVMToSCF.h - LLVM to SCF conversion pass ---------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This header declares the LLVM-to-SCF conversion pass, which lowers LLVM
// dialect operations to SCF, arith, memref, and func dialects. This pass is
// a key component of the Loom MLIR pipeline for processing C/C++ input.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_LLVMTOSCF_H
#define LOOM_CONVERSION_LLVMTOSCF_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace loom {

std::unique_ptr<mlir::Pass> createLowerLLVMToSCFPass();

} // namespace loom

#endif // LOOM_CONVERSION_LLVMTOSCF_H
