#ifndef LOOM_TOOLS_LOOM_PIPELINE_H
#define LOOM_TOOLS_LOOM_PIPELINE_H

#include "loom_args.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace loom {

// Compile C sources and import to MLIR LLVM dialect in one step.
// Writes intermediate .ll file. Returns the MLIR module or null on failure.
mlir::OwningOpRef<mlir::ModuleOp>
compileAndImport(const LoomArgs &args, mlir::MLIRContext &ctx,
                 const std::string &llOutputPath);

// Run the LLVM-to-CF conversion pass pipeline.
mlir::LogicalResult runLLVMToCF(mlir::ModuleOp module);

// Run CF-to-SCF conversion pipeline.
mlir::LogicalResult runCFToSCF(mlir::ModuleOp module);

// Run SCF-to-DFG conversion pipeline.
mlir::LogicalResult runSCFToDFG(mlir::ModuleOp module);

// Run host code generation pass.
// Generates a host C source together with loom_accel.h and loom_accel.c in the
// output directory.
mlir::LogicalResult runHostCodeGen(mlir::ModuleOp module,
                                   const std::string &outputPath,
                                   const std::string &originalSource);

// Write MLIR module to file.
mlir::LogicalResult writeMLIR(mlir::ModuleOp module, const std::string &path);

} // namespace loom

#endif
