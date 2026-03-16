#ifndef FCC_TOOLS_FCC_PIPELINE_H
#define FCC_TOOLS_FCC_PIPELINE_H

#include "fcc_args.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace fcc {

// Compile C sources and import to MLIR LLVM dialect in one step.
// Writes intermediate .ll file. Returns the MLIR module or null on failure.
mlir::OwningOpRef<mlir::ModuleOp>
compileAndImport(const FccArgs &args, mlir::MLIRContext &ctx,
                 const std::string &llOutputPath);

// Run the LLVM-to-CF conversion pass pipeline.
mlir::LogicalResult runLLVMToCF(mlir::ModuleOp module);

// Run CF-to-SCF conversion pipeline.
mlir::LogicalResult runCFToSCF(mlir::ModuleOp module);

// Run SCF-to-DFG conversion pipeline.
mlir::LogicalResult runSCFToDFG(mlir::ModuleOp module);

// Run host code generation pass.
// Generates a host C source and fcc_accel.h in the output directory.
mlir::LogicalResult runHostCodeGen(mlir::ModuleOp module,
                                   const std::string &outputPath,
                                   const std::string &originalSource);

// Write MLIR module to file.
mlir::LogicalResult writeMLIR(mlir::ModuleOp module, const std::string &path);

} // namespace fcc

#endif
