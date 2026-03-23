//===-- PrecompiledKernelLoader.h - Load pre-lowered DFG files -----*- C++ -*-===//
//
// Loads pre-compiled DFG MLIR files (one per kernel) from a directory.
// This is a legitimate alternative to TDG-based lowering: the user provides
// already-lowered handshake.func DFG modules, and the loader parses them
// into KernelDesc entries for the multi-core pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_PRECOMPILEDKERNELLOADER_H
#define LOOM_SYSTEMCOMPILER_PRECOMPILEDKERNELLOADER_H

#include "loom/SystemCompiler/SystemTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"

#include <string>
#include <vector>

namespace loom {
namespace tapestry {

/// Load pre-compiled DFG MLIR files from a directory.
///
/// Scans `directory` for files matching *.mlir. Each file is parsed and
/// must contain a module with at least one handshake.func. The kernel name
/// is derived from the filename (without extension).
///
/// Returns the loaded KernelDesc entries. On parse failure for any file,
/// returns an empty vector and prints diagnostics.
std::vector<KernelDesc> loadPrecompiledKernels(const std::string &directory,
                                               mlir::MLIRContext &ctx);

/// Load a single pre-compiled DFG MLIR file.
///
/// The kernel name is derived from the filename stem.
/// Returns a KernelDesc with the parsed module, or a KernelDesc with a null
/// dfgModule on failure.
KernelDesc loadSingleKernel(const std::string &filePath,
                            mlir::MLIRContext &ctx);

/// Create a synthetic kernel DFG for testing.
/// Builds a simple handshake.func with the given name that computes
/// result = a + b (two i32 inputs, one i32 output).
KernelDesc createSyntheticAddKernel(const std::string &name,
                                    mlir::MLIRContext &ctx);

/// Create a synthetic kernel DFG with a multiply-accumulate pattern.
/// Builds a handshake.func: result = a * b + c.
KernelDesc createSyntheticMacKernel(const std::string &name,
                                    mlir::MLIRContext &ctx);

} // namespace tapestry
} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_PRECOMPILEDKERNELLOADER_H
