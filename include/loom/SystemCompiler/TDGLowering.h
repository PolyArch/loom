//===-- TDGLowering.h - Kernel DFG lowering interface -------------*- C++ -*-===//
//
// Interface for lowering kernel sources to DFG (handshake.func) modules.
// Wraps the existing Loom SCF-to-DFG pipeline for use in the multi-core
// compilation flow.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_TDGLOWERING_H
#define LOOM_SYSTEMCOMPILER_TDGLOWERING_H

#include "loom/SystemCompiler/SystemTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace loom {
namespace tapestry {

/// Lower kernel MLIR modules from SCF/CF form to DFG (handshake.func) form.
///
/// For each KernelDesc whose dfgModule contains func.func operations in
/// SCF/CF form, runs the Loom lowering pipeline:
///   LLVM -> CF -> SCF -> DFG
///
/// Kernels that already contain handshake.func operations are left as-is.
///
/// Returns true on success (all kernels lowered). On failure, prints
/// diagnostics and returns false; partial results may remain in `kernels`.
bool lowerKernelsToDFG(std::vector<KernelDesc> &kernels,
                       mlir::MLIRContext &ctx);

/// Lower a single MLIR module through the SCF-to-DFG pipeline.
/// The module should contain func.func operations in SCF form.
/// Returns success/failure.
mlir::LogicalResult lowerModuleToDFG(mlir::ModuleOp module);

} // namespace tapestry
} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_TDGLOWERING_H
