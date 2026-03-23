#ifndef LOOM_SYSTEMCOMPILER_KERNELPROFILER_H
#define LOOM_SYSTEMCOMPILER_KERNELPROFILER_H

#include "loom/SystemCompiler/L1CoreAssignment.h"

#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace loom {

/// Extracts kernel resource profiles from DFG (handshake.func) modules.
///
/// Walks the MLIR operations in each kernel's DFG to count operation types,
/// estimate SPM requirements, and compute minimum II bounds. The resulting
/// KernelProfile structs are used by the L1 core assignment solver.
class KernelProfiler {
public:
  /// Profile a single kernel's DFG module.
  ///
  /// \param kernelDFG  The MLIR module containing the kernel's handshake.func.
  /// \param ctx        MLIR context for type queries.
  /// \returns          Resource profile for the kernel.
  KernelProfile profile(mlir::ModuleOp kernelDFG, mlir::MLIRContext *ctx);

  /// Profile all kernels in a TDG module.
  ///
  /// Scans the module for all handshake.func operations and profiles each.
  ///
  /// \param tdgModule  The top-level TDG MLIR module.
  /// \param ctx        MLIR context for type queries.
  /// \returns          Vector of profiles, one per kernel found.
  std::vector<KernelProfile> profileAll(mlir::ModuleOp tdgModule,
                                        mlir::MLIRContext *ctx);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_KERNELPROFILER_H
