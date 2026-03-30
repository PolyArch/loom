//===-- TDGToSSGBuilder.h - TDG MLIR -> SSG conversion ------------*- C++ -*-===//
//
// Converts a TDG MLIR module (produced by tapestry::emitTDG) into a
// System Scheduling Graph (SSG) for consumption by the hierarchical compiler.
//
// The SSG is a SystemGraph<KernelNode, DataDependency> containing kernel nodes
// connected by data dependency edges. Each KernelNode carries compute profile
// data and variant information extracted from the corresponding DFG modules.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_TDGTOSSGBUILDER_H
#define LOOM_SYSTEMCOMPILER_TDGTOSSGBUILDER_H

#include "loom/Graph/SystemGraphTypes.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <map>
#include <string>

namespace loom {

//===----------------------------------------------------------------------===//
// TDGToSSGBuilder
//===----------------------------------------------------------------------===//

/// Converts a TDG MLIR module and per-kernel DFG modules into an SSG.
///
/// The builder:
///   1. Walks tdg.kernel ops to create KernelNode entries.
///   2. For each kernel, looks up the corresponding DFG module and profiles
///      it using KernelProfiler to populate computeProfile.
///   3. Walks tdg.contract ops to create DataDependency edges.
///   4. Validates the resulting graph (no duplicate kernel names, DAG check).
class TDGToSSGBuilder {
public:
  /// Build an SSG from a TDG MLIR module and per-kernel DFG modules.
  ///
  /// \param tdgModule   The TDG MLIR module containing tdg.graph, tdg.kernel,
  ///                    and tdg.contract ops.
  /// \param dfgModules  Map from kernel name to its DFG (handshake.func) module.
  ///                    Missing entries produce a warning; the kernel still
  ///                    appears in the SSG but with empty profile/variants.
  /// \param ctx         MLIR context for type queries.
  /// \returns           A populated SSG (SystemGraph<KernelNode, DataDependency>).
  SSG build(mlir::ModuleOp tdgModule,
            const std::map<std::string, mlir::ModuleOp> &dfgModules,
            mlir::MLIRContext &ctx);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_TDGTOSSGBUILDER_H
