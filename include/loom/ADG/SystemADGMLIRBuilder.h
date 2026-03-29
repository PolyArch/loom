//===-- SystemADGMLIRBuilder.h - MLIR builder for system ADG ------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Builder class that constructs a system-level fabric.module using the MLIR
// OpBuilder API. Core type definitions are provided as mlir::ModuleOp
// references and cloned directly into the system module, eliminating the
// string parsing intermediary.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_ADG_SYSTEMADGMLIRBUILDER_H
#define LOOM_ADG_SYSTEMADGMLIRBUILDER_H

#include "loom/ADG/SystemADGBuilder.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <string>
#include <vector>

namespace loom {
namespace adg {

/// Builds a system-level MLIR module containing:
///   - Per-core fabric.module definitions (cloned from registered ModuleOps)
///   - A system fabric.module with:
///     - fabric.instance ops for each core
///     - fabric.router ops for NoC routers
///     - fabric.shared_mem ops for L2 banks and DRAM
///     - fabric.noc_link ops for topology connectivity
class SystemADGMLIRBuilder {
public:
  /// Core type descriptor for the builder.
  struct CoreType {
    std::string typeName;
    mlir::ModuleOp coreModule;
    unsigned id;
  };

  /// Build a system-level MLIR module from the given configuration.
  /// Returns the top-level mlir::ModuleOp owning the system fabric.
  ///
  /// The caller owns the returned module (through the context).
  static mlir::ModuleOp build(mlir::MLIRContext *ctx,
                              const std::string &systemName,
                              const std::vector<CoreType> &coreTypes,
                              const std::vector<SystemCoreInstance> &instances,
                              const NoCSpec &nocSpec,
                              const SharedMemorySpec &sharedMemSpec);

private:
  /// Emit core type fabric.module definitions into the wrapper module.
  /// Clones fabric.module ops from the provided ModuleOps.
  static void emitCoreTypeDefinitions(mlir::OpBuilder &builder,
                                      mlir::ModuleOp wrapper,
                                      const std::vector<CoreType> &coreTypes,
                                      const std::vector<SystemCoreInstance> &instances);

  /// Emit the system-level fabric.module with instances, routers, shared
  /// memory, and NoC links.
  static void emitSystemModule(mlir::OpBuilder &builder,
                               mlir::ModuleOp wrapper,
                               const std::string &systemName,
                               const std::vector<CoreType> &coreTypes,
                               const std::vector<SystemCoreInstance> &instances,
                               const NoCSpec &nocSpec,
                               const SharedMemorySpec &sharedMemSpec);

  /// Emit fabric.router ops for each core position.
  static void emitRouters(mlir::OpBuilder &builder, mlir::Location loc,
                          const std::vector<SystemCoreInstance> &instances,
                          const NoCSpec &nocSpec);

  /// Emit fabric.shared_mem ops for L2 banks and external memory.
  static void emitSharedMemory(mlir::OpBuilder &builder, mlir::Location loc,
                               const SharedMemorySpec &sharedMemSpec);

  /// Emit fabric.noc_link ops for mesh topology.
  static void emitMeshLinks(mlir::OpBuilder &builder, mlir::Location loc,
                            const std::vector<SystemCoreInstance> &instances,
                            const NoCSpec &nocSpec);

  /// Emit fabric.noc_link ops for ring topology.
  static void emitRingLinks(mlir::OpBuilder &builder, mlir::Location loc,
                            const std::vector<SystemCoreInstance> &instances,
                            const NoCSpec &nocSpec);

  /// Emit fabric.noc_link ops for hierarchical topology.
  static void emitHierarchicalLinks(mlir::OpBuilder &builder, mlir::Location loc,
                                    const std::vector<SystemCoreInstance> &instances,
                                    const NoCSpec &nocSpec);
};

} // namespace adg
} // namespace loom

#endif // LOOM_ADG_SYSTEMADGMLIRBUILDER_H
