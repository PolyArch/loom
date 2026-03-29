//===-- SystemADGBuilder.h - System-level ADG Builder -------------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// C++ API for constructing multi-core system-level ADG descriptions.
// Takes per-core fabric.module definitions produced by ADGBuilder and composes
// them into a system-level fabric.module with NoC connectivity and shared
// memory hierarchy.
//
// The builder accepts core type definitions as mlir::ModuleOp (produced by
// ADGBuilder) and returns an mlir::ModuleOp from build(), eliminating the
// string intermediary that was previously used.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_ADG_SYSTEMADGBUILDER_H
#define LOOM_ADG_SYSTEMADGBUILDER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Handle and Specification Types
//===----------------------------------------------------------------------===//

/// Opaque handle to a registered core type.
struct CoreTypeHandle {
  unsigned id;
};

/// Describes an instantiated core within the system.
struct SystemCoreInstance {
  std::string instanceName;
  CoreTypeHandle coreType;
  int row;
  int col;
};

/// Network-on-Chip specification.
struct NoCSpec {
  enum Topology { MESH, RING, HIERARCHICAL };
  Topology topology = MESH;
  unsigned flitWidth = 32;
  unsigned virtualChannels = 2;
  unsigned linkBandwidth = 1;
  unsigned routerPipelineStages = 2;
};

/// Shared memory hierarchy specification.
struct SharedMemorySpec {
  uint64_t l2SizeBytes = 262144; // 256KB default
  unsigned numBanks = 4;
  unsigned bankWidthBytes = 32;
};

//===----------------------------------------------------------------------===//
// SystemADGBuilder
//===----------------------------------------------------------------------===//

/// Builder for constructing a multi-core system-level ADG.
///
/// Usage:
///   1. Create builder with an MLIRContext and system name
///   2. Register core types as ModuleOp (from per-core ADGBuilder outputs)
///   3. Instantiate cores at grid positions
///   4. Configure NoC and shared memory
///   5. Call build() to generate the system fabric.module (returns ModuleOp)
///   6. Optionally call exportMLIR() to write to file
///
/// The caller owns the MLIRContext and the returned ModuleOp remains valid
/// as long as the context is alive, even after the builder is destroyed.
class SystemADGBuilder {
public:
  /// Construct with an external MLIRContext (recommended).
  /// The caller owns the context. The returned ModuleOp from build() is
  /// valid as long as the context is alive.
  SystemADGBuilder(mlir::MLIRContext *ctx, const std::string &systemName);

  /// Convenience constructor that creates an internal MLIRContext.
  /// The returned ModuleOp from build() becomes invalid when the builder
  /// is destroyed.
  explicit SystemADGBuilder(const std::string &systemName);

  ~SystemADGBuilder();

  SystemADGBuilder(const SystemADGBuilder &) = delete;
  SystemADGBuilder &operator=(const SystemADGBuilder &) = delete;

  /// Register a core type from its ModuleOp representation.
  /// The coreModule should contain a fabric.module as produced by
  /// ADGBuilder. The typeName should match the fabric.module name.
  CoreTypeHandle registerCoreType(const std::string &typeName,
                                  mlir::ModuleOp coreModule);

  /// Instantiate a core of the given type at a grid position.
  SystemCoreInstance instantiateCore(CoreTypeHandle type,
                                    const std::string &name, int row, int col);

  /// Set NoC configuration for the system.
  void setNoCSpec(const NoCSpec &spec);

  /// Set shared memory hierarchy configuration.
  void setSharedMemorySpec(const SharedMemorySpec &spec);

  /// Build the system-level fabric.module.
  /// Generates core instances, NoC connectivity, and shared memory.
  /// Returns the fully assembled system ModuleOp.
  mlir::ModuleOp build();

  /// Export the built system MLIR to a file.
  /// Must call build() first.
  void exportMLIR(const std::string &path);

  /// Query the registered core instances.
  const std::vector<SystemCoreInstance> &getCoreInstances() const;

  /// Query the NoC specification.
  const NoCSpec &getNoCSpec() const;

  /// Query the shared memory specification.
  const SharedMemorySpec &getSharedMemorySpec() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace adg
} // namespace loom

#endif // LOOM_ADG_SYSTEMADGBUILDER_H
