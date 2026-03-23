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
//===----------------------------------------------------------------------===//

#ifndef LOOM_ADG_SYSTEMADGBUILDER_H
#define LOOM_ADG_SYSTEMADGBUILDER_H

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
///   1. Create builder with system name
///   2. Register core types (from per-core ADGBuilder outputs)
///   3. Instantiate cores at grid positions
///   4. Configure NoC and shared memory
///   5. Call build() to generate the system fabric.module
///   6. Export to MLIR
class SystemADGBuilder {
public:
  explicit SystemADGBuilder(const std::string &systemName);
  ~SystemADGBuilder();

  SystemADGBuilder(const SystemADGBuilder &) = delete;
  SystemADGBuilder &operator=(const SystemADGBuilder &) = delete;

  /// Register a core type from its MLIR text representation.
  /// The typeName should match the fabric.module name in the MLIR.
  CoreTypeHandle registerCoreType(const std::string &typeName,
                                  const std::string &mlirText);

  /// Instantiate a core of the given type at a grid position.
  SystemCoreInstance instantiateCore(CoreTypeHandle type,
                                    const std::string &name, int row, int col);

  /// Set NoC configuration for the system.
  void setNoCSpec(const NoCSpec &spec);

  /// Set shared memory hierarchy configuration.
  void setSharedMemorySpec(const SharedMemorySpec &spec);

  /// Build the system-level fabric.module.
  /// Generates core instances, NoC connectivity, and shared memory.
  void build();

  /// Export the system MLIR to a file.
  void exportSystemMLIR(const std::string &path);

  /// Get the generated system MLIR as text.
  std::string getSystemMLIR() const;

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
