//===-- SystemTypes.h - Multi-core system data types ---------------*- C++ -*-===//
//
// Core data types for the Tapestry multi-core compilation pipeline.
// Defines the SystemArchitecture, ContractSpec, L2Assignment, and
// BendersResult structures that tie together the ADG, DFG, and Mapper.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_SYSTEMTYPES_H
#define LOOM_SYSTEMCOMPILER_SYSTEMTYPES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {
namespace tapestry {

/// Description of one core type in the multi-core system.
struct CoreTypeDesc {
  std::string name;
  unsigned numInstances = 1;

  /// The ADG (fabric.module) for this core type, as an MLIR module.
  /// Owned externally (by SystemArchitecture).
  mlir::ModuleOp adgModule;

  /// Capacity estimates used by the master problem.
  unsigned totalPEs = 0;
  unsigned totalFUs = 0;
  unsigned spmSizeBytes = 0;
};

/// A complete multi-core system architecture.
struct SystemArchitecture {
  std::string name;
  std::vector<CoreTypeDesc> coreTypes;
};

/// A communication contract between two kernels.
struct ContractSpec {
  std::string producerKernel;
  std::string consumerKernel;
  std::string dataType;
  uint64_t elementCount = 0;
  uint64_t bandwidthBytesPerCycle = 0;

  /// Which core type index the producer/consumer are assigned to.
  /// Set by the master problem.
  int producerCoreType = -1;
  int consumerCoreType = -1;

  /// Communication cost computed from assignment.
  double communicationCost = 0.0;
};

/// A kernel descriptor for the multi-core pipeline.
struct KernelDesc {
  std::string name;

  /// The lowered DFG module containing a handshake.func for this kernel.
  mlir::ModuleOp dfgModule;

  /// Resource requirements estimated from the DFG.
  unsigned requiredPEs = 0;
  unsigned requiredFUs = 0;
  unsigned requiredMemoryBytes = 0;
};

/// Per-kernel mapping assignment from the L2 (sub-problem).
struct L2Assignment {
  std::string kernelName;
  int coreTypeIndex = -1;
  int coreInstanceIndex = -1;

  /// The ADG module for the assigned core type.
  mlir::ModuleOp coreADG;

  /// Whether the sub-problem mapper succeeded.
  bool mappingSuccess = false;

  /// Mapper quality metrics fed back to the master.
  double mappingCost = 0.0;
  double routingCongestion = 0.0;
  unsigned unroutedEdges = 0;
};

/// Result of the full Benders compilation.
struct BendersResult {
  bool success = false;
  unsigned iterations = 0;
  double totalCost = 0.0;
  std::string diagnostics;

  std::vector<L2Assignment> assignments;
};

/// Configuration for the Benders decomposition driver.
struct BendersConfig {
  unsigned maxIterations = 10;
  double mapperBudgetSeconds = 15.0;
  unsigned mapperSeed = 0;
  bool verbose = false;
};

/// Drives the Benders decomposition for heterogeneous multi-core compilation.
///
/// Takes a SystemArchitecture, a set of kernel DFGs, and inter-kernel
/// contracts, then iteratively partitions kernels across core types using
/// an L1 master / L2 sub-problem decomposition.
class BendersDriver {
public:
  BendersDriver(const SystemArchitecture &arch,
                std::vector<KernelDesc> kernels,
                std::vector<ContractSpec> contracts,
                mlir::MLIRContext &ctx);

  /// Run the decomposition and return the result.
  BendersResult compile(const BendersConfig &config);

private:
  SystemArchitecture arch_;
  std::vector<KernelDesc> kernels_;
  std::vector<ContractSpec> contracts_;
  mlir::MLIRContext &ctx_;
};

} // namespace tapestry
} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_SYSTEMTYPES_H
