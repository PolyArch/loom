#ifndef LOOM_SYSTEMCOMPILER_L2CORECOMPILER_H
#define LOOM_SYSTEMCOMPILER_L2CORECOMPILER_H

#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MapperOptions.h"
#include "loom/Mapper/Types.h"
#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/CostSummary.h"
#include "loom/SystemCompiler/InfeasibilityCut.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace loom {

class Graph;
class MappingState;
class ADGFlattener;
struct MapperTimingSummary;

/// Assignment of kernels to a single core instance, produced by the L1 solver.
struct L2Assignment {
  std::string coreInstanceName;
  std::string coreType;
  mlir::ModuleOp coreADG; // The core's fabric MLIR module

  /// A single kernel assigned to this core.
  struct KernelAssignment {
    std::string kernelName;
    mlir::ModuleOp kernelDFG; // handshake.func MLIR module
    std::vector<ContractSpec> inputContracts;
    std::vector<ContractSpec> outputContracts;
    std::optional<unsigned> targetII;
  };
  std::vector<KernelAssignment> kernels;

  /// NoC port bindings: (ADG port index, contract edge name).
  std::vector<std::pair<unsigned, std::string>> nocPortBindings;
};

/// Result for a single kernel mapping attempt.
struct L2KernelResult {
  std::string kernelName;
  bool success = false;
  std::optional<Mapper::Result> mapperResult;
  std::optional<std::vector<uint8_t>> configBlob;
  std::optional<InfeasibilityCut> cut;
};

/// Aggregated result from the L2 core compiler.
struct L2Result {
  bool allKernelsMapped = false;
  CoreCostSummary costSummary;
  std::vector<L2KernelResult> kernelResults;
  std::vector<uint8_t> aggregateConfig;
};

/// Tracks cumulative hardware resource usage across sequential kernel mappings
/// on the same core, providing exclusion sets for subsequent mappings.
class ResourceTracker {
public:
  /// Record all hardware nodes used by a successful mapping.
  void addMapping(const MappingState &state, const Graph &adg);

  /// Get the set of hardware node IDs that are already occupied.
  const std::set<IdIndex> &getUsedNodes() const { return usedNodes_; }

  /// Get cumulative SPM bytes used.
  uint64_t getSpmBytesUsed() const { return spmBytesUsed_; }

  /// Add SPM usage.
  void addSpmUsage(uint64_t bytes) { spmBytesUsed_ += bytes; }

private:
  std::set<IdIndex> usedNodes_;
  uint64_t spmBytesUsed_ = 0;
};

/// Extract performance metrics from a successful mapping result.
KernelMetrics extractMetrics(const Mapper::Result &result, const Graph &dfg,
                             const Graph &adg, const ADGFlattener &flattener,
                             const std::string &kernelName);

/// Analyze a mapping failure and produce a structured infeasibility cut.
InfeasibilityCut analyzeFailure(const Mapper::Result &result, const Graph &dfg,
                                const Graph &adg,
                                const ADGFlattener &flattener,
                                const std::string &kernelName,
                                const std::string &coreType,
                                std::optional<unsigned> targetII);

/// L2 subproblem solver: maps one or more kernels to a single core.
class L2CoreCompiler {
public:
  /// Compile the assignment, mapping each kernel in sequence.
  L2Result compile(const L2Assignment &assignment,
                   const MapperOptions &baseMapperOpts,
                   mlir::MLIRContext *ctx);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_L2CORECOMPILER_H
