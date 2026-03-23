#ifndef LOOM_SYSTEMCOMPILER_BENDERSDRIVER_H
#define LOOM_SYSTEMCOMPILER_BENDERSDRIVER_H

#include "loom/SystemCompiler/BufferAllocator.h"
#include "loom/SystemCompiler/CostSummary.h"
#include "loom/SystemCompiler/DMAScheduler.h"
#include "loom/SystemCompiler/InfeasibilityCut.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/L2CoreCompiler.h"
#include "loom/SystemCompiler/NoCScheduler.h"

#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/MapperOptions.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace loom {

//===----------------------------------------------------------------------===//
// Compilation Input
//===----------------------------------------------------------------------===//

/// Input bundle for the top-level Tapestry compilation driver.
struct TapestryCompilationInput {
  /// TDG module containing kernel bodies and contract annotations.
  mlir::ModuleOp tdgModule;

  /// Multi-core system architecture specification.
  SystemArchitecture architecture;

  /// Default mapper options used as the basis for L2 core compilations.
  MapperOptions baseMapperOpts;

  /// MLIR context for module creation during lowering.
  mlir::MLIRContext *ctx = nullptr;
};

//===----------------------------------------------------------------------===//
// Compilation Result
//===----------------------------------------------------------------------===//

/// Result of compiling a single core instance's assigned kernels.
struct CoreResult {
  /// Instance name (unique across the system).
  std::string coreInstanceName;

  /// Core type name (shared across instances of the same type).
  std::string coreType;

  /// ADG MLIR module describing this core's fabric.
  mlir::ModuleOp adgModule;

  /// Names of kernels assigned to this core.
  std::vector<std::string> assignedKernels;

  /// L2 compilation result (per-kernel mapping + config).
  L2Result l2Result;

  /// NoC route descriptors for this core's inter-core connections.
  NoCRoute nocRoute;

  /// Buffer allocations on this core.
  BufferAllocation buffers;

  /// Aggregate configuration blob (all kernels combined) for simulation
  /// and binary output.
  std::vector<uint8_t> aggregateConfigBlob;

  /// Per-hardware-node configuration slices for downstream RTL generation.
  std::vector<ConfigGen::ConfigSlice> configSlices;
};

/// Aggregate system-level performance metrics.
struct SystemMetrics {
  double throughput = 0.0;
  double criticalPathLatency = 0.0;
  double totalNoCTransferCycles = 0.0;
  double avgCoreUtilization = 0.0;
  double maxCoreUtilization = 0.0;
  unsigned numBendersIterations = 0;
  double compilationTimeSec = 0.0;
};

/// Record of a single Benders iteration for analysis and diagnostics.
struct IterationRecord {
  unsigned iteration = 0;
  AssignmentResult assignment;
  std::vector<InfeasibilityCut> cuts;
  std::vector<CoreCostSummary> costSummaries;
  bool converged = false;
};

/// Complete result of the Tapestry system-level compilation.
struct TapestryCompilationResult {
  bool success = false;

  /// Human-readable diagnostic messages (errors, warnings, info).
  std::string diagnostics;

  /// Per-core compilation results.
  std::vector<CoreResult> coreResults;

  /// System-level outputs from the final converged iteration.
  AssignmentResult finalAssignment;
  NoCSchedule finalNoCSchedule;
  BufferAllocationPlan finalBufferPlan;
  DMASchedule finalDMASchedule;

  /// Aggregate performance metrics.
  SystemMetrics metrics;

  /// Full iteration history for post-hoc analysis.
  std::vector<IterationRecord> iterationHistory;
};

//===----------------------------------------------------------------------===//
// Benders Decomposition Driver
//===----------------------------------------------------------------------===//

/// Options controlling the Benders decomposition compilation loop.
struct BendersDriverOptions {
  /// Maximum number of L1/L2 iterations.
  unsigned maxIterations = 10;

  /// Minimum relative objective improvement to continue iterating.
  double costTighteningThreshold = 0.01;

  /// Number of parallel L2 compilations (0 = auto, one per core).
  unsigned numParallelL2 = 0;

  /// Enable cost tightening feedback from L2 to L1.
  bool enableCostTightening = true;

  /// Enable infeasibility cut feedback from L2 to L1.
  bool enableInfeasibilityCuts = true;

  /// Perfect NoC mode: treat all inter-core transfers as zero cost
  /// for upper-bound throughput analysis.
  bool perfectNoC = false;

  /// Verbose logging output.
  bool verbose = false;
};

/// Orchestrates the bilevel Benders decomposition loop for system-level
/// compilation. The L1 master problem assigns kernels to cores and schedules
/// NoC transfers; the L2 subproblems map each core's assigned kernels to
/// hardware. Infeasibility cuts and cost feedback from L2 drive convergence.
class BendersDriver {
public:
  using Options = BendersDriverOptions;

  /// Run the full bilevel compilation loop.
  ///
  /// \param input  Compilation input (TDG module, architecture, mapper opts).
  /// \param opts   Driver options (iteration limits, convergence thresholds).
  /// \returns      Complete compilation result with per-core mappings and
  ///               system metrics.
  TapestryCompilationResult compile(const TapestryCompilationInput &input,
                                    const Options &opts);
};

//===----------------------------------------------------------------------===//
// Helper Function Declarations (defined in BendersHelpers.cpp)
//===----------------------------------------------------------------------===//

/// Build per-core L2 assignment bundles from the L1 assignment result.
std::vector<L2Assignment>
buildL2Assignments(const AssignmentResult &assignment,
                   const std::map<std::string, mlir::ModuleOp> &kernelDFGs,
                   const std::vector<ContractSpec> &contracts,
                   const SystemArchitecture &arch);

/// Update contract cost estimates using achieved metrics from L2 compilation.
void updateContractCosts(std::vector<ContractSpec> &contracts,
                         const std::vector<CoreCostSummary> &costSummaries);

/// Compute a scalar objective value from assignment, NoC schedule, and L2
/// cost summaries.
double computeObjective(const AssignmentResult &assignment,
                        const NoCSchedule &nocSchedule,
                        const std::vector<CoreCostSummary> &costSummaries);

/// Assemble a complete TapestryCompilationResult from per-core L2 results,
/// the final assignment, NoC schedule, buffer plan, and DMA schedule.
TapestryCompilationResult
assembleResult(const std::vector<L2Result> &l2Results,
               const std::vector<L2Assignment> &l2Assignments,
               const AssignmentResult &assignment,
               const NoCSchedule &nocSchedule,
               const BufferAllocationPlan &bufferPlan,
               const DMASchedule &dmaSchedule,
               const std::vector<CoreCostSummary> &costSummaries);

//===----------------------------------------------------------------------===//
// TDG Lowering Declarations (defined in TDGLowering.cpp)
//===----------------------------------------------------------------------===//

/// Lower TDG kernel bodies into per-kernel DFG (handshake.func) modules.
/// Returns a map from kernel name to its standalone MLIR module.
/// This is a stub implementation; the full pipeline uses existing Loom
/// lowering passes (SCF -> DFG).
std::map<std::string, mlir::ModuleOp>
lowerKernelsToDFG(mlir::ModuleOp tdgModule, mlir::MLIRContext *ctx);

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_BENDERSDRIVER_H
