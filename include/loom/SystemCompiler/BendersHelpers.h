#ifndef LOOM_SYSTEMCOMPILER_BENDERSHELPERS_H
#define LOOM_SYSTEMCOMPILER_BENDERSHELPERS_H

#include "loom/SystemCompiler/BufferAllocator.h"
#include "loom/SystemCompiler/DMAScheduler.h"
#include "loom/SystemCompiler/ExecutionModel.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/L2CoreCompiler.h"
#include "loom/SystemCompiler/NoCScheduler.h"

#include "mlir/IR/BuiltinOps.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom {

/// Per-core result within the full Tapestry compilation.
struct TapestryCoreResult {
  std::string coreInstanceName;
  std::string coreType;
  mlir::ModuleOp adgModule;
  std::vector<std::string> assignedKernels;
  L2Result l2Result;
  std::vector<uint8_t> aggregateConfigBlob;
  std::optional<NoCRoute> nocRoute;
  std::optional<BufferAllocation> buffers;
};

/// Complete result from the Tapestry multi-core compilation flow.
struct TapestryCompilationResult {
  bool success = false;
  AssignmentResult finalAssignment;
  NoCSchedule finalNoCSchedule;
  BufferAllocationPlan finalBufferPlan;
  DMASchedule finalDMASchedule;
  std::vector<TapestryCoreResult> coreResults;

  /// Temporal schedule: per-core kernel ordering and timing.
  /// Populated after all L2 compilations succeed.
  std::optional<TemporalSchedule> temporalSchedule;
};

/// Build L2 assignments from L1 assignment results, kernel DFGs, and contracts.
std::vector<L2Assignment>
buildL2Assignments(const AssignmentResult &assignment,
                   const std::map<std::string, mlir::ModuleOp> &kernelDFGs,
                   const std::vector<ContractSpec> &contracts,
                   const SystemArchitecture &arch);

/// Update contract cost estimates from L2 compiler cost summaries.
void updateContractCosts(std::vector<ContractSpec> &contracts,
                         const std::vector<CoreCostSummary> &costSummaries);

/// Compute the combined objective value from assignment, NoC schedule, and L2
/// cost summaries.
double computeObjective(const AssignmentResult &assignment,
                        const NoCSchedule &nocSchedule,
                        const std::vector<CoreCostSummary> &costSummaries);

/// Extended objective computation that incorporates buffer allocation costs.
/// Buffer penalty: DRAM fallback is costlier than L2, which is costlier than
/// SPM. The penalty is added as a weighted term to the base objective.
double computeObjective(const AssignmentResult &assignment,
                        const NoCSchedule &nocSchedule,
                        const BufferAllocationPlan &bufferPlan,
                        const std::vector<CoreCostSummary> &costSummaries);

/// Assemble the final TapestryCompilationResult from all sub-results.
TapestryCompilationResult
assembleResult(const std::vector<L2Result> &l2Results,
               const std::vector<L2Assignment> &l2Assignments,
               const AssignmentResult &assignment,
               const NoCSchedule &nocSchedule,
               const BufferAllocationPlan &bufferPlan,
               const DMASchedule &dmaSchedule,
               const std::vector<CoreCostSummary> &costSummaries);

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_BENDERSHELPERS_H
