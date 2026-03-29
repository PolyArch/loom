#ifndef LOOM_SYSTEMCOMPILER_L1COREASSIGNMENT_H
#define LOOM_SYSTEMCOMPILER_L1COREASSIGNMENT_H

#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/InfeasibilityCut.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// System Architecture Description
//===----------------------------------------------------------------------===//

/// Specification for a single core type in the system.
struct CoreTypeSpec {
  std::string typeName;
  unsigned instanceCount = 0;

  /// Scratchpad memory size in bytes.
  uint64_t spmBytes = 0;

  /// Resource summary derived from ADG analysis.
  unsigned numPEs = 0;
  unsigned numFUs = 0;

  /// Map of operation type name to the number of FUs supporting it.
  std::map<std::string, unsigned> fuTypeCounts;
};

/// Network-on-Chip specification for core assignment cost modeling.
struct L1NoCSpec {
  enum Topology { MESH, RING, HIERARCHICAL };
  Topology topology = MESH;
  unsigned meshRows = 1;
  unsigned meshCols = 1;
  unsigned flitWidth = 32;
  unsigned virtualChannels = 2;
  unsigned linkBandwidth = 1;
  unsigned routerPipelineStages = 2;
};

/// Shared memory hierarchy specification.
struct L1SharedMemorySpec {
  uint64_t l2SizeBytes = 262144; // 256KB default
  unsigned numBanks = 4;
  unsigned bankWidthBytes = 32;
};

/// Complete system architecture specification for L1 assignment.
struct SystemArchitecture {
  std::vector<CoreTypeSpec> coreTypes;
  L1NoCSpec nocSpec;
  L1SharedMemorySpec sharedMemSpec;

  /// Total number of core instances across all types.
  unsigned totalCoreInstances() const;

  /// Get the core type spec for a given instance index.
  /// Instances are numbered sequentially: type0 instances, then type1, etc.
  const CoreTypeSpec &typeForInstance(unsigned instanceIdx) const;

  /// Get the core type index for a given instance index.
  unsigned typeIndexForInstance(unsigned instanceIdx) const;

  /// Get the core type name for a given instance index.
  const std::string &typeNameForInstance(unsigned instanceIdx) const;

  /// Get the first instance index for a given core type index.
  unsigned firstInstanceOfType(unsigned typeIdx) const;
};

//===----------------------------------------------------------------------===//
// Kernel Profile
//===----------------------------------------------------------------------===//

/// Resource profile of a kernel, extracted from DFG analysis.
struct KernelProfile {
  std::string name;

  /// Required operations: op_type_name -> count needed.
  std::map<std::string, unsigned> requiredOps;

  /// Estimated scratchpad memory requirement in bytes.
  uint64_t estimatedSPMBytes = 0;

  /// Estimated minimum initiation interval.
  unsigned estimatedMinII = 1;

  /// Estimated total compute cycles.
  double estimatedComputeCycles = 0.0;

  /// Core type preferences from kernel analysis (empty = no preference).
  std::set<std::string> preferredCoreTypes;

  /// Total operation count (sum of all requiredOps values).
  unsigned totalOpCount() const;
};

//===----------------------------------------------------------------------===//
// Assignment Result
//===----------------------------------------------------------------------===//

/// Per-core assignment information in the result.
struct CoreAssignment {
  unsigned coreInstanceIdx = 0;
  std::string coreTypeName;
  std::vector<std::string> assignedKernels;
  double estimatedUtilization = 0.0;
};

/// Result of the L1 core assignment solver.
struct AssignmentResult {
  bool feasible = false;

  /// Kernel name -> core instance index.
  std::map<std::string, unsigned> kernelToCore;

  /// Per-core assignment details.
  std::vector<CoreAssignment> coreAssignments;

  /// Optional kernel start times (from solver or prior scheduling).
  std::map<std::string, uint64_t> kernelStartTimes;

  /// Objective value breakdown.
  double totalCriticalPathLatency = 0.0;
  double totalNoCTransferCost = 0.0;
  double totalLocalityBonus = 0.0;
  double objectiveValue = 0.0;
};

//===----------------------------------------------------------------------===//
// L1 Core Assigner
//===----------------------------------------------------------------------===//

/// Options for the L1 core assignment solver.
struct L1AssignerOptions {
  double latencyWeight = 1.0;
  double nocCostWeight = 0.5;
  double loadBalanceWeight = 0.3;

  /// Maximum allowed utilization gap (max - min) for load balancing,
  /// expressed as a fraction in [0, 1].
  double loadBalanceThreshold = 0.3;

  /// Maximum solver wall-clock time in seconds.
  unsigned maxSolverTimeSec = 60;

  /// Whether to add data locality bonus for co-located connected kernels.
  bool enableDataLocality = true;

  /// Number of solver worker threads (0 = auto-detect).
  unsigned numWorkers = 0;

  /// Verbosity flag.
  bool verbose = false;
};

/// L1 master problem solver: assigns kernels to core instances using CP-SAT.
///
/// This is the master problem in a Benders decomposition bilevel compiler.
/// Decision variables x[k][c] = 1 if kernel k is assigned to core instance c.
/// The solver respects capacity constraints, type compatibility, Benders
/// infeasibility cuts from previous L2 iterations, and load balancing.
class L1CoreAssigner {
public:
  /// Solve the core assignment problem.
  ///
  /// \param kernels    Resource profiles for each kernel to assign.
  /// \param contracts  Inter-kernel communication contracts (edges in TDG).
  /// \param arch       System architecture specification.
  /// \param cuts       Infeasibility cuts from previous L2 iterations.
  /// \param opts       Solver options.
  /// \returns          Assignment result (feasible flag + mapping).
  AssignmentResult solve(const std::vector<KernelProfile> &kernels,
                         const std::vector<ContractSpec> &contracts,
                         const SystemArchitecture &arch,
                         const std::vector<InfeasibilityCut> &cuts,
                         const L1AssignerOptions &opts);
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Compute Manhattan distance between two core instances on a mesh NoC.
int manhattanDistance(unsigned coreA, unsigned coreB, unsigned meshCols);

/// Estimate data element size in bytes from a type name string.
unsigned estimateElementSize(const std::string &dataTypeName);

/// Check if a kernel is compatible with a core type
/// (all required ops are supported by the core's FU types).
bool isKernelCompatible(const KernelProfile &kernel,
                        const CoreTypeSpec &coreType);

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_L1COREASSIGNMENT_H
