//===-- HWOuterOptimizer.h - System-level hardware optimizer -------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// C++ interface for the OUTER-HW system-level hardware optimizer.
//
// Invokes the Python BO-based optimizer (scripts/dse/hw_outer_optimizer.py)
// via subprocess and parses the resulting SystemTopologySpec JSON into
// C++ data structures. Can be called from the C13 co-optimization loop.
//
// Input:  TDG module + contracts + options
// Output: System MLIR ADG (built via SystemADGBuilder)
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_HWOUTEROPTIMIZER_H
#define LOOM_SYSTEMCOMPILER_HWOUTEROPTIMIZER_H

#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
namespace json {
class Array;
class Object;
} // namespace json
} // namespace llvm

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace loom {

//===----------------------------------------------------------------------===//
// Core Type Library (C++ mirror of Python CoreTypeLibrary)
//===----------------------------------------------------------------------===//

/// High-level role classification for a core type.
enum class CoreRole {
  FP_HEAVY,
  CONTROL_HEAVY,
  MEMORY_HEAVY,
  BALANCED,
};

/// Convert CoreRole to string.
const char *coreRoleToString(CoreRole role);

/// Parse CoreRole from string.
CoreRole coreRoleFromString(const std::string &s);

/// Compute mix classification for combinatorial KHG types.
enum class ComputeMix {
  FP_MIX,   // FP-heavy compute mix
  INT_MIX,  // Integer-heavy compute mix
  MEM_MIX,  // Memory-heavy compute mix
};

/// Array size classification for combinatorial KHG types.
enum class ArraySize {
  SMALL,  // ~8 PEs (3x3)
  LARGE,  // ~12 PEs (4x3)
};

/// Total number of types in the fixed core type library.
constexpr unsigned NUM_LIBRARY_TYPES = 30;

/// Per-type design parameters mirroring Python CoreDesignParams from
/// core_type_library.py.  Captures FU counts, array geometry, SPM sizing,
/// and temporal PE configuration so that the workload JSON carries the full
/// metadata the Python optimizer needs for affinity scoring.
struct CoreTypeDesignParams {
  unsigned arrayRows = 2;
  unsigned arrayCols = 2;
  unsigned fuAluCount = 0;
  unsigned fuMulCount = 0;
  unsigned fuFpCount = 0;
  unsigned fuMemCount = 2;
  unsigned spmSizeKB = 0;
  bool isTemporal = false;
  unsigned instructionSlots = 0;
  unsigned numRegisters = 0;
  unsigned dataWidth = 32;
  bool hasFP = false;
  bool hasInt = true;
};

/// One entry in the core type library produced by OUTER-HW.
struct CoreTypeLibraryEntry {
  unsigned typeIndex = 0;

  /// Canonical type ID (e.g. "D1", "CFSY8").
  std::string typeId;

  CoreRole role = CoreRole::BALANCED;
  unsigned instanceCount = 1;

  /// Resource lower bounds for INNER-HW (C12) to satisfy.
  unsigned minPEs = 4;
  unsigned minSPMKB = 4;
  std::vector<std::string> requiredFUTypes;

  /// Names of kernels assigned to this core type.
  std::vector<std::string> assignedKernels;

  /// Combinatorial dimensions (for KHG types).
  ComputeMix computeMix = ComputeMix::INT_MIX;
  bool hasSPM = true;
  ArraySize arraySize = ArraySize::SMALL;

  /// Full design parameters for this core type (FU counts, array geometry,
  /// SPM sizing, temporal PE config).  Serialized into the workload JSON
  /// so the Python optimizer can perform affinity scoring.
  CoreTypeDesignParams designParams;
};

/// The complete core type library produced by OUTER-HW.
struct CoreTypeLibrary {
  std::vector<CoreTypeLibraryEntry> entries;

  unsigned numTypes() const { return entries.size(); }

  unsigned totalInstances() const {
    unsigned total = 0;
    for (const auto &e : entries)
      total += e.instanceCount;
    return total;
  }
};

//===----------------------------------------------------------------------===//
// System Topology Specification
//===----------------------------------------------------------------------===//

/// Core placement on the mesh: (typeIndex, instanceId, row, col).
struct CorePlacement {
  unsigned typeIndex = 0;
  unsigned instanceId = 0;
  int row = 0;
  int col = 0;
};

/// System-level topology specification produced by OUTER-HW.
struct SystemTopologySpec {
  /// NoC parameters.
  std::string nocTopology = "mesh";
  unsigned nocBandwidth = 1;
  unsigned meshRows = 2;
  unsigned meshCols = 2;

  /// Shared L2 memory parameters.
  uint64_t l2TotalSizeKB = 256;
  unsigned l2BankCount = 4;

  /// Core type library.
  CoreTypeLibrary coreLibrary;

  /// Core placement coordinates.
  std::vector<CorePlacement> corePlacements;
};

//===----------------------------------------------------------------------===//
// Optimizer Options and Result
//===----------------------------------------------------------------------===//

/// Configuration for the OUTER-HW optimizer.
struct HWOuterOptimizerOptions {
  /// Maximum BO iterations.
  unsigned maxIterations = 100;

  /// BO random seed.
  unsigned seed = 42;

  /// Path to the Python script.
  std::string pythonScriptPath;

  /// Path to the Python interpreter.
  std::string pythonBin = "python3";

  /// Tier-2 promotion threshold (Tier-1 score above this triggers Tier-2).
  double tier2Threshold = 0.3;

  /// Path to workload TDG file (passed to Python optimizer).
  std::string tdgPath;

  /// Path to tapestry-pipeline binary (for Tier-2 evaluation).
  std::string pipelineBin = "tapestry-pipeline";

  /// Working directory for temporary files.
  std::string workDir;

  /// Enable verbose logging.
  bool verbose = false;
};

/// Result of the OUTER-HW optimization.
struct HWOuterOptimizerResult {
  bool success = false;
  SystemTopologySpec topology;
  double bestScore = 0.0;
  unsigned iterationsUsed = 0;
  unsigned tdcRejections = 0;
  unsigned tier1Evaluations = 0;
  unsigned tier2Evaluations = 0;
  double wallTimeSec = 0.0;
  std::string diagnostics;
};

//===----------------------------------------------------------------------===//
// HWOuterOptimizer
//===----------------------------------------------------------------------===//

/// System-level hardware optimizer callable from C++ (C13 co-optimization).
///
/// Invokes the Python BO optimizer via subprocess to explore the system-level
/// design space (core type count, NoC topology, L2 sizing), then parses the
/// result JSON into a SystemTopologySpec. Optionally generates a system MLIR
/// ADG using SystemADGBuilder.
class HWOuterOptimizer {
public:
  /// Construct with options.
  explicit HWOuterOptimizer(const HWOuterOptimizerOptions &options);

  /// Run the optimization given contracts extracted from the TDG.
  ///
  /// \param contracts  Inter-kernel communication contracts.
  /// \param kernelProfiles  Kernel resource profiles from the TDG.
  /// \returns  Optimization result with the best topology.
  HWOuterOptimizerResult
  optimize(const std::vector<ContractSpec> &contracts,
           const std::vector<loom::KernelProfile> &kernelProfiles);

  /// Generate a system-level MLIR ADG from the optimization result.
  ///
  /// Uses SystemADGBuilder to produce a fabric.module with core instances,
  /// routers, links, and shared memory banks.
  ///
  /// \param result  The optimization result to convert.
  /// \param ctx     MLIR context for building the module.
  /// \returns       The generated system MLIR as text, or empty on failure.
  std::string generateSystemMLIR(const HWOuterOptimizerResult &result,
                                 mlir::MLIRContext *ctx);

  /// Get the current options.
  const HWOuterOptimizerOptions &getOptions() const { return options_; }

private:
  HWOuterOptimizerOptions options_;

  /// Write the workload profile JSON consumed by the Python optimizer.
  std::string writeWorkloadJSON(
      const std::vector<ContractSpec> &contracts,
      const std::vector<loom::KernelProfile> &kernelProfiles,
      const std::string &outputDir);

  /// Parse the topology spec JSON produced by the Python optimizer.
  bool parseTopologyJSON(const std::string &jsonPath,
                         SystemTopologySpec &outSpec);

  /// Parse a CoreTypeLibrary from a JSON object with a "core_types" key.
  bool parseCoreLibraryFromObject(const llvm::json::Object *libObj,
                                  CoreTypeLibrary &outLibrary);

  /// Parse a CoreTypeLibrary from a flat JSON array (Python output format).
  bool parseCoreLibraryFromArray(const llvm::json::Array *libArr,
                                 CoreTypeLibrary &outLibrary);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_HWOUTEROPTIMIZER_H
