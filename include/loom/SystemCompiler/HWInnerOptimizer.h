//===-- HWInnerOptimizer.h - Per-core ADG optimizer ---------------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// C++ interface for the INNER-HW per-core ADG optimizer (C12).
//
// Given a core type specification from OUTER-HW (C11), optimizes the concrete
// ADG parameters across 13 design dimensions to minimize area while ensuring
// all assigned kernels map successfully. Uses a three-tier evaluation:
//   Tier A: Analytical derivation (area model from FU/PE counts)
//   Tier B: Small-scale BO + real Loom mapper evaluation
//   Tier C: Simulation (deferred, returns Tier B results)
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_HWINNEROPTIMIZER_H
#define LOOM_SYSTEMCOMPILER_HWINNEROPTIMIZER_H

#include "loom/SystemCompiler/HWOuterOptimizer.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace loom {

//===----------------------------------------------------------------------===//
// PE Type Enumeration
//===----------------------------------------------------------------------===//

enum class PEType {
  SPATIAL,
  TEMPORAL,
};

const char *peTypeToString(PEType t);
PEType peTypeFromString(const std::string &s);

//===----------------------------------------------------------------------===//
// Switch Type Enumeration
//===----------------------------------------------------------------------===//

enum class SwitchType {
  SPATIAL,
  TEMPORAL,
};

const char *switchTypeToString(SwitchType t);
SwitchType switchTypeFromString(const std::string &s);

//===----------------------------------------------------------------------===//
// Routing Topology Enumeration
//===----------------------------------------------------------------------===//

enum class RoutingTopology {
  CHESS,
  MESH,
  LATTICE,
  RING,
};

const char *topologyToString(RoutingTopology t);
RoutingTopology topologyFromString(const std::string &s);

//===----------------------------------------------------------------------===//
// Per-Core Design Parameters (13 Dimensions)
//===----------------------------------------------------------------------===//

/// Concrete design parameters for a single core's ADG across all 13 dimensions.
struct CoreDesignParams {
  // --- Dimension 1: PE type ---
  PEType peType = PEType::SPATIAL;

  // --- Dimension 2: Array dimensions ---
  unsigned arrayRows = 2;
  unsigned arrayCols = 2;

  // --- Dimension 3: Data width ---
  unsigned dataWidth = 32;

  // --- Dimension 4: FU repertoire ---
  /// Set of operation names this core supports (e.g. "arith.addi").
  std::set<std::string> fuRepertoire;

  // --- Dimension 5: FU body structure ---
  /// If true, allow multi-op FU bodies (configurable DAG via fabric.mux).
  bool multiOpFUBodies = false;

  // --- Dimension 6: Switch type ---
  SwitchType switchType = SwitchType::SPATIAL;

  // --- Dimension 7: Switch decomposability ---
  /// Sub-lane decomposable bit width. -1 means no decomposition.
  int decomposableBits = -1;

  // --- Dimension 8: SPM ---
  unsigned spmSizeKB = 4;
  unsigned spmLdPorts = 1;
  unsigned spmStPorts = 1;

  // --- Dimension 9: External memory ---
  unsigned extmemCount = 2;
  unsigned extmemLdPorts = 1;
  unsigned extmemStPorts = 1;

  // --- Dimension 10: Routing topology ---
  RoutingTopology topology = RoutingTopology::CHESS;

  // --- Dimension 11: Temporal PE params ---
  unsigned instructionSlots = 4;
  unsigned numRegisters = 4;
  unsigned regFifoDepth = 0;
  bool shareOperandBuffer = false;
  unsigned operandBufferSize = 0;

  // --- Dimension 12: Scalar I/O ---
  unsigned scalarInputs = 3;
  unsigned scalarOutputs = 1;

  // --- Dimension 13: Connectivity matrix ---
  /// If empty, full crossbar is assumed.
  std::vector<std::vector<bool>> connectivity;

  /// Total PE count for this configuration.
  unsigned totalPEs() const { return arrayRows * arrayCols; }
};

//===----------------------------------------------------------------------===//
// Per-Kernel Mapping Result
//===----------------------------------------------------------------------===//

/// Result of mapping a single kernel to a candidate ADG.
struct KernelMappingResult {
  std::string kernelName;
  bool success = false;
  unsigned achievedII = 0;
  double mappingTimeSec = 0.0;
};

//===----------------------------------------------------------------------===//
// ADG Optimization Result
//===----------------------------------------------------------------------===//

/// Full result of per-core ADG optimization.
struct ADGOptResult {
  bool success = false;

  /// The optimized design parameters.
  CoreDesignParams params;

  /// Analytical area estimate (in arbitrary area units).
  double areaEstimate = 0.0;

  /// MLIR text of the generated fabric.module.
  std::string adgMLIR;

  /// Per-kernel mapping results from the best candidate (Tier B).
  std::vector<KernelMappingResult> mappingResults;

  /// Optimization statistics.
  unsigned tier1Evaluations = 0;
  unsigned tier2Evaluations = 0;
  unsigned tier2Successes = 0;
  double wallTimeSec = 0.0;
  std::string diagnostics;
};

//===----------------------------------------------------------------------===//
// Inner Optimizer Options
//===----------------------------------------------------------------------===//

/// Configuration options for the INNER-HW optimizer.
struct HWInnerOptimizerOptions {
  /// Enable Tier-B (BO + mapper) optimization. If false, only Tier-A
  /// analytical derivation is used.
  bool tier2Enabled = true;

  /// Maximum BO iterations for Tier-B.
  unsigned maxInnerIter = 30;

  /// BO random seed.
  unsigned seed = 42;

  /// Mapper timeout per kernel per candidate (seconds).
  double mapperTimeoutSec = 10.0;

  /// Mapper seed.
  unsigned mapperSeed = 0;

  /// Enable Tier-C simulation (deferred: returns Tier-B results).
  bool tier3Enabled = false;

  /// PE type selection parallelism threshold.
  /// If parallelism_ratio > threshold, select spatial PE.
  double parallelismThreshold = 3.0;

  /// Maximum array dimension for BO exploration.
  unsigned maxArrayDim = 10;

  /// Minimum array dimension.
  unsigned minArrayDim = 2;

  /// Working directory for temporary ADG files.
  std::string workDir;

  /// Enable verbose logging.
  bool verbose = false;
};

//===----------------------------------------------------------------------===//
// Area Model
//===----------------------------------------------------------------------===//

/// Analytical area model for a core design.
/// Returns area in abstract units proportional to transistor count.
double estimateCoreArea(const CoreDesignParams &params);

/// Estimate FU area for a single operation type.
double estimateFUArea(const std::string &opName, unsigned dataWidth);

//===----------------------------------------------------------------------===//
// PE Type Selection
//===----------------------------------------------------------------------===//

/// Compute parallelism ratio from kernel operation profiles.
///   parallelism_ratio = max(DFG_concurrent_ops) / unique_op_types
double computeParallelismRatio(
    const std::vector<KernelProfile> &profiles);

/// Select PE type based on parallelism ratio heuristic.
PEType selectPEType(const std::vector<KernelProfile> &profiles,
                    double threshold);

//===----------------------------------------------------------------------===//
// FU Repertoire
//===----------------------------------------------------------------------===//

/// Compute the union of required operations across kernel profiles.
std::set<std::string> computeRequiredFURepertoire(
    const std::vector<KernelProfile> &profiles);

/// Attempt to prune one FU from the repertoire. Returns the pruned set
/// if all kernels can still map, or the original set if not.
std::set<std::string> tryPruneFU(const std::set<std::string> &repertoire,
                                 const std::string &candidate);

//===----------------------------------------------------------------------===//
// ADG Generation
//===----------------------------------------------------------------------===//

/// Build a concrete Fabric MLIR ADG string from design parameters.
/// The result can be parsed into an MLIR ModuleOp.
std::string buildADGFromParams(const CoreDesignParams &params,
                               const std::string &moduleName);

//===----------------------------------------------------------------------===//
// Tier-A: Analytical Derivation
//===----------------------------------------------------------------------===//

/// Derive initial ADG parameters from core type constraints and kernel profiles.
CoreDesignParams deriveInitialParams(
    const CoreTypeLibraryEntry &coreType,
    const std::vector<KernelProfile> &assignedProfiles,
    const HWInnerOptimizerOptions &opts);

//===----------------------------------------------------------------------===//
// HWInnerOptimizer
//===----------------------------------------------------------------------===//

/// Per-core ADG optimizer. Called once per core type from OUTER-HW.
///
/// Three-tier evaluation:
///   Tier A: Analytical lower-bound derivation (microseconds)
///   Tier B: Small-scale BO with real Loom mapper (seconds)
///   Tier C: Simulation for Pareto candidates (deferred)
///
/// Usage:
///   HWInnerOptimizer optimizer(options);
///   auto result = optimizer.optimize(coreType, profiles, ctx);
///   if (result.success) { /* use result.adgMLIR */ }
class HWInnerOptimizer {
public:
  explicit HWInnerOptimizer(const HWInnerOptimizerOptions &options);

  /// Run the full optimization pipeline for one core type.
  ///
  /// \param coreType         Core type specification from OUTER-HW (C11).
  /// \param assignedProfiles Kernel profiles for kernels assigned to this type.
  /// \param ctx              MLIR context (for ADG parsing if needed).
  /// \returns                Optimization result with the best ADG.
  ADGOptResult optimize(const CoreTypeLibraryEntry &coreType,
                        const std::vector<KernelProfile> &assignedProfiles,
                        mlir::MLIRContext *ctx);

  /// Get the current options.
  const HWInnerOptimizerOptions &getOptions() const { return options_; }

private:
  HWInnerOptimizerOptions options_;

  /// Tier-A: derive initial parameters analytically.
  CoreDesignParams runTierA(const CoreTypeLibraryEntry &coreType,
                            const std::vector<KernelProfile> &profiles);

  /// Tier-B: small-scale BO with real mapper verification.
  ADGOptResult runTierB(const CoreDesignParams &initial,
                        const CoreTypeLibraryEntry &coreType,
                        const std::vector<KernelProfile> &profiles,
                        mlir::MLIRContext *ctx);

  /// Evaluate a single candidate: build ADG, map all kernels.
  /// Returns (areaIfAllMapped, mappingResults). Area is negative infinity
  /// if any kernel fails to map.
  std::pair<double, std::vector<KernelMappingResult>>
  evaluateCandidate(const CoreDesignParams &candidate,
                    const CoreTypeLibraryEntry &coreType,
                    const std::string &moduleName,
                    mlir::MLIRContext *ctx);

  /// Generate a neighbor candidate from the current best for BO exploration.
  CoreDesignParams perturbCandidate(const CoreDesignParams &base,
                                    unsigned iteration);
};

//===----------------------------------------------------------------------===//
// Batch Optimization (for parallel core type optimization)
//===----------------------------------------------------------------------===//

/// Optimize multiple core types. Each core type is optimized independently.
/// Results are indexed by core type index.
std::vector<ADGOptResult> optimizeAllCoreTypes(
    const CoreTypeLibrary &library,
    const std::vector<KernelProfile> &allProfiles,
    mlir::MLIRContext *ctx,
    const HWInnerOptimizerOptions &opts);

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_HWINNEROPTIMIZER_H
