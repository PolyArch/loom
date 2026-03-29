#ifndef LOOM_SYSTEMCOMPILER_TDCVERIFICATION_H
#define LOOM_SYSTEMCOMPILER_TDCVERIFICATION_H

#include "loom/SystemCompiler/BufferAllocator.h"
#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/NoCScheduler.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Origin tracking for inferred vs user-specified dimensions
//===----------------------------------------------------------------------===//

/// Tracks whether a contract dimension was explicitly provided by the
/// programmer, filled in by the inferrer, or left absent (no constraint).
enum class DimensionOrigin {
  USER_SPECIFIED, ///< Programmer explicitly set this dimension
  INFERRED,       ///< Inferrer filled in a conservative default
  ABSENT          ///< No value; dimension is unconstrained
};

/// Per-dimension origin tracking for a single TDCEdgeSpec.
/// Used by the verifier to determine which dimensions to check.
struct TDCEdgeSpecOrigin {
  DimensionOrigin ordering = DimensionOrigin::ABSENT;
  DimensionOrigin throughput = DimensionOrigin::ABSENT;
  DimensionOrigin placement = DimensionOrigin::ABSENT;
  DimensionOrigin shape = DimensionOrigin::ABSENT;
};

//===----------------------------------------------------------------------===//
// Contract Inference
//===----------------------------------------------------------------------===//

/// Result of contract inference: inferred specs + origin tracking.
struct TDCInferenceResult {
  std::vector<TDCEdgeSpec> edgeSpecs;
  std::vector<TDCEdgeSpecOrigin> edgeOrigins;
  std::vector<TDCPathSpec> pathSpecs;

  /// Errors encountered during inference (e.g. path referencing missing edge).
  std::vector<std::string> errors;

  bool hasErrors() const { return !errors.empty(); }
};

/// Contract inferrer: fills missing optional dimensions with conservative
/// defaults. Inference defaults guide the compiler but are not verified
/// against the programmer's intent (only USER_SPECIFIED dimensions are
/// checked by verification).
///
/// Conservative defaults:
///   - Missing ordering   -> Ordering::FIFO (preserves program order)
///   - Missing throughput  -> remains nullopt (no performance floor)
///   - Missing placement   -> Placement::AUTO (compiler decides)
///   - Missing shape       -> remains nullopt (compiler infers from analysis)
class TDCContractInferrer {
public:
  /// Infer defaults for edge specs and validate path spec references.
  /// The input vectors are copied; originals are not modified.
  TDCInferenceResult infer(const std::vector<TDCEdgeSpec> &edgeSpecs,
                           const std::vector<TDCPathSpec> &pathSpecs) const;
};

//===----------------------------------------------------------------------===//
// Verification result structs
//===----------------------------------------------------------------------===//

/// Per-edge verification result.
struct TDCEdgeVerificationResult {
  std::string producerKernel;
  std::string consumerKernel;

  bool orderingSatisfied = true;
  bool throughputSatisfied = true;
  bool placementSatisfied = true;
  bool shapeSatisfied = true;

  std::optional<double> achievedThroughput;

  /// Human-readable diagnostic when any check fails.
  std::string diagnostic;

  bool allSatisfied() const {
    return orderingSatisfied && throughputSatisfied && placementSatisfied &&
           shapeSatisfied;
  }
};

/// Per-path verification result.
struct TDCPathVerificationResult {
  std::string startProducer;
  std::string startConsumer;
  std::string endProducer;
  std::string endConsumer;

  bool latencySatisfied = true;

  std::optional<int64_t> achievedLatency;

  /// Human-readable diagnostic when the check fails.
  std::string diagnostic;
};

/// Top-level verification report aggregating all results.
struct TDCVerificationReport {
  std::vector<TDCEdgeVerificationResult> edgeResults;
  std::vector<TDCPathVerificationResult> pathResults;

  bool allSatisfied = true;
};

//===----------------------------------------------------------------------===//
// Dynamic verification metric interfaces
//===----------------------------------------------------------------------===//

/// Per-edge metrics measured by cycle-accurate simulation.
struct DynamicEdgeMetrics {
  std::string producerKernel;
  std::string consumerKernel;

  /// Sustained throughput in elements per cycle.
  double sustainedThroughput = 0.0;

  /// Number of observed ordering violations (out-of-order deliveries).
  int64_t orderingViolationCount = 0;
};

/// Per-path metrics measured by cycle-accurate simulation.
struct DynamicPathMetrics {
  std::string startProducer;
  std::string startConsumer;
  std::string endProducer;
  std::string endConsumer;

  /// Observed end-to-end latency in cycles.
  int64_t observedLatency = 0;
};

//===----------------------------------------------------------------------===//
// Compile-time tile information for shape verification
//===----------------------------------------------------------------------===//

/// Tile dimensions produced by the tiling engine for an edge.
struct EdgeTileDimensions {
  std::string producerKernel;
  std::string consumerKernel;

  /// Actual tile dimensions chosen by the compiler.
  std::vector<int64_t> tileDims;
};

//===----------------------------------------------------------------------===//
// Verification input bundle
//===----------------------------------------------------------------------===//

/// All compile-time artifacts needed for static verification.
struct StaticVerificationInputs {
  AssignmentResult assignment;
  BufferAllocationPlan bufferPlan;
  NoCSchedule nocSchedule;
  std::vector<EdgeTileDimensions> tileDimensions;
};

//===----------------------------------------------------------------------===//
// Top-level verification entry point
//===----------------------------------------------------------------------===//

/// Run post-compilation contract verification.
///
/// Static verification (shape, placement, ordering) is always performed.
/// Dynamic verification (throughput, latency, ordering violations) is
/// performed only when dynamic metrics are provided.
///
/// Only dimensions marked USER_SPECIFIED in the origin vector are checked.
/// Dimensions marked INFERRED or ABSENT are skipped.
TDCVerificationReport
verifyContracts(const std::vector<TDCEdgeSpec> &edgeSpecs,
                const std::vector<TDCEdgeSpecOrigin> &edgeOrigins,
                const std::vector<TDCPathSpec> &pathSpecs,
                const StaticVerificationInputs &staticInputs,
                const std::vector<DynamicEdgeMetrics> *dynamicEdgeMetrics,
                const std::vector<DynamicPathMetrics> *dynamicPathMetrics);

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_TDCVERIFICATION_H
