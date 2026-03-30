#ifndef LOOM_SYSTEMCOMPILER_TDCVERIFICATION_H
#define LOOM_SYSTEMCOMPILER_TDCVERIFICATION_H

/// TDCVerification: contract inference (fill missing dimensions with
/// conservative defaults) and post-compilation verification (static and
/// dynamic) that programmer-specified TDC contracts are satisfied.
///
/// Uses the canonical Ordering, Placement, TDCEdgeSpec, and TDCPathSpec types
/// from Contract.h, and the ConstraintSet / evaluateSymbolicExpr from
/// ContractConstraintTranslator.h.

#include "loom/SystemCompiler/BufferAllocator.h"
#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/ContractConstraintTranslator.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Dimension origin tracking
//===----------------------------------------------------------------------===//

/// Tracks whether a dimension was explicitly provided by the user, inferred
/// by the contract inferrer, or absent (no value at all).
enum class DimensionOrigin {
  USER_SPECIFIED, ///< Programmer wrote this dimension explicitly
  INFERRED,       ///< Filled in by inference with a conservative default
  ABSENT          ///< No value: neither user-specified nor inferred
};

/// Per-dimension origin tracking for a single TDCEdgeSpec.
struct TDCEdgeSpecOrigin {
  DimensionOrigin ordering = DimensionOrigin::ABSENT;
  DimensionOrigin throughput = DimensionOrigin::ABSENT;
  DimensionOrigin placement = DimensionOrigin::ABSENT;
  DimensionOrigin shape = DimensionOrigin::ABSENT;
};

//===----------------------------------------------------------------------===//
// Contract inference
//===----------------------------------------------------------------------===//

/// Result of contract inference: the (possibly modified) specs and their
/// origin records.
struct InferenceResult {
  std::vector<TDCEdgeSpec> edgeSpecs;
  std::vector<TDCEdgeSpecOrigin> origins;
  /// Diagnostics from inference (e.g. missing path edge references).
  std::vector<std::string> errors;

  bool ok() const { return errors.empty(); }
};

/// Infer missing dimensions in a collection of edge specs.
///
/// Conservative defaults per spec section 5:
///   - Missing ordering -> Ordering::FIFO (safest; preserves program order)
///   - Missing throughput -> remains nullopt (no performance floor)
///   - Missing placement -> Placement::AUTO (compiler decides)
///   - Missing shape -> remains nullopt (compiler infers from analysis)
///
/// The inferrer works on copies; the original specs are unchanged.
InferenceResult
inferEdgeContracts(const std::vector<TDCEdgeSpec> &edgeSpecs);

/// Validate that path spec edge references actually exist in the edge specs.
/// Returns a vector of error messages (empty on success).
std::vector<std::string>
validatePathReferences(const std::vector<TDCPathSpec> &pathSpecs,
                       const std::vector<TDCEdgeSpec> &edgeSpecs);

//===----------------------------------------------------------------------===//
// Per-edge and per-path verification results
//===----------------------------------------------------------------------===//

/// Verification result for a single edge contract.
struct TDCEdgeVerificationResult {
  std::string producerKernel;
  std::string consumerKernel;

  // Static verification results
  bool orderingSatisfied = true;
  bool placementSatisfied = true;
  bool shapeSatisfied = true;

  // Dynamic verification results
  bool throughputSatisfied = true;

  /// Achieved throughput (elements/cycle) from dynamic metrics, if available.
  std::optional<double> achievedThroughput;

  /// Diagnostic messages for any failures.
  std::vector<std::string> diagnostics;
};

/// Verification result for a single path contract.
struct TDCPathVerificationResult {
  std::string startProducer;
  std::string startConsumer;
  std::string endProducer;
  std::string endConsumer;

  bool latencySatisfied = true;

  /// Achieved latency (cycles) from dynamic metrics, if available.
  std::optional<int64_t> achievedLatency;

  /// Diagnostic messages for any failures.
  std::vector<std::string> diagnostics;
};

//===----------------------------------------------------------------------===//
// Aggregate verification report
//===----------------------------------------------------------------------===//

struct TDCVerificationReport {
  bool allSatisfied = true;

  std::vector<TDCEdgeVerificationResult> edgeResults;
  std::vector<TDCPathVerificationResult> pathResults;

  /// Summary diagnostics.
  std::vector<std::string> diagnostics;
};

//===----------------------------------------------------------------------===//
// Dynamic metrics interfaces
//===----------------------------------------------------------------------===//

/// Per-edge dynamic simulation metrics.
struct DynamicEdgeMetrics {
  std::string producerKernel;
  std::string consumerKernel;

  /// Sustained throughput in elements/cycle.
  double sustainedThroughput = 0.0;

  /// Number of out-of-order delivery violations observed in simulation.
  int64_t orderingViolationCount = 0;
};

/// Per-path dynamic simulation metrics.
struct DynamicPathMetrics {
  std::string startProducer;
  std::string startConsumer;
  std::string endProducer;
  std::string endConsumer;

  /// Observed end-to-end latency in cycles.
  int64_t observedLatency = 0;
};

//===----------------------------------------------------------------------===//
// Compile-time outputs for static verification
//===----------------------------------------------------------------------===//

/// Per-edge tile dimensions from the tiling engine.
struct EdgeTileDimensions {
  std::string producerKernel;
  std::string consumerKernel;
  std::vector<int64_t> dimensions;
};

/// Per-edge scheduling slot information.
struct EdgeSchedulingSlot {
  std::string producerKernel;
  std::string consumerKernel;
  /// Producer completes at this cycle for each tile.
  std::vector<uint64_t> producerCompletionTimes;
  /// Consumer begins at this cycle for each tile.
  std::vector<uint64_t> consumerStartTimes;
};

//===----------------------------------------------------------------------===//
// Static verification
//===----------------------------------------------------------------------===//

/// Run static verification on compile-time outputs against TDC contracts.
/// Only USER_SPECIFIED dimensions are checked; INFERRED and ABSENT are skipped.
///
/// Checks:
///   - Shape: allocated tile dims must exactly match specified shape
///   - Placement: buffer location must match specified placement
///     (LOCAL_SPM -> {SPM_PRODUCER, SPM_CONSUMER}, SHARED_L2 -> SHARED_L2,
///      EXTERNAL -> EXTERNAL_DRAM)
///   - Ordering: for FIFO edges, producer completion time for tile T must
///     precede consumer start time for tile T
TDCVerificationReport
verifyStatic(const std::vector<TDCEdgeSpec> &edgeSpecs,
             const std::vector<TDCEdgeSpecOrigin> &origins,
             const BufferAllocationPlan &bufferPlan,
             const std::vector<EdgeTileDimensions> &tileDims,
             const std::vector<EdgeSchedulingSlot> &schedSlots,
             const std::map<std::string, int64_t> &params = {});

//===----------------------------------------------------------------------===//
// Dynamic verification
//===----------------------------------------------------------------------===//

/// Run dynamic verification on simulation metrics against TDC contracts.
/// Only USER_SPECIFIED dimensions are checked.
///
/// Checks:
///   - Throughput: achievedThroughput >= specifiedThroughput
///   - Ordering (dynamic): violation count must be zero when FIFO specified
///   - Latency (path): observedLatency <= specifiedLatency
TDCVerificationReport
verifyDynamic(const std::vector<TDCEdgeSpec> &edgeSpecs,
              const std::vector<TDCEdgeSpecOrigin> &origins,
              const std::vector<TDCPathSpec> &pathSpecs,
              const std::vector<DynamicEdgeMetrics> &edgeMetrics,
              const std::vector<DynamicPathMetrics> &pathMetrics,
              const std::map<std::string, int64_t> &params);

//===----------------------------------------------------------------------===//
// Top-level verification entry point
//===----------------------------------------------------------------------===//

/// Single entry point for full TDC verification. Runs static verification
/// unconditionally and dynamic verification when dynamic metrics are provided.
///
/// \param edgeSpecs   User-specified edge contracts.
/// \param origins     Origin tracking (which dimensions to check).
/// \param pathSpecs   User-specified path contracts.
/// \param bufferPlan  Buffer allocation decisions from compilation.
/// \param tileDims    Tile dimensions from the tiling engine.
/// \param schedSlots  Scheduling slot information.
/// \param edgeMetrics Optional dynamic simulation metrics for edges.
/// \param pathMetrics Optional dynamic simulation metrics for paths.
/// \param params      Parameter map for resolving symbolic expressions.
TDCVerificationReport
verifyContracts(const std::vector<TDCEdgeSpec> &edgeSpecs,
                const std::vector<TDCEdgeSpecOrigin> &origins,
                const std::vector<TDCPathSpec> &pathSpecs,
                const BufferAllocationPlan &bufferPlan,
                const std::vector<EdgeTileDimensions> &tileDims,
                const std::vector<EdgeSchedulingSlot> &schedSlots,
                const std::optional<std::vector<DynamicEdgeMetrics>> &edgeMetrics,
                const std::optional<std::vector<DynamicPathMetrics>> &pathMetrics,
                const std::map<std::string, int64_t> &params);

//===----------------------------------------------------------------------===//
// Legacy structural verification API (retained for backward compatibility)
//===----------------------------------------------------------------------===//

/// A single verification diagnostic.
struct TDCDiagnostic {
  enum class Severity { Warning, Error };
  Severity severity;
  std::string message;
};

/// Legacy aggregate verification result for structural checks only.
struct TDCVerificationResult {
  bool valid = true;
  std::vector<TDCDiagnostic> diagnostics;
  void addWarning(const std::string &msg);
  void addError(const std::string &msg);
};

/// Verify a single TDCEdgeSpec for structural correctness.
TDCVerificationResult verifyEdgeSpec(const TDCEdgeSpec &spec);

/// Verify a single TDCPathSpec for structural correctness.
TDCVerificationResult verifyPathSpec(const TDCPathSpec &spec);

/// Verify a collection of edge and path specs (structural only).
/// This overload with the 2-parameter signature is the legacy structural
/// verification API, preserved for backward compatibility.
TDCVerificationResult
verifyContractsStructural(const std::vector<TDCEdgeSpec> &edges,
                          const std::vector<TDCPathSpec> &paths);

} // namespace loom

#endif
