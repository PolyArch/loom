#ifndef LOOM_SYSTEMCOMPILER_CONTRACTCONSTRAINTTRANSLATOR_H
#define LOOM_SYSTEMCOMPILER_CONTRACTCONSTRAINTTRANSLATOR_H

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// TDC Specification Types (G1 bridge)
//===----------------------------------------------------------------------===//

/// Ordering discipline for a TDC edge.
enum class TDCOrdering { FIFO, UNORDERED };

/// Buffer placement target for a TDC edge.
enum class TDCPlacement { AUTO, LOCAL_SPM, SHARED_L2, EXTERNAL };

/// TDC specification for a single inter-kernel communication edge.
/// Each optional dimension, when absent, leaves the compiler free to explore.
struct TDCEdgeSpec {
  std::string producerKernel;
  std::string consumerKernel;

  std::optional<TDCOrdering> ordering;
  std::optional<std::string> throughput; // symbolic or numeric expression
  std::optional<TDCPlacement> placement;
  std::optional<std::string> shape; // e.g. "[128, 256]" or "[batch_size, dim]"
};

/// TDC specification for a critical-path latency bound across multiple edges.
struct TDCPathSpec {
  std::string startProducer;
  std::string startConsumer;
  std::string endProducer;
  std::string endConsumer;
  std::string latency; // symbolic or numeric expression
};

//===----------------------------------------------------------------------===//
// Constraint Types
//===----------------------------------------------------------------------===//

/// FIFO ordering dependency between a producer and consumer kernel.
struct SchedulingConstraint {
  std::string producer;
  std::string consumer;
};

/// Minimum sustained throughput inequality on an edge (elements-per-cycle).
struct RateConstraint {
  std::string edgeProducer;
  std::string edgeConsumer;
  int64_t minRate = 0;
};

/// Memory-level placement restriction for an edge buffer.
enum class MemoryLevel { LOCAL_SPM, SHARED_L2, EXTERNAL };

struct MemoryConstraint {
  std::string edgeProducer;
  std::string edgeConsumer;
  MemoryLevel level = MemoryLevel::LOCAL_SPM;
};

/// Fixed tile dimensions for an edge, disabling retiling freedom.
struct TilingConstraint {
  std::string edgeProducer;
  std::string edgeConsumer;
  std::vector<int64_t> dimensions;
};

/// Critical-path upper bound from a start edge to an end edge.
struct PathLatencyConstraint {
  std::string startProducer;
  std::string startConsumer;
  std::string endProducer;
  std::string endConsumer;
  int64_t maxCycles = 0;
};

/// Aggregated constraint set produced by the translator.
struct ConstraintSet {
  std::vector<SchedulingConstraint> schedulingConstraints;
  std::vector<RateConstraint> rateConstraints;
  std::vector<MemoryConstraint> memoryConstraints;
  std::vector<TilingConstraint> tilingConstraints;
  std::vector<PathLatencyConstraint> pathLatencyConstraints;

  bool empty() const {
    return schedulingConstraints.empty() && rateConstraints.empty() &&
           memoryConstraints.empty() && tilingConstraints.empty() &&
           pathLatencyConstraints.empty();
  }

  size_t totalConstraintCount() const {
    return schedulingConstraints.size() + rateConstraints.size() +
           memoryConstraints.size() + tilingConstraints.size() +
           pathLatencyConstraints.size();
  }
};

//===----------------------------------------------------------------------===//
// Pruning Mask
//===----------------------------------------------------------------------===//

/// Per-edge bitfield indicating which search axes are frozen by TDC specs.
///   Bit 0: ordering locked (FIFO specified) -- skip reordering transforms.
///   Bit 1: throughput floor set -- skip configs below rate.
///   Bit 2: placement locked -- skip alternative memory level options.
///   Bit 3: shape locked -- skip tiling exploration for this edge.
struct PruningMask {
  std::string edgeProducer;
  std::string edgeConsumer;
  uint8_t mask = 0;

  static constexpr uint8_t ORDERING_LOCKED = 1u << 0;
  static constexpr uint8_t THROUGHPUT_LOCKED = 1u << 1;
  static constexpr uint8_t PLACEMENT_LOCKED = 1u << 2;
  static constexpr uint8_t SHAPE_LOCKED = 1u << 3;

  bool isOrderingLocked() const { return (mask & ORDERING_LOCKED) != 0; }
  bool isThroughputLocked() const { return (mask & THROUGHPUT_LOCKED) != 0; }
  bool isPlacementLocked() const { return (mask & PLACEMENT_LOCKED) != 0; }
  bool isShapeLocked() const { return (mask & SHAPE_LOCKED) != 0; }
};

//===----------------------------------------------------------------------===//
// Symbolic Expression Evaluator
//===----------------------------------------------------------------------===//

/// Parameter map for resolving symbolic variables in expressions.
using ParameterMap = std::map<std::string, int64_t>;

/// Result of evaluating a symbolic expression. On success, `value` holds the
/// numeric result and `error` is empty. On failure, `error` holds a diagnostic.
struct EvalResult {
  int64_t value = 0;
  std::string error;

  bool ok() const { return error.empty(); }
};

/// Evaluate a symbolic expression string with the given parameter map.
/// Supports: integer literals, named parameters, binary operators (+,-,*,/),
/// and parenthesized sub-expressions. Division is integer (truncating).
EvalResult evaluateSymbolicExpr(const std::string &expr,
                                const ParameterMap &params);

/// Parse a shape string like "[128, 256]" or "[batch_size, dim]" into
/// individual dimension expressions.
std::vector<std::string> parseShapeDimensions(const std::string &shapeExpr);

//===----------------------------------------------------------------------===//
// ContractConstraintTranslator
//===----------------------------------------------------------------------===//

/// Diagnostic from the constraint translator (warnings about potentially
/// infeasible constraints, unknown variables, etc.).
struct TranslatorDiagnostic {
  enum Severity { WARNING, ERROR };
  Severity severity = WARNING;
  std::string message;
};

/// Translates TDCEdgeSpec and TDCPathSpec into a ConstraintSet and pruning
/// masks consumed by the bilevel compiler.
class ContractConstraintTranslator {
public:
  /// Translate TDC specs into solver constraints.
  ///
  /// \param edgeSpecs  Per-edge TDC specifications.
  /// \param pathSpecs  Path-level TDC specifications.
  /// \param params     Parameter map for symbolic expression resolution.
  /// \returns          Aggregated constraint set.
  ConstraintSet translate(const std::vector<TDCEdgeSpec> &edgeSpecs,
                          const std::vector<TDCPathSpec> &pathSpecs,
                          const ParameterMap &params);

  /// Compute per-edge pruning masks from the TDC edge specs.
  /// The mask for each edge indicates which search axes are frozen.
  std::vector<PruningMask>
  computePruningMasks(const std::vector<TDCEdgeSpec> &edgeSpecs);

  /// Access diagnostics emitted during the last translate() call.
  const std::vector<TranslatorDiagnostic> &getDiagnostics() const {
    return diagnostics_;
  }

  /// Clear accumulated diagnostics.
  void clearDiagnostics() { diagnostics_.clear(); }

private:
  void emitDiag(TranslatorDiagnostic::Severity sev, const std::string &msg);

  void translateEdgeSpec(const TDCEdgeSpec &spec, const ParameterMap &params,
                         ConstraintSet &out);
  void translatePathSpec(const TDCPathSpec &spec, const ParameterMap &params,
                         ConstraintSet &out);

  std::vector<TranslatorDiagnostic> diagnostics_;
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_CONTRACTCONSTRAINTTRANSLATOR_H
