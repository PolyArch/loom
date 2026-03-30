#ifndef LOOM_SYSTEMCOMPILER_CONTRACTCONSTRAINTTRANSLATOR_H
#define LOOM_SYSTEMCOMPILER_CONTRACTCONSTRAINTTRANSLATOR_H

/// ContractConstraintTranslator: translates TDC edge/path contracts into
/// typed solver constraints for downstream compilation (L1CoreAssigner,
/// BufferAllocator, tiling engine).
///
/// Uses the canonical Ordering, Placement, TDCEdgeSpec, and TDCPathSpec types
/// from Contract.h. Produces a ConstraintSet of typed constraints and
/// per-edge pruning masks for search space reduction.

#include "loom/SystemCompiler/Contract.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Typed constraint representations for solver backends
//===----------------------------------------------------------------------===//

/// Scheduling precedence: producer must complete tile T before consumer
/// begins tile T (per-tile FIFO ordering).
struct SchedulingConstraint {
  std::string producer;
  std::string consumer;
};

/// Minimum sustained throughput inequality on an edge (elements-per-cycle
/// lower bound). The minRate is a concrete numeric value resolved from a
/// symbolic expression via evaluateSymbolicExpr.
struct RateConstraint {
  std::string edgeProducer;
  std::string edgeConsumer;
  int64_t minRate = 0;
};

/// Buffer placement restriction to a specific memory level.
enum class MemoryLevel {
  LOCAL_SPM,
  SHARED_L2,
  EXTERNAL
};

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

/// Critical-path upper bound from a start edge to an end edge (cycle count).
struct PathLatencyConstraint {
  std::string startProducer;
  std::string startConsumer;
  std::string endProducer;
  std::string endConsumer;
  int64_t maxCycles = 0;
};

//===----------------------------------------------------------------------===//
// ConstraintSet: aggregate of all typed constraints
//===----------------------------------------------------------------------===//

struct ConstraintSet {
  std::vector<SchedulingConstraint> scheduling;
  std::vector<RateConstraint> rate;
  std::vector<MemoryConstraint> memory;
  std::vector<TilingConstraint> tiling;
  std::vector<PathLatencyConstraint> pathLatency;

  /// Diagnostic messages for statically detected issues (e.g. infeasible
  /// constraints). These are warnings -- the ConstraintSet is still populated
  /// so downstream solvers receive the complete picture.
  std::vector<std::string> diagnostics;

  bool empty() const {
    return scheduling.empty() && rate.empty() && memory.empty() &&
           tiling.empty() && pathLatency.empty();
  }
};

//===----------------------------------------------------------------------===//
// Pruning mask: per-edge bitfield indicating which search axes are frozen
//===----------------------------------------------------------------------===//

/// Bit positions in the pruning mask.
enum PruningBit : uint8_t {
  PRUNING_ORDERING_LOCKED = 0,    // Bit 0: ordering specified (FIFO)
  PRUNING_THROUGHPUT_FLOOR = 1,   // Bit 1: throughput floor set
  PRUNING_PLACEMENT_LOCKED = 2,   // Bit 2: placement specified
  PRUNING_SHAPE_LOCKED = 3        // Bit 3: shape specified
};

/// Key for identifying an edge: (producer, consumer).
struct EdgeKey {
  std::string producer;
  std::string consumer;

  bool operator<(const EdgeKey &rhs) const {
    if (producer != rhs.producer)
      return producer < rhs.producer;
    return consumer < rhs.consumer;
  }

  bool operator==(const EdgeKey &rhs) const {
    return producer == rhs.producer && consumer == rhs.consumer;
  }
};

//===----------------------------------------------------------------------===//
// Symbolic expression evaluator
//===----------------------------------------------------------------------===//

/// Result of evaluating a symbolic expression. On success, value is set.
/// On failure, error contains a diagnostic message.
struct SymbolicEvalResult {
  std::optional<int64_t> value;
  std::string error;

  bool ok() const { return value.has_value(); }
};

/// Evaluate a symbolic arithmetic expression with a parameter map.
/// Supported: integer literals, named parameters, binary operators (+, -, *, /),
/// and parenthesized sub-expressions. Integer division truncates toward zero.
///
/// Returns an error result if unknown variables are encountered or if the
/// expression is malformed.
SymbolicEvalResult
evaluateSymbolicExpr(const std::string &expr,
                     const std::map<std::string, int64_t> &params);

//===----------------------------------------------------------------------===//
// ContractConstraintTranslator
//===----------------------------------------------------------------------===//

/// Translates TDC edge/path specs into a typed ConstraintSet consumed by
/// the bilevel compiler (L1CoreAssigner, BufferAllocator, tiling engine).
class ContractConstraintTranslator {
public:
  /// Construct the translator with an optional parameter map for resolving
  /// symbolic expressions in throughput, shape, and latency dimensions.
  explicit ContractConstraintTranslator(
      std::map<std::string, int64_t> params = {});

  /// Translate a collection of edge and path specs into a typed ConstraintSet.
  ConstraintSet
  translate(const std::vector<TDCEdgeSpec> &edges,
            const std::vector<TDCPathSpec> &paths) const;

  /// Compute per-edge pruning masks indicating which search axes are frozen.
  /// The map key is (producer, consumer) for each edge spec.
  std::map<EdgeKey, uint8_t>
  computePruningMasks(const std::vector<TDCEdgeSpec> &edges) const;

  /// Compute pruning mask for a single edge spec.
  uint8_t computePruningMask(const TDCEdgeSpec &edgeSpec) const;

private:
  std::map<std::string, int64_t> params_;

  /// Translate a single edge spec into constraints appended to the set.
  void translateEdge(const TDCEdgeSpec &edge, ConstraintSet &out) const;

  /// Translate a single path spec into constraints appended to the set.
  void translatePath(const TDCPathSpec &path, ConstraintSet &out) const;

  /// Resolve a shape expression "[dim0, dim1, ...]" into concrete dimensions.
  /// Returns empty on parse failure.
  std::vector<int64_t> resolveShape(const std::string &shapeExpr) const;
};

//===----------------------------------------------------------------------===//
// Legacy compatibility: flat TranslatedConstraint for migration
//===----------------------------------------------------------------------===//

/// A single translated constraint produced from a TDC contract dimension.
/// Retained for backward compatibility during migration.
struct TranslatedConstraint {
  std::string label;
  std::string dimension;
  std::string expression;
  std::string enumValue;
};

/// Translate a single TDCEdgeSpec into a vector of legacy solver constraints.
std::vector<TranslatedConstraint>
translateEdgeConstraints(const TDCEdgeSpec &edgeSpec);

/// Translate a single TDCPathSpec into a vector of legacy solver constraints.
std::vector<TranslatedConstraint>
translatePathConstraints(const TDCPathSpec &pathSpec);

/// Translate a collection of edge and path specs into a flat constraint list.
std::vector<TranslatedConstraint>
translateAllConstraints(const std::vector<TDCEdgeSpec> &edges,
                        const std::vector<TDCPathSpec> &paths);

/// Convert a legacy ContractSpec into a TDCEdgeSpec.
TDCEdgeSpec contractSpecToEdgeSpec(const ContractSpec &legacy);

} // namespace loom

#endif
