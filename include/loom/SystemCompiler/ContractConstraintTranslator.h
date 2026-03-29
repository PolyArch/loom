#ifndef LOOM_SYSTEMCOMPILER_CONTRACTCONSTRAINTTRANSLATOR_H
#define LOOM_SYSTEMCOMPILER_CONTRACTCONSTRAINTTRANSLATOR_H

/// ContractConstraintTranslator: translates TDC edge/path contracts into
/// solver-friendly constraint representations for downstream compilation.
///
/// Uses the canonical Ordering, Placement, TDCEdgeSpec, and TDCPathSpec types
/// from Contract.h (no duplicate enum definitions).

#include "loom/SystemCompiler/Contract.h"

#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Constraint representation for solver backends
//===----------------------------------------------------------------------===//

/// A single translated constraint produced from a TDC contract dimension.
struct TranslatedConstraint {
  /// Human-readable label for debugging (e.g., "ordering:FIFO:matmul->softmax").
  std::string label;

  /// The contract dimension that produced this constraint.
  /// One of: "ordering", "throughput", "placement", "shape", "latency".
  std::string dimension;

  /// Symbolic expression for the constraint bound (e.g., "batch * hidden / 1000").
  /// Empty for enumerated constraints (ordering, placement).
  std::string expression;

  /// For enumerated dimensions, the enum value as a string.
  /// E.g., "FIFO" for ordering, "SHARED_L2" for placement.
  std::string enumValue;
};

//===----------------------------------------------------------------------===//
// Translator API
//===----------------------------------------------------------------------===//

/// Translate a single TDCEdgeSpec into a vector of solver constraints.
/// Only dimensions that are set (has_value) produce constraints.
std::vector<TranslatedConstraint>
translateEdgeConstraints(const TDCEdgeSpec &edgeSpec);

/// Translate a single TDCPathSpec into a vector of solver constraints.
std::vector<TranslatedConstraint>
translatePathConstraints(const TDCPathSpec &pathSpec);

/// Translate a collection of edge and path specs into a flat constraint list.
std::vector<TranslatedConstraint>
translateAllConstraints(const std::vector<TDCEdgeSpec> &edges,
                        const std::vector<TDCPathSpec> &paths);

/// Convert a legacy ContractSpec into a TDCEdgeSpec, extracting only the
/// dimensions that map to the TDC model (ordering, placement).
TDCEdgeSpec contractSpecToEdgeSpec(const ContractSpec &legacy);

} // namespace loom

#endif
