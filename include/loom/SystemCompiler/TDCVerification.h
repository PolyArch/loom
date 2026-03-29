#ifndef LOOM_SYSTEMCOMPILER_TDCVERIFICATION_H
#define LOOM_SYSTEMCOMPILER_TDCVERIFICATION_H

/// TDCVerification: validates TDC edge and path contracts for structural
/// correctness and internal consistency.
///
/// Uses the canonical Ordering, Placement, TDCEdgeSpec, and TDCPathSpec types
/// from Contract.h.

#include "loom/SystemCompiler/Contract.h"

#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Verification result
//===----------------------------------------------------------------------===//

/// A single verification diagnostic.
struct TDCDiagnostic {
  enum class Severity { Warning, Error };

  Severity severity;
  std::string message;
};

/// Aggregate verification result for a set of contracts.
struct TDCVerificationResult {
  bool valid = true;
  std::vector<TDCDiagnostic> diagnostics;

  /// Append a warning.
  void addWarning(const std::string &msg);

  /// Append an error (sets valid = false).
  void addError(const std::string &msg);
};

//===----------------------------------------------------------------------===//
// Verification API
//===----------------------------------------------------------------------===//

/// Verify a single TDCEdgeSpec for structural correctness:
///   - producer and consumer kernel names must be non-empty
///   - dataTypeName must be non-empty
///   - shape expression (if set) must parse successfully via parseShapeExpr
TDCVerificationResult verifyEdgeSpec(const TDCEdgeSpec &spec);

/// Verify a single TDCPathSpec for structural correctness:
///   - all four endpoint kernel names must be non-empty
///   - latency expression must be non-empty
TDCVerificationResult verifyPathSpec(const TDCPathSpec &spec);

/// Verify a collection of edge and path specs, returning a merged result.
TDCVerificationResult
verifyContracts(const std::vector<TDCEdgeSpec> &edges,
                const std::vector<TDCPathSpec> &paths);

} // namespace loom

#endif
