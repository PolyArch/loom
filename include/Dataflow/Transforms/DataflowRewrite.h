#ifndef LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_H
#define LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <optional>

namespace dataflow {

/// The closed set of typed Dataflow-only rewrites implemented here. A rewrite
/// kind is a semantic identity, not a pass name or a free string: one enum
/// value owns one source pattern, one complete set of legality preconditions,
/// and one deterministic result construction. Adding a kind means adding a
/// proven observable-equivalence contract, never a new dispatch mechanism.
enum class DataflowRewriteKind {
  /// Eliminate an exact `unpack(pack(vector))` or `pack(unpack(bits))`.
  PackUnpackRoundTripEliminate,
  /// Eliminate an exact `serialize(parallelize(data, phase))`. The reverse
  /// composition is not an identity and is never matched.
  ParallelizeSerializeRoundTripEliminate,
  /// Replace an exactly foldable pure Compute actor over same-ctrl
  /// `dataflow.constant` operands with one `dataflow.constant` on that ctrl.
  ActivationPreservingConstantFold,
};

/// Applies exactly one selected rewrite kind. Graph changes are made on a
/// private module candidate and published only after native verification and
/// whole-program `validateFinalizedProgram` succeed, so a rejected rewrite
/// leaves the pass-visible module untouched.
///
/// This optional rewrite surface is separate from mandatory canonical
/// finalization: the pass runs no canonicalizer and no finalization pipeline.
std::unique_ptr<::mlir::Pass>
createDataflowRewritePass(DataflowRewriteKind kind);

/// Applies one typed rewrite to a private clone and finalizes the complete
/// result through the sole Canonical Dataflow finalizer. A no-op returns an
/// empty optional; a changed result has a distinct immutable identity.
llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeDataflowRewrite(const CanonicalDataflowArtifact &parent,
                           DataflowRewriteKind kind);

/// Registers `--dataflow-rewrite` with the global pass registry so developer
/// tools can drive it as `--dataflow-rewrite=kind=<value>`.
void registerDataflowTransformsPasses();

} // namespace dataflow

#endif // LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_H
