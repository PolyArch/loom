#ifndef LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_H
#define LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <variant>
#include <vector>

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

/// Split one fixed-vector elementwise actor into equal leading-dimension
/// chunks. The ActorRef belongs to the exact parent artifact; it is invocation
/// lineage, not a second persistent actor identity.
struct ElementwiseVectorChunkRewrite final {
  ActorRef actor;
  std::int64_t leadingBlocksPerChunk;

  friend bool operator==(const ElementwiseVectorChunkRewrite &lhs,
                         const ElementwiseVectorChunkRewrite &rhs) {
    return lhs.actor == rhs.actor &&
           lhs.leadingBlocksPerChunk == rhs.leadingBlocksPerChunk;
  }
};

/// Scalarize one fixed-vector elementwise actor at every row-major position.
struct ElementwiseVectorScalarizeRewrite final {
  ActorRef actor;

  friend bool operator==(const ElementwiseVectorScalarizeRewrite &lhs,
                         const ElementwiseVectorScalarizeRewrite &rhs) {
    return lhs.actor == rhs.actor;
  }
};

using DataflowRewriteDecision =
    std::variant<DataflowRewriteKind, ElementwiseVectorChunkRewrite,
                 ElementwiseVectorScalarizeRewrite>;

llvm::ArrayRef<std::uint8_t> dataflowRewriteDecisionSchemaBytes();
llvm::Expected<std::vector<std::uint8_t>>
encodeDataflowRewriteDecision(const DataflowRewriteDecision &decision);
llvm::Expected<DataflowRewriteDecision>
adoptDataflowRewriteDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes);

/// Stable total order used only for deterministic invocation lineage.
bool dataflowRewriteDecisionLess(const DataflowRewriteDecision &lhs,
                                 const DataflowRewriteDecision &rhs);

/// Enumerates the finite decomposition domain for one exact actor. An empty
/// result means that the actor is outside the closed pure fixed-vector
/// elementwise domain.
llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateElementwiseVectorDecompositionDecisions(
    const CanonicalDataflowArtifact &parent, ActorRef actor);

/// Number of narrow compute actors materialized by one decision. Generators
/// charge this exact amount against their resolved semantic work limit before
/// constructing the candidate.
llvm::Expected<std::uint64_t>
dataflowRewriteExpansionCost(const CanonicalDataflowArtifact &parent,
                             const DataflowRewriteDecision &decision);

/// Applies exactly one selected rewrite kind. Graph changes are made on a
/// private module candidate and published only after native verification and
/// whole-program `validateFinalizedProgram` succeed, so a rejected rewrite
/// leaves the pass-visible module untouched.
///
/// This optional rewrite surface is separate from mandatory canonical
/// finalization: the pass runs no canonicalizer and no finalization pipeline.
llvm::Expected<std::unique_ptr<::mlir::Pass>>
createDataflowRewritePass(DataflowRewriteKind kind);

/// Applies one typed rewrite to a private clone and finalizes the complete
/// result through the sole Canonical Dataflow finalizer. A no-op returns an
/// empty optional; a changed result has a distinct immutable identity.
llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeDataflowRewrite(const CanonicalDataflowArtifact &parent,
                           DataflowRewriteKind kind);

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeDataflowRewrite(const CanonicalDataflowArtifact &parent,
                           const DataflowRewriteDecision &decision);

/// Registers `--dataflow-rewrite` with the global pass registry so developer
/// tools can drive it as `--dataflow-rewrite=kind=<value>`.
void registerDataflowTransformsPasses();

} // namespace dataflow

#endif // LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_H
