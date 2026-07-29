#ifndef LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H
#define LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Frontend/Raising/Passes.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

namespace loom::frontend {

/// One ordinary child Structured Program and its mechanically derived D0
/// projection. The pair is returned only after both artifact owners finalize
/// successfully and the exact Fabric admits every canonical actor.
struct MaterializedOwnershipCandidate final {
  StructuredProgramCandidate structuredProgram;
  dataflow::CanonicalDataflowArtifact canonicalDataflow;
};

/// Typed decisions that must be materialized in the selected Structured
/// Program before the mechanical Dataflow boundary. An absent decision never
/// selects a default; a selected region that still contains such a choice
/// fails canonical publication.
struct SpatialOwnershipOptions final {
  lowering::CanonicalDataflowLoweringOptions lowering;
  std::optional<raising::FMulAddExecutionShape> fmuladdExecutionShape;
  std::optional<unsigned> canonicalIndexWidth;
};

/// One finite, scope-local ownership decision point. This is an ephemeral
/// generator value rather than a selected candidate or persistent program
/// record. Applying it may still fail semantic materialization or exact-Fabric
/// admission.
struct SpatialOwnershipDecisionPoint final {
  std::optional<raising::FMulAddExecutionShape> fmuladdExecutionShape;
  std::optional<unsigned> canonicalIndexWidth;

  friend bool operator==(const SpatialOwnershipDecisionPoint &lhs,
                         const SpatialOwnershipDecisionPoint &rhs) {
    return lhs.fmuladdExecutionShape == rhs.fmuladdExecutionShape &&
           lhs.canonicalIndexWidth == rhs.canonicalIndexWidth;
  }
};

enum class SpatialOwnershipCandidateRejectionKind {
  NonFinalizable,
  ExactFabricInadmissible,
};

/// An expected negative result for one otherwise well-formed scope-local
/// candidate attempt. Other errors remain invocation or implementation
/// failures and must not be silently treated as search-space pruning.
class SpatialOwnershipCandidateRejection final
    : public llvm::ErrorInfo<SpatialOwnershipCandidateRejection> {
public:
  static char ID;

  SpatialOwnershipCandidateRejection(
      SpatialOwnershipCandidateRejectionKind kind, std::string message)
      : kind_(kind), message_(std::move(message)) {}

  SpatialOwnershipCandidateRejectionKind kind() const { return kind_; }
  std::string message() const override { return message_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SpatialOwnershipCandidateRejectionKind kind_;
  std::string message_;
};

/// One finite ownership search coordinate in the exact parent candidate.
/// The StructuredEntityRef is the only identity of the selected program
/// structure; its resolved operation type mechanically determines how the ABI
/// envelope is rewritten.
struct SpatialOwnershipScope final {
  StructuredEntityRef selection;

  friend bool operator==(const SpatialOwnershipScope &lhs,
                         const SpatialOwnershipScope &rhs) {
    return lhs.selection == rhs.selection;
  }
};

/// One definition-level ownership scope that belongs to the finite domain but
/// cannot be materialized by the current lowering semantics. Declarations and
/// operations that are not ownership scopes are omitted rather than recorded
/// as rejected candidates.
struct RejectedSpatialOwnershipScope final {
  SpatialOwnershipScope scope;
  std::string message;

  friend bool operator==(const RejectedSpatialOwnershipScope &lhs,
                         const RejectedSpatialOwnershipScope &rhs) {
    return lhs.scope == rhs.scope && lhs.message == rhs.message;
  }
};

using SpatialOwnershipScopeDomainEntry =
    std::variant<SpatialOwnershipScope, RejectedSpatialOwnershipScope>;

/// The complete finite scope domain and its mechanically derived ownership
/// hierarchy. Parent ordinals are local to this exact domain and therefore do
/// not become Structured Program identity or persistent candidate lineage.
class SpatialOwnershipScopeDomain final {
public:
  using const_iterator =
      std::vector<SpatialOwnershipScopeDomainEntry>::const_iterator;

  std::size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }
  const SpatialOwnershipScopeDomainEntry &operator[](std::size_t index) const {
    return entries_[index];
  }
  const_iterator begin() const { return entries_.begin(); }
  const_iterator end() const { return entries_.end(); }

  std::optional<std::uint64_t>
  parentScopeOrdinal(std::size_t scopeOrdinal) const {
    return parentScopeOrdinals_[scopeOrdinal];
  }

private:
  std::vector<SpatialOwnershipScopeDomainEntry> entries_;
  std::vector<std::optional<std::uint64_t>> parentScopeOrdinals_;

  friend llvm::Expected<SpatialOwnershipScopeDomain>
  enumerateSpatialOwnershipScopeDomain(const StructuredProgramCandidate &);
};

/// A private clone in which one exact ownership scope has had all selected
/// semantic decisions materialized, but no execution owner has yet replaced
/// the selected operation. `liveIns` and `liveOuts` are the exact ordered
/// selected boundary used by ownership materialization for both callable and
/// nested-operation scopes. The parent candidate owns the MLIRContext and must
/// outlive this ephemeral projection.
struct PreparedSpatialOwnershipSelection final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::Operation *operation = nullptr;
  std::vector<mlir::Value> liveIns;
  std::vector<mlir::Value> liveOuts;
};

/// Enumerates the complete finite ownership scope domain in the parent
/// candidate's canonical operation order. Expected definition-level failures
/// remain typed entries; non-scope operations and external declarations are
/// not candidate attempts. Each accepted scope remains independent: callers
/// derive and explore its decision domain separately rather than constructing
/// one cross-scope Cartesian product.
llvm::Expected<SpatialOwnershipScopeDomain>
enumerateSpatialOwnershipScopeDomain(const StructuredProgramCandidate &parent);

/// Derives the finite typed decision domain for one exact ownership scope.
/// Canonical address widths come from the closed Fabric index-width schema;
/// exact concrete target admission remains part of candidate materialization.
/// Fmuladd alternatives are exposed only when the selected scope contains the
/// corresponding unresolved LLVM operation. The result is deterministic and
/// performs no ranking or implicit default selection.
llvm::Expected<std::vector<SpatialOwnershipDecisionPoint>>
enumerateSpatialOwnershipDecisionDomain(
    const StructuredProgramCandidate &parent, const StructuredEntityRef &scope);

/// Clone one exact parent and materialize one point from the selected scope's
/// typed decision domain without changing ownership. Candidate publication and
/// independent execution oracles both consume this single implementation.
llvm::Expected<PreparedSpatialOwnershipSelection>
prepareSpatialOwnershipSelection(const StructuredProgramCandidate &parent,
                                 const SpatialOwnershipScope &scope,
                                 const SpatialOwnershipDecisionPoint &decision);

/// Materializes one explicit point from one exact scope-local decision domain.
/// This performs semantic finalization and exact-Fabric hard pruning, but no
/// ranking, fallback, implicit decision, or Mapping.
llvm::Expected<MaterializedOwnershipCandidate>
materializeSpatialOwnershipDecision(
    const StructuredProgramCandidate &parent,
    const SpatialOwnershipScope &scope,
    const SpatialOwnershipDecisionPoint &decision,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const lowering::CanonicalDataflowLoweringOptions &lowering = {});

/// Materializes one exact dependency-closed Spatial ownership scope. A
/// callable selection retains the callable as the LLVM ABI authority; a
/// nested structured selection replaces that operation at its exact position.
/// Both forms create one private rank-zero thread containing exactly one
/// loom.spatial_region. Value live-outs cross the thread boundary through
/// caller-owned result storage, while dataflow.thread retains no data results.
/// Fabric is used only for hard-negative actor-capability pruning; this
/// function performs no Mapping or QoR choice.
llvm::Expected<MaterializedOwnershipCandidate>
materializeSpatialOwnership(const StructuredProgramCandidate &parent,
                            const StructuredEntityRef &selection,
                            const ::loom::fabric::FinalizedFabricRoot &fabric,
                            const SpatialOwnershipOptions &options = {});

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H
