#ifndef LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H
#define LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Frontend/Raising/Passes.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <system_error>
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
/// Program before the mechanical Dataflow boundary. The canonical index width
/// applies to every dynamic LLVM GEP in the selected callable. An absent
/// decision never selects a default; a selected region that still contains
/// such a choice fails canonical publication.
struct WholeCallableSpatialOwnershipOptions final {
  lowering::CanonicalDataflowLoweringOptions lowering;
  std::optional<raising::FMulAddExecutionShape> fmuladdExecutionShape;
  std::optional<unsigned> canonicalIndexWidth;
};

/// Typed decisions selected for one operation-owned Spatial candidate. An
/// explicit FMA execution shape and canonical index width are materialized
/// only within the selected operation in the child Structured Program. Wider
/// LLVM GEP indices are narrowed only when their complete value domain is
/// proven to fit that signed width; absence never selects an ambient or
/// Fabric-derived default.
struct OperationSpatialOwnershipOptions final {
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

enum class SpatialOwnershipScopeKind {
  WholeCallable,
  Operation,
};

/// One finite ownership search coordinate in the exact parent candidate.
/// Scope kind selects the materialization rule; the StructuredEntityRef is the
/// only identity of the selected program structure.
struct SpatialOwnershipScope final {
  SpatialOwnershipScopeKind kind;
  StructuredEntityRef selection;

  friend bool operator==(const SpatialOwnershipScope &lhs,
                         const SpatialOwnershipScope &rhs) {
    return lhs.kind == rhs.kind && lhs.selection == rhs.selection;
  }
};

/// Enumerates every ownership scope in the parent candidate's canonical
/// operation order. Each returned scope remains independent: callers derive
/// and explore its decision domain separately rather than constructing one
/// cross-scope Cartesian product.
llvm::Expected<std::vector<SpatialOwnershipScope>>
enumerateSpatialOwnershipScopes(const StructuredProgramCandidate &parent);

/// Enumerates whole-callable ownership scopes in the exact parent candidate's
/// canonical operation order. Only callables satisfying the complete
/// whole-callable structural contract are returned; this is a generator
/// domain, not a QoR choice or an implicit ownership decision.
llvm::Expected<std::vector<StructuredEntityRef>>
enumerateWholeCallableSpatialOwnershipScopes(
    const StructuredProgramCandidate &parent);

/// Enumerates ownership-free structured scopes in the exact parent
/// candidate's canonical operation order. These are search coordinates, not
/// accepted or ranked candidates: applying a concrete typed decision may
/// still fail legality, materialization, or exact-Fabric admission.
llvm::Expected<std::vector<StructuredEntityRef>>
enumerateOperationSpatialOwnershipScopes(
    const StructuredProgramCandidate &parent);

/// Derives the finite typed decision domain for one exact ownership scope.
/// Canonical address widths come from the closed Fabric index-width schema;
/// exact concrete target admission remains part of candidate materialization.
/// Fmuladd alternatives are exposed only when the selected scope contains the
/// corresponding unresolved LLVM operation. The result is deterministic and
/// performs no ranking or implicit default selection.
llvm::Expected<std::vector<SpatialOwnershipDecisionPoint>>
enumerateSpatialOwnershipDecisionDomain(
    const StructuredProgramCandidate &parent, const StructuredEntityRef &scope);

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

/// Materializes the explicit whole-callable SpatialCore ownership choice for
/// one exact void, single-block LLVM callable. Its original body moves into an
/// ordered thread while the same LLVM callable remains the ABI authority and
/// launches that thread. Fabric is used only for hard-negative actor-capability
/// pruning; this function performs no Mapping or QoR choice.
llvm::Expected<MaterializedOwnershipCandidate>
materializeWholeCallableSpatialOwnership(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &callable,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const WholeCallableSpatialOwnershipOptions &options = {});

/// Materializes the explicit SpatialCore ownership choice for one exact
/// structured operation inside an ordinary LLVM callable. The operation must
/// sit outside any dataflow.thread, dataflow.graph, or loom.spatial_region,
/// own at least one region, and have no SSA result used outside itself. Every
/// external SSA live-in becomes an explicit input of one new private rank-zero
/// dataflow.thread holding one loom.spatial_region; the original LLVM callable
/// remains the ABI authority and launches that thread at the operation's exact
/// position. Fabric is used only for hard-negative actor-capability pruning;
/// this function performs no Mapping or QoR choice.
llvm::Expected<MaterializedOwnershipCandidate>
materializeOperationSpatialOwnership(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &operation,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const OperationSpatialOwnershipOptions &options = {});

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_OWNERSHIPCANDIDATEGENERATOR_H
