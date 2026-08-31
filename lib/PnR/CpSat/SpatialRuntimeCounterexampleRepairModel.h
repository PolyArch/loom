#ifndef LOOM_LIB_PNR_CPSAT_SPATIALRUNTIMECOUNTEREXAMPLEREPAIRMODEL_H
#define LOOM_LIB_PNR_CPSAT_SPATIALRUNTIMECOUNTEREXAMPLEREPAIRMODEL_H

#include "SpatialBindingRelationModel.h"
#include "SpatialLocalDispositionModel.h"

#include "PnR/FrozenConstraintIndex.h"
#include "PnR/SpatialCandidateState.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr::detail {

/// One canonical finite way to make a runtime-counterexample clause false.
/// The clause and literal ordinals are the only semantic anchors; all dense
/// net, terminal, traversal, and tag values are re-resolved from the frozen
/// constraint index when the branch is encoded or executed.
enum class SpatialRuntimeCounterexampleBreakerKind : std::uint8_t {
  RegisterFifoDisposition,
  TransferAttachment,
  NetTraversal,
  NetTag,
  MappingIdentity,
};

struct SpatialRuntimeCounterexampleBreaker final {
  PnrIndex clauseOrdinal = getInvalidPnrIndex();
  PnrIndex clauseLocalLiteralOrdinal = getInvalidPnrIndex();
  SpatialRuntimeCounterexampleBreakerKind kind =
      SpatialRuntimeCounterexampleBreakerKind::TransferAttachment;
  /// Derived exact replacement for NetTag. It is never persisted and is
  /// ignored for every other kind.
  std::optional<llvm::APInt> physicalTagValue;

  friend bool operator==(const SpatialRuntimeCounterexampleBreaker &lhs,
                         const SpatialRuntimeCounterexampleBreaker &rhs) {
    return lhs.clauseOrdinal == rhs.clauseOrdinal &&
           lhs.clauseLocalLiteralOrdinal ==
               rhs.clauseLocalLiteralOrdinal &&
           lhs.kind == rhs.kind &&
           lhs.physicalTagValue == rhs.physicalTagValue;
  }
};

/// Enumerates branch anchors in canonical literal order. Local-disposition is
/// a distinct branch because selecting register-FIFO makes every route,
/// attachment, and route-tag literal for the net false without inventing a
/// second persistent literal kind.
llvm::Expected<std::vector<SpatialRuntimeCounterexampleBreaker>>
enumerateSpatialRuntimeCounterexampleBreakers(
    const SpatialCandidateState &candidate, PnrIndex clauseOrdinal);

/// Returns the exact frozen literal named by one canonical breaker.
llvm::Expected<const FrozenNoGoodResolvedLiteral *>
resolveSpatialRuntimeCounterexampleBreaker(
    const FrozenSpatialPnrProblem &problem,
    const SpatialRuntimeCounterexampleBreaker &breaker);

/// Adds the finite binding/local-disposition portion of one breaker branch to
/// a transport repair CP model. Internal traversal removal and exact tag-value
/// selection are executed by their ordinary typed Candidate actions after the
/// solver returns; this function never substitutes a constant truth value for
/// either decision.
llvm::Error addSpatialRuntimeCounterexampleBreakerConstraint(
    operations_research::sat::CpModelBuilder &model,
    const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<operations_research::sat::IntVar> bindingVariables,
    llvm::ArrayRef<int> decisionVariables,
    const SpatialLocalDispositionModel &localDispositions,
    const SpatialRuntimeCounterexampleBreaker &breaker);

/// True only when the parent Candidate selects the traversal on the internal
/// RouteTree scope named by a NetTraversal breaker. A source- or sink-local
/// occurrence is completely broken by the CP attachment constraints and must
/// not be passed to the internal endpoint router as a fictitious arc cut.
llvm::Expected<bool> spatialRuntimeTraversalRequiresRouteCut(
    const SpatialCandidateState &candidate,
    const SpatialRuntimeCounterexampleBreaker &breaker);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_CPSAT_SPATIALRUNTIMECOUNTEREXAMPLEREPAIRMODEL_H
