#ifndef LOOM_PNR_CPSAT_SPATIALHANDSHAKECYCLECORECONSTRAINT_H
#define LOOM_PNR_CPSAT_SPATIALHANDSHAKECYCLECORECONSTRAINT_H

#include "SpatialBindingRelationModel.h"

#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialHandshakeSupplyDeficit.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr::detail {

struct SpatialHandshakeCycleCoreConstraintResult final {
  /// The class the core named, empty when it named none.
  SpatialHandshakeCoreCoPlacement coPlacement;
  /// Whether the escape clause entered the model.
  bool encoded = false;
  /// The decisions of the co-placement class that the region can still move.
  std::uint64_t classDecisionCount = 0;
  /// Legal choices of those decisions that reach a Temporal PE ingress or a
  /// neighbourhood a buffered mesh FIFO separates from the core.
  std::uint64_t escapingChoiceCount = 0;
  /// Distinct PE occurrences the escaping legal choices land on. One target
  /// shared by every escape is a different supply question from many.
  std::uint64_t escapeTargetPeCount = 0;
  /// Whether some decision of the class lies outside the bounded region, so
  /// the class cannot be stated here.
  bool outsideRegion = false;
};

/// States the co-placement class a recurring cycle core names, in the relation
/// model's own vocabulary: the compute realizations of the core's actors may
/// not all be bound to FU occurrences of the neighbourhood the core's fragments
/// name. At least one must leave it, which is the only placement change that
/// reaches an isolation point, because a Temporal PE ingress isolates a value
/// leaving its own PE and a buffered mesh FIFO isolates a route leaving the
/// tile. Every escaping assignment stays admissible, so the clause excludes no
/// placement that could still open the cycle. Nothing is stated when the
/// restart has no established core, when the class reaches a decision the
/// bounded region pins, or when no legal choice escapes.
llvm::Expected<SpatialHandshakeCycleCoreConstraintResult>
stateSpatialHandshakeCycleCoreClass(
    operations_research::sat::CpModelBuilder &model,
    const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<operations_research::sat::IntVar> variables,
    llvm::ArrayRef<int> decisionVariables,
    llvm::ArrayRef<PnrIndex> legalValueOffsets,
    llvm::ArrayRef<std::int64_t> legalValues,
    const SpatialHandshakeCycleCore &core);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_CPSAT_SPATIALHANDSHAKECYCLECORECONSTRAINT_H
