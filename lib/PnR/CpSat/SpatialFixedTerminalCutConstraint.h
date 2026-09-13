#ifndef LOOM_PNR_CPSAT_SPATIALFIXEDTERMINALCUTCONSTRAINT_H
#define LOOM_PNR_CPSAT_SPATIALFIXEDTERMINALCUTCONSTRAINT_H

#include "SpatialBindingRelationModel.h"
#include "SpatialLocalDispositionModel.h"

#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPathFinderRouter.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr::detail {

struct SpatialFixedTerminalCutConstraintResult final {
  bool encoded = false;
  bool currentAssignmentEscapes = false;
};

/// Admits one certificate into a repair's learned set, in the set's own
/// canonical order: forced net cuts sorted and deduplicated, certificates
/// ordered by capacity and then by that cut sequence. A certificate the set
/// already holds is refused, which is how the repair recognises negotiated
/// routing repeating an active cut.
bool insertSpatialFixedTerminalCutCertificate(
    std::vector<SpatialFixedTerminalCutCertificate> &certificates,
    SpatialFixedTerminalCutCertificate certificate);

llvm::Expected<SpatialFixedTerminalCutConstraintResult>
addSpatialFixedTerminalCutEscapeConstraint(
    operations_research::sat::CpModelBuilder &model,
    const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<operations_research::sat::IntVar> variables,
    llvm::ArrayRef<int> decisionVariables,
    llvm::ArrayRef<PnrIndex> legalValueOffsets,
    llvm::ArrayRef<std::int64_t> legalValues,
    const SpatialLocalDispositionModel &localDispositions,
    const SpatialFixedTerminalCutCertificate &certificate,
    std::vector<std::uint8_t> &blockedTraversals,
    std::vector<std::uint8_t> &reachableEndpoints,
    std::vector<PnrIndex> &worklist);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_CPSAT_SPATIALFIXEDTERMINALCUTCONSTRAINT_H
