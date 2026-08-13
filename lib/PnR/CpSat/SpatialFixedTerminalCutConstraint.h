#ifndef LOOM_PNR_CPSAT_SPATIALFIXEDTERMINALCUTCONSTRAINT_H
#define LOOM_PNR_CPSAT_SPATIALFIXEDTERMINALCUTCONSTRAINT_H

#include "SpatialBindingRelationModel.h"

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

llvm::Expected<SpatialFixedTerminalCutConstraintResult>
addSpatialFixedTerminalCutEscapeConstraint(
    operations_research::sat::CpModelBuilder &model,
    const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<operations_research::sat::IntVar> variables,
    llvm::ArrayRef<int> decisionVariables,
    llvm::ArrayRef<PnrIndex> legalValueOffsets,
    llvm::ArrayRef<std::int64_t> legalValues,
    const SpatialFixedTerminalCutCertificate &certificate,
    std::vector<std::uint8_t> &blockedTraversals,
    std::vector<std::uint8_t> &reachableEndpoints,
    std::vector<PnrIndex> &worklist);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_CPSAT_SPATIALFIXEDTERMINALCUTCONSTRAINT_H
