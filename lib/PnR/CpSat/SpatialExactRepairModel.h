#ifndef LOOM_LIB_PNR_CPSAT_SPATIALEXACTREPAIRMODEL_H
#define LOOM_LIB_PNR_CPSAT_SPATIALEXACTREPAIRMODEL_H

#include "PnR/SpatialCandidateState.h"
#include "SpatialBindingRelationModel.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace loom::pnr::detail {

llvm::Expected<std::optional<std::string>>
admitAtomicExactRepairModel(const SpatialBindingRelationModel &bindings,
                            llvm::ArrayRef<PnrIndex> decisions,
                            llvm::ArrayRef<std::uint64_t> contextOveruse);

llvm::Expected<std::optional<std::string>>
admitTransportExactRepairModel(const SpatialBindingRelationModel &bindings,
                               llvm::ArrayRef<PnrIndex> decisions,
                               llvm::ArrayRef<std::uint64_t> contextOveruse);

llvm::Expected<std::uint64_t>
countExactRepairRegionDecisions(llvm::ArrayRef<PnrIndex> decisions,
                                llvm::ArrayRef<PnrIndex> affectedNets,
                                const FrozenSpatialPnrProblem &problem);

llvm::Expected<PnrIndex>
currentExactRepairBindingChoice(const SpatialCandidateState &candidate,
                                const SpatialBindingRelationModel &bindings,
                                PnrIndex decision);

llvm::Expected<operations_research::sat::IntVar>
addExactRepairMutationCountObjective(
    operations_research::sat::CpModelBuilder &model,
    llvm::ArrayRef<operations_research::sat::IntVar> variables,
    const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<PnrIndex> decisions,
    llvm::ArrayRef<int> mutationParentLocals = {},
    llvm::ArrayRef<operations_research::sat::IntVar> additionalVariables = {},
    llvm::ArrayRef<std::int64_t> additionalCurrentValues = {});

void addExactRepairInitializerRelationConstraint(
    operations_research::sat::CpModelBuilder &model,
    const InitializerRelationModel &relationModel,
    const InitializerRelationRecord &record,
    llvm::ArrayRef<operations_research::sat::IntVar> projections);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_CPSAT_SPATIALEXACTREPAIRMODEL_H
