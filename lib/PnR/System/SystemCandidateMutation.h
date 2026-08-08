#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMCANDIDATEMUTATION_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMCANDIDATEMUTATION_H

#include "PnR/System/SystemAction.h"
#include "PnR/System/SystemCandidateState.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr::detail {

llvm::Expected<std::vector<SystemServiceTargetSelection>>
systemServiceTargetChoices(const SystemCandidateState &candidate,
                           PnrIndex context);

llvm::Expected<std::vector<::loom::fabric::FabricUsePatternRef>>
systemInstructionUsePatternChoices(const SystemCandidateState &candidate,
                                   PnrIndex use);

llvm::Expected<std::vector<::loom::fabric::FabricUsePatternRef>>
systemServiceUsePatternChoices(const SystemCandidateState &candidate,
                               PnrIndex use);

llvm::Expected<SystemCandidateStateHandle>
rebuildSystemCandidateWithServiceTarget(const SystemCandidateState &candidate,
                                        PnrIndex context, PnrIndex choice);

llvm::Expected<SystemCandidateStateHandle>
rebuildSystemCandidateWithInstructionUsePattern(
    const SystemCandidateState &candidate, PnrIndex use, PnrIndex choice);

llvm::Expected<SystemCandidateStateHandle>
rebuildSystemCandidateWithServiceUsePattern(
    const SystemCandidateState &candidate, PnrIndex use, PnrIndex choice);

llvm::Expected<SystemCandidateStateHandle>
rebuildSystemCandidateRoutes(const SystemCandidateState &candidate,
                             const SystemTransportRoutingAction &action,
                             std::uint64_t &endpointExpansions,
                             std::uint64_t &negotiationIterations);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMCANDIDATEMUTATION_H
