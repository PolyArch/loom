#ifndef LOOM_PNR_SYSTEM_SYSTEMCANDIDATESERVICERESOLVER_H
#define LOOM_PNR_SYSTEM_SYSTEMCANDIDATESERVICERESOLVER_H

#include "PnR/System/SystemCandidateState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom::pnr::detail {

llvm::Expected<SystemServiceTargetDomain>
resolveSystemServiceTargetDomain(const FrozenSystemPnrProblem &problem,
                                 PnrIndex context,
                                 llvm::ArrayRef<PnrIndex> threadChoices,
                                 llvm::ArrayRef<PnrIndex> graphChoices);

llvm::Expected<std::vector<PnrIndex>>
resolveSystemServiceTerminalDomain(const FrozenSystemPnrProblem &problem,
                                   PnrIndex leg, PnrIndex terminal,
                                   llvm::ArrayRef<PnrIndex> threadChoices,
                                   llvm::ArrayRef<PnrIndex> graphChoices);

llvm::Error
verifySystemServiceTargetDomains(const FrozenSystemPnrProblem &problem,
                                 llvm::ArrayRef<PnrIndex> threadChoices,
                                 llvm::ArrayRef<PnrIndex> graphChoices);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_SYSTEM_SYSTEMCANDIDATESERVICERESOLVER_H
