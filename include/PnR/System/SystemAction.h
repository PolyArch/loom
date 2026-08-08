#ifndef LOOM_PNR_SYSTEM_SYSTEMACTION_H
#define LOOM_PNR_SYSTEM_SYSTEMACTION_H

#include "Common/ResolvedPnrPolicy.h"
#include "PnR/DeterministicSearchProtocol.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom::pnr {

struct SystemExecutionBindingAction final {
  PnrIndex decision = 0;
  PnrIndex choice = 0;
};

struct SystemActionProposalDomain final {
  llvm::ArrayRef<SystemExecutionBindingAction> bindingChoices;
};

llvm::Expected<std::optional<SystemExecutionBindingAction>>
proposeSystemAction(const ResolvedPnrActionProposalPolicy &policy,
                    SystemActionProposalDomain domain,
                    DeterministicPnrRandomStream &stream);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMACTION_H
